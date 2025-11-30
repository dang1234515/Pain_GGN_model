import os
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

# Local imports
import eeg_util
from eeg_util import DLog
from graph_conv_layer import *
from encoder_decoder import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class GGN(nn.Module):
    def __init__(self, adj, args, out_mid_features=False):
        super(GGN, self).__init__()
        self.args = args
        self.log = False
        self.adj_eps = 0.1
        self.adj = adj
        self.adj_x = adj
        # Add layer normalization
        self.norm = nn.LayerNorm(args.feature_len)

        self.N = adj.shape[0]
        print('N:', self.N)
        en_hid_dim = args.encoder_hid_dim
        en_out_dim = 16
        self.out_mid_features = out_mid_features
        self.pos = self.get_electrode_positions(self.args.pos)
        self.vae_loss = torch.tensor(0.0, dtype=torch.float32).cuda() 

        print(self.args.lgg)
        
        if args.encoder == 'rnn':
            self.encoder = RNNEncoder(args, args.feature_len, en_hid_dim, args.bidirect)
            decoder_in_dim = en_hid_dim
            if args.bidirect:
                decoder_in_dim *= 2
            de_out_dim = args.decoder_out_dim
        elif args.encoder == 'transformerencoder':
            self.encoder = TemporalTransformerEncoder(args, args.feature_len, en_hid_dim)
            decoder_in_dim = en_hid_dim
            de_out_dim = args.decoder_out_dim
        elif args.encoder == 'lstm':
            self.encoder = LSTMEncoder(args, args.feature_len, en_hid_dim, en_out_dim, args.bidirect)
            decoder_in_dim = en_hid_dim
            if args.bidirect:
                decoder_in_dim *= 2
            de_out_dim = args.decoder_out_dim
        elif args.encoder == 'cnn2d':
            cnn = CNN2d(in_dim=args.feature_len, 
                               hid_dim=en_hid_dim, 
                               out_dim=args.decoder_out_dim, 
                               width=34, height=self.N, stride=2, layers=3, dropout=args.dropout)
            self.encoder = cnn
            de_out_dim = args.decoder_out_dim
        else:
            self.encoder = MultiEncoders(args, args.feature_len, en_hid_dim, en_out_dim)
            decoder_in_dim = en_out_dim * 2
            de_out_dim = args.decoder_out_dim + decoder_in_dim

        if args.gnn_adj_type == 'rand':
            self.adj = None
            self.adj_tensor = None

        if args.lgg:
            if args.lgg_mode == 'VAE':
                self.LGG = LatentGraphGenerator_VAE(args, adj, decoder_in_dim, args.lgg_hid_dim)
            elif args.lgg_mode == 'MCMC':
                self.LGG = LatentGraphGenerator_MCMC(args, adj, args.lgg_tau, decoder_in_dim, args.lgg_hid_dim, args.lgg_k)
            elif args.lgg_mode == 'attention':
                self.LGG = nn.MultiheadAttention(embed_dim=decoder_in_dim, num_heads=4, batch_first=True)
            else:
                self.LGG = LatentGraphGenerator_gumble(args, adj, args.lgg_tau, decoder_in_dim, args.lgg_hid_dim, args.lgg_k)

        if args.decoder == 'gnn':
            if args.cut_encoder_dim > 0:
                decoder_in_dim *= args.cut_encoder_dim
            self.decoder = GNNDecoder(self.N, args, decoder_in_dim, de_out_dim)
            if args.agg_type == 'cat':
                de_out_dim *= self.N
        elif args.decoder == 'gat_cnn':
            self.adj_x = torch.ones((self.N, self.N)).float().cuda()
            print('gat adj_x: ', self.adj_x)
            g_pooling = GateGraphPooling(args, self.N)
            gnn = GAT(decoder_in_dim, args.gnn_hid_dim, de_out_dim, 
                               dropout=args.dropout, pooling=g_pooling)
            cnn_in_dim = decoder_in_dim
            cnn = CNN2d(cnn_in_dim, args.decoder_hid_dim, de_out_dim,
                            width=34, height=self.N, stride=2, layers=3, dropout=args.dropout)
            self.decoder = SpatialDecoder(args, gnn, cnn)
            de_out_dim *= 2
            
        elif args.decoder == 'lgg_cnn':
            gnn = GNNDecoder(self.N, args, decoder_in_dim, args.gnn_out_dim)
            cnn_in_dim = decoder_in_dim
            cnn = CNN2d(cnn_in_dim, args.decoder_hid_dim, de_out_dim,
                        width=self.args.time_len, height=self.N, stride=2, layers=3, dropout=args.dropout)
            self.decoder = SpatialDecoder(args, gnn, cnn)
            if args.agg_type == 'cat':
                de_out_dim += args.gnn_out_dim * self.N
            else:
                if args.lgg and args.lgg_time:
                    de_out_dim += args.gnn_out_dim * 34
                else:
                    de_out_dim += args.gnn_out_dim
        elif args.decoder == 'transformer_cnn':
            transformer_encoder = TransformerEncoderModel(decoder_in_dim, args.gnn_out_dim, 
                                                          electrode_positions=self.pos, pos_encoding_type=self.args.pe_type)
            cnn_in_dim = decoder_in_dim
            cnn = CNN2d(cnn_in_dim, args.decoder_hid_dim, de_out_dim,
                        width=self.args.time_len, height=self.N, stride=2, layers=3, dropout=args.dropout)
            self.decoder = transformer_cnn(args, transformer_encoder, cnn)
            de_out_dim += args.gnn_out_dim
        else:
            self.decoder = None
            de_out_dim = decoder_in_dim * self.N
            
        self.predictor = ClassPredictor(de_out_dim, hidden_channels=args.predictor_hid_dim,
                                class_num=args.predict_class_num, num_layers=args.predictor_num, dropout=args.dropout)
        
        self.warmup = args.lgg_warmup
        self.epoch = 0
        
        DLog.log('-------- encoder: -----------\n', self.encoder)
        DLog.log('-------- decoder: -----------\n', self.decoder)
        
        self.reset_parameters()
    
    def get_electrode_positions(self, pos):
        if pos is not None:
            position = np.load(pos)
            position = torch.tensor(position, dtype=torch.float32)
            return position
        else:
            return None
        
    def adj_to_coo_longTensor(self, adj):
        """adj is cuda tensor"""
        DLog.debug(adj)
        adj[adj > self.adj_eps] = 1
        adj[adj <= self.adj_eps] = 0

        idx = torch.nonzero(adj).T.long() # (row, col)
        DLog.debug('idx shape:', idx.shape)
        return idx

    def encode(self, x):
        # B,C,N,T = x.shape
        x = self.encoder(x)
        return x

    def fake_decoder(self, adj, x):
        DLog.debug('fake decoder in shape:', x.shape)
        # trans to BC:
        if len(x.shape) == 4:
            x = x[:,-1,...]
            
        if len(x.shape) > 2:
            x = torch.flatten(x, start_dim=1)
        
        return x

    def decode(self, x, B, N, adj):
        if self.decoder is None:
            x = self.fake_decoder(adj, x)
            DLog.debug('decoder out shape:', x.shape)
            return x

        if self.args.cut_encoder_dim > 0:
            x = x[:,:,:,-self.args.cut_encoder_dim:]

        x = self.decoder(adj, x)
        DLog.debug('decoder out shape:', x.shape)
        return x

    def alternative_freeze_grad(self, epoch):
        self.epoch = epoch
        if self.epoch > self.warmup: # Epoch exceeds warmup stage
            if epoch % 2==0:
                # freeze LGG
                eeg_util.freeze_module(self.LGG) 
                eeg_util.unfreeze_module(self.encoder) 
            else:
                # freeze encoder
                eeg_util.freeze_module(self.encoder) 
                eeg_util.unfreeze_module(self.LGG) 

    def forward(self, x, *options):
        """
        input x shape: B, C, N, T
        output x: class/regression value
        """
        B,C,N,T = x.shape

        # (1) encoder:
        x = self.encode(x)

        # before: BNCT
        x = x.permute(0, 3, 1, 2)
        # permute to BTNC

        # (2) adj selection:
        # LGG, latent graph generator:
        if self.args.lgg:
            if self.args.lgg_time:
                adj_x_times = []
                for t in range(T):
                    x_t = x[:, t, ...]
                    if self.training:
                        if self.epoch < self.warmup:
                            adj_x = self.LGG(x_t, self.adj)
                        else:
                            adj_x = self.LGG(x_t, self.adj)
                    else:
                        adj_x = self.LGG(x_t, self.adj)
                        DLog.debug('Model is Eval!')
                    adj_x_times.append(adj_x)
                self.adj_x = adj_x_times
            else:
                x_t = x[:, -1, ...]  # NOTE: take last time step. x_t: (B, N, C)
                if self.training and self.epoch < self.warmup:
                    if self.args.lgg_mode == 'VAE':
                        self.adj_x, self.vae_loss = self.LGG(x_t, self.adj)
                    elif self.args.lgg_mode == 'attention':
                        attn_output, attn_weights = self.LGG(x_t, x_t, x_t)
                        self.adj_x = attn_weights
                    else:
                        self.adj_x = self.LGG(x_t, self.adj)
                else:
                    if self.args.lgg_mode == 'VAE':
                        self.adj_x, self.vae_loss = self.LGG(x_t, self.adj)
                    elif self.args.lgg_mode == 'attention':
                        attn_output, attn_weights = self.LGG(x_t, x_t, x_t)
                        self.adj_x = attn_weights
                    else:
                        self.adj_x = self.LGG(x_t, self.adj)
                    DLog.debug('Model is Eval!')
                    DLog.debug('adj_x_shape:',self.adj.shape)

        # (3) decoder:
        DLog.debug('decoder input shape:', x.shape)
        x = self.decode(x, B, N, self.adj_x)
        DLog.debug('decoder output shape:', x.shape)

        if self.out_mid_features:
            return x
        
        # (4) predictor:
        x = self.predictor(x)
        if self.args.task == 'regression' and self.args.reg_normalized:
            x = torch.sigmoid(x)
        return x, self.vae_loss
    
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.count=0

class ClassPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, class_num, num_layers,
                 dropout=0.5):
        super(ClassPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        DLog.log('Predictor in channel:', in_channels)
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, class_num))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        DLog.debug('input prediction x shape:', x.shape)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

class LatentGraphGenerator_MCMC(nn.Module):
    def __init__(self, args, A_0, tau, in_dim, hid_dim, K=10, num_samples=1000):
        super(LatentGraphGenerator_MCMC, self).__init__()
        self.N = A_0.shape[0] # num of nodes
        self.args = args
        self.A_0 = A_0
        self.gumbel_tau = tau
        self.num_samples = num_samples

        if args.gnn_pooling == 'att':
            pooling = AttGraphPooling(args, self.N, in_dim, 64)
        elif args.gnn_pooling == 'cpool':
            pooling = CompactPooling(args, 3, self.N)
        elif args.gnn_pooling.upper() == 'NONE':
            pooling = None
        else:
            pooling = GateGraphPooling(args, self.N)

        # Multi-layer GNN for generating mu, sigma, pi parameters
        self.mu_nn = MultilayerGNN(self.N, 2, pooling, in_dim, hid_dim, K, args.dropout)
        self.sig_nn = MultilayerGNN(self.N, 2, pooling, in_dim, hid_dim, K, args.dropout)
        self.pi_nn = MultilayerGNN(self.N, 2, pooling, in_dim, hid_dim, K, args.dropout)

        # Trainable initial adjacency matrix
        self.adj_fix = nn.Parameter(self.A_0)

        print('adj_fix shape:', self.adj_fix.shape)

    def update_A_mcmc(self, mu, sig, pi, num_samples=1000):
        # Update adjacency matrix A using MCMC
        B, N, K = mu.shape
        A = torch.zeros(B, N, N).cuda()
        current_A = self.A_0 

        # Sampling for each node and component
        selected_mu = torch.zeros(B, N).cuda()
        selected_sig = torch.zeros(B, N).cuda()
        component_probs = torch.softmax(pi, dim=-1)  # (B, N, K)
            
        for k in range(K):
            selected_mu += component_probs[:, :, k] * mu[:, :, k]  
            selected_sig += component_probs[:, :, k] * sig[:, :, k] 

        for _ in range(num_samples):        
            noise = torch.randn_like(selected_mu)
            proposed_mu = selected_mu + noise * selected_sig 
            proposed_A = torch.sigmoid(torch.einsum('bnc, bmc -> bnm', proposed_mu.unsqueeze(-1), proposed_mu.unsqueeze(-1).transpose(2, 1)))  

            acceptance_ratio = self.compute_acceptance_ratio(current_A, proposed_A, pi)
            rand_vals = torch.rand(B).cuda()
            accept_mask = rand_vals < acceptance_ratio

            current_A = torch.where(accept_mask.unsqueeze(-1).unsqueeze(-1), proposed_A, current_A)
            A += current_A
            
        A /= num_samples
        return A

    def compute_acceptance_ratio(self, current_A, proposed_A, pi):
        """Calculate Metropolis-Hastings acceptance ratio."""
        current_prob = self.calculate_log_likelihood(current_A, pi)
        proposed_prob = self.calculate_log_likelihood(proposed_A, pi)
        acceptance_ratio = torch.exp(proposed_prob - current_prob)
        acceptance_ratio = torch.clamp(acceptance_ratio, 0, 1)
        return acceptance_ratio.mean().item()

    def calculate_log_likelihood(self, A, pi):
        """Calculate log likelihood of adjacency matrix A."""
        B, N, K = pi.shape
        pi_expanded = pi.unsqueeze(-1).expand(-1, -1, -1, N) 
        edge_probs = torch.sum(pi_expanded, dim=2) 
        edge_probs = torch.clamp(edge_probs, min=1e-8, max=1.0)
        log_likelihood = torch.sum(A * torch.log(edge_probs) + (1 - A) * torch.log(1 - edge_probs), dim=(0, 1))  
        return log_likelihood
    
    def forward(self, x, adj_t=None):
        if adj_t is None:
            adj_t = self.adj_fix

        mu = self.mu_nn(x, adj_t)
        sig = self.sig_nn(x, adj_t)
        pi = self.pi_nn(x, adj_t)
        A = self.update_A_mcmc(mu, sig, pi)
        return A

class LatentGraphGenerator_VAE(nn.Module):
    def __init__(self, args, A_0, in_dim, hid_dim, K=1):
        super().__init__()
        self.N = A_0.shape[0]
        self.args = args
        self.kld_weight = args.kld_weight
        
        # Trainable initial adjacency matrix
        self.adj_fix = nn.Parameter(A_0)
        
        # Encoder: Spatial feature extraction
        self.encoder = nn.Sequential(
            GraphConv(self.N, in_dim, hid_dim*2, args.dropout),
            nn.LayerNorm(hid_dim*2),
            nn.ReLU(),
            nn.Linear(hid_dim*2, hid_dim)
        )
        
        # Mean and log variance generation
        self.mu_net = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hid_dim, self.N)
        )
        self.logvar_net = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hid_dim, self.N)
        )
        
        # Decoder: Bilinear attention mechanism
        self.bilinear = nn.Bilinear(self.N, self.N, self.N)
        self.act = nn.LeakyReLU(0.2)
            
    def decoder(self, x1, x2):
        x = self.bilinear(x1, x2)
        x = self.act(x)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, adj_t=None):
        B, N, C = x.shape
        h = self.encoder(x) 
        
        mu = self.mu_net(h)  
        logvar = self.logvar_net(h)  
        
        z = self.reparameterize(mu, logvar)
        
        A_logits = self.decoder(z, z)  
        A = torch.sigmoid(A_logits) 
        
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = self.kld_weight * kld_loss

        if self.args.task == 'simulation':
            return A, kld_loss
        
        return A, total_loss

class LatentGraphGenerator_gumble(nn.Module):
    def __init__(self, args, A_0, tau, in_dim, hid_dim, K=10):
        super(LatentGraphGenerator_gumble,self).__init__()
        self.N = A_0.shape[0] 
        self.args = args
        self.A_0 = A_0
        self.args = args

        if args.gnn_pooling == 'att':
            pooling = AttGraphPooling(args, self.N, in_dim, 64)
        elif args.gnn_pooling == 'cpool':
            pooling = CompactPooling(args, 3, self.N)
        elif args.gnn_pooling.upper() == 'NONE':
            pooling = None
        else:
            pooling = GateGraphPooling(args, self.N)
            
        self.gumbel_tau = tau
        self.mu_nn = MultilayerGNN(self.N, 2, pooling, in_dim, hid_dim, K, args.dropout)
        self.sig_nn = MultilayerGNN(self.N, 2, pooling, in_dim, hid_dim, K, args.dropout)
        self.pi_nn = MultilayerGNN(self.N, 2, pooling, in_dim, hid_dim, K, args.dropout)
        
        self.adj_fix = nn.Parameter(self.A_0)
        print('adj_fix shape:', self.adj_fix.shape)
        self.init_norm()

    def init_norm(self):
        self.Norm = torch.randn(size=(1000, self.args.batch_size, self.N)).cuda()
        self.norm_index = 0

    def get_norm_noise(self, size):
        if self.norm_index >= 999:
            self.init_norm()

        if size == self.args.batch_size:
            self.norm_index += 1
            noise = self.Norm[self.norm_index].squeeze()
        else:
            noise = torch.randn((size, self.N)).cuda()
        
        # Clip noise to reasonable range
        noise = torch.clamp(noise, min=-3.0, max=3.0)
        return noise
        
    def update_A(self, mu, sig, pi):
        # Add numerical stability check
        logits = torch.log_softmax(pi, dim=-1)
        pi_onehot = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=False, dim=-1)

        mu_k = torch.sum(mu * pi_onehot, dim=-1) 
        sig_k = torch.sum(sig * pi_onehot, dim=-1) 

        n = self.get_norm_noise(mu_k.shape[0]) 
        
        S = mu_k + n*sig_k
        S = S.unsqueeze(dim=-1)
        Sim = torch.einsum('bnc, bcm -> bnm', S, S.transpose(2, 1)) 

        P = torch.sigmoid(Sim)
        P = torch.clamp(P, min=1e-8, max=1.0-1e-8)
        pp = torch.stack((P, 1-P), dim=3)
        pp = torch.clamp(pp, min=1e-8, max=1.0)
        pp_logits = torch.log(pp)
        pp_onehot = F.gumbel_softmax(pp_logits, tau=self.gumbel_tau, hard=False, dim=-1)
        A = pp_onehot[:,:,:,0]
        return A

    def forward(self, x, adj_t=None):
        if adj_t is None:
            adj_t = self.adj_fix
        
        mu = self.mu_nn(x, adj_t)
        sig = self.sig_nn(x, adj_t)
        pi = self.pi_nn(x, adj_t)
        A = self.update_A(mu, sig, pi)
        return A

class GGN_simulation(nn.Module):
    def __init__(self, adj, args, out_mid_features=False):
        super(GGN_simulation, self).__init__()
        self.args = args
        self.log = False
        self.adj_eps = 0.1
        self.adj = adj
        self.adj_x = adj
        self.map_layer = nn.Linear(198,66)

        self.N = adj.shape[0]
        print('N:', self.N)
        en_hid_dim = args.encoder_hid_dim
        en_out_dim = 16
        self.out_mid_features = out_mid_features
        args.feature_len = 17

        self.kld_loss = torch.tensor(0.0, dtype=torch.float32).cuda()
        
        if args.encoder == 'rnn':
            self.encoder = RNNEncoder(args, args.feature_len, en_hid_dim, args.bidirect)
            decoder_in_dim = en_hid_dim
            if args.bidirect:
                decoder_in_dim *= 2
            de_out_dim = args.decoder_out_dim
        elif args.encoder == 'transformerencoder':
            self.encoder = TemporalTransformerEncoder(args, args.feature_len, en_hid_dim)
            decoder_in_dim = en_hid_dim
            de_out_dim = args.decoder_out_dim
        elif args.encoder == 'lstm':
            self.encoder = LSTMEncoder(args, args.feature_len, en_hid_dim, en_out_dim, args.bidirect)
            decoder_in_dim = en_hid_dim
            if args.bidirect:
                decoder_in_dim *= 2
            de_out_dim = args.decoder_out_dim
        elif args.encoder == 'cnn2d':
            cnn = CNN2d(in_dim=args.feature_len, 
                               hid_dim=en_hid_dim, 
                               out_dim=args.decoder_out_dim, 
                               width=34, height=self.N, stride=2, layers=3, dropout=args.dropout)
            self.encoder = cnn
            de_out_dim = args.decoder_out_dim
        else:
            self.encoder = MultiEncoders(args, args.feature_len, en_hid_dim, en_out_dim)
            decoder_in_dim = en_out_dim * 2
            de_out_dim = args.decoder_out_dim + decoder_in_dim

        if args.gnn_adj_type == 'rand':
            self.adj = None
            self.adj_tensor = None

        if args.lgg:
            if args.lgg_mode == 'VAE':
                self.LGG = LatentGraphGenerator_VAE(args, adj, decoder_in_dim, args.lgg_hid_dim)
            elif args.lgg_mode == 'MCMC':
                self.LGG = LatentGraphGenerator_MCMC(args, adj, args.lgg_tau, decoder_in_dim, args.lgg_hid_dim, args.lgg_k)
            else:
                self.LGG = LatentGraphGenerator_gumble(args, adj, args.lgg_tau, decoder_in_dim, args.lgg_hid_dim, args.lgg_k)

        self.warmup = args.lgg_warmup
        self.epoch = 0
        self.reset_parameters()

    def adj_to_coo_longTensor(self, adj):
        """adj is cuda tensor"""
        DLog.debug(adj)
        adj[adj > self.adj_eps] = 1
        adj[adj <= self.adj_eps] = 0
        idx = torch.nonzero(adj).T.long() 
        DLog.debug('idx shape:', idx.shape)
        return idx

    def encode(self, x):
        x = self.encoder(x)
        return x

    def forward(self, x, *options):
        B,C,N,T = x.shape
        # (1) encoder:
        x = self.encode(x)
        x = x.permute(0, 3, 1, 2)

        # (2) adj selection:
        # LGG, latent graph generator:
        if self.args.lgg:
            if self.args.lgg_time:
                adj_x_times = []
                for t in range(T):
                    x_t = x[:, t, ...]
                    if self.training:
                        if self.epoch < self.warmup:
                            adj_x = self.LGG(x_t, self.adj)
                        else:
                            adj_x = self.LGG(x_t, self.adj)
                    else:
                        adj_x = self.LGG(x_t, self.adj)
                    adj_x_times.append(adj_x)
                self.adj_x = adj_x_times
            else:
                x_t = x[:, -1, ...] 
                if self.training and self.epoch < self.warmup:
                    if self.args.lgg_mode == 'VAE':
                        self.adj_x, self.kld_loss = self.LGG(x_t, self.adj)
                    else:
                        self.adj_x = self.LGG(x_t, self.adj)
                else:
                    if self.args.lgg_mode == 'VAE':
                        self.adj_x, self.kld_loss = self.LGG(x_t, self.adj)
                    else:
                        self.adj_x = self.LGG(x_t, self.adj)
                    DLog.debug('Model is Eval!')
        
        return self.adj_x, self.kld_loss
       
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.count=0