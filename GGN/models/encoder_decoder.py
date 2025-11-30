import math
import os
import torch
import torch.nn.functional as F
from torch import nn

# Local imports
import eeg_util
from eeg_util import DLog
from graph_conv_layer import *

class GATLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=1, dropout=0.1):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.attn_weight = nn.Parameter(torch.Tensor(input_dim, num_heads, output_dim))
        self.attn_bias = nn.Parameter(torch.Tensor(num_heads, output_dim))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.attn_weight)
        nn.init.zeros_(self.attn_bias)

    def forward(self, x, adj):
        q = torch.matmul(x, self.attn_weight)  
        k = q.transpose(-2, -3)  
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.output_dim ** 0.5)
        attention_scores = attention_scores + adj.unsqueeze(0).unsqueeze(1)  
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        v = torch.matmul(x, self.attn_weight)  
        out = torch.matmul(attention_weights, v)  
        out = out.view(out.size(0), out.size(1), -1)  
        return out, attention_weights.mean(dim=1)  

def conv_L(in_len, kernel, stride, padding=0):
    return int((in_len - kernel + 2 * padding) / stride + 1)

def cal_cnn_outlen(modules, in_len, height=True):
    conv_l = in_len
    pos = 0 if height else 1
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            conv_l = conv_L(in_len, m.kernel_size[pos], m.stride[0], m.padding[pos])
            in_len = conv_l
        if isinstance(m, nn.AvgPool2d) or isinstance(m, nn.MaxPool2d):
            conv_l = conv_L(in_len, m.kernel_size[pos], m.stride, m.padding)
            in_len = conv_l                
    return conv_l

class CNN2d(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, width, height, kernel=(3, 3), stride=1, layers=2, dropout=0.6, pooling=False):
        super(CNN2d, self).__init__()
        self.dropout = dropout
        self.pooling = pooling
        b1_dim = int(hid_dim/2)

        self.b1 = self.cnn_block(in_dim, b1_dim)
        if pooling:
            self.pool1 = nn.MaxPool2d((3,3), 2)

        self.bx = nn.ModuleList()
        for _ in range(layers-2):
            self.bx.append(self.cnn_block(b1_dim, hid_dim, kernel, stride))

        self.bn = self.cnn_block(hid_dim, b1_dim, kernel, stride)

        if pooling:
            self.pool2 = nn.AvgPool2d((2,2), 3)
            
        self.len_h = cal_cnn_outlen(self.modules(), height)
        self.len_w = cal_cnn_outlen(self.modules(), width, False)

        DLog.log('CNNEncoder2d out len:', self.len_h, self.len_w)
        
        self.l1 = nn.Linear(b1_dim * self.len_h * self.len_w, out_dim)
        
    def cnn_block(self, in_dim, out_dim, kernel=(3, 3), stride=1):
        block = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=(1,1)),
                  nn.ReLU(),
                  nn.Conv2d(out_dim, out_dim, kernel_size=kernel, stride=stride),
                  nn.ReLU(),
                  nn.BatchNorm2d(out_dim),
                  nn.Dropout(self.dropout))
        return block

    def forward(self, x):
        DLog.debug('conv2d in', x.shape)
        x = self.b1(x)
        if self.pooling:
            x = self.pool1(x)
        for b in self.bx:
            x = b(x)
        x = self.bn(x)
        if self.pooling:
            x = self.pool2(x)

        x = self.l1(torch.flatten(x, start_dim=1))
        return x

class CNN1d(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, height, kernel=3, 
                    stride=1, layers=2, dropout=0.6,tag=None, linear=False):
        super(CNN1d, self).__init__()
        self.dropout = dropout
        self.linear = linear
        b1_dim = int(hid_dim/2)

        self.b1 = self.cnn_block(in_dim, b1_dim)
        self.bx = nn.ModuleList()
        for _ in range(layers-2):
            self.bx.append(self.cnn_block(b1_dim, hid_dim, kernel, 2))

        self.bn = self.cnn_block(hid_dim, out_dim, kernel, stride)
        self.width_len = cal_cnn_outlen(self.modules(), height)

        DLog.log(f'{tag}: CNNEncoder out len:', self.width_len)
        
        if self.linear:
            self.l1 = nn.Linear(out_dim * self.width_len, out_dim)
        
    def cnn_block(self, in_dim, out_dim, kernel=3, stride=1):
        block = nn.Sequential(nn.Conv1d(in_dim, out_dim, kernel_size=1),
                  nn.ReLU(),
                  nn.Conv1d(out_dim, out_dim, kernel_size=kernel, stride=stride),
                  nn.ReLU(),
                  nn.BatchNorm1d(out_dim),
                  nn.Dropout(self.dropout))
        return block

    def forward(self, x):
        x = self.b1(x)
        for b in self.bx:
            x = b(x)
        x = self.bn(x)
        if self.linear:
            x = self.l1(torch.flatten(x, start_dim=1))
        return x

class LSTMEncoder(nn.Module):
    def __init__(self, args, in_dim, hid_dim, out_dim, bidirect=False):
        super(LSTMEncoder, self).__init__()
        self.rnn = nn.LSTM(input_size=in_dim,
                          hidden_size=hid_dim,
                          num_layers=2,
                          batch_first=True,
                          bidirectional=bidirect)
    
    def forward(self, x):
        B,C,N,T = x.shape
        x = x.transpose(1, 2).reshape(B*N, C, T).transpose(1, 2)
        x, h = self.rnn(x, None)
        x = x.reshape(B, N, T, -1).transpose(2, 3)
        return x

class RNNEncoder(nn.Module):
    def __init__(self, args, in_dim, hid_dim, bidirect=False):
        super(RNNEncoder, self).__init__()
        self.rnn = nn.GRU(input_size=in_dim,
                          hidden_size=hid_dim,
                          num_layers=2,
                          batch_first=True,
                          bidirectional=bidirect)
    
    def forward(self, x):
        B,C,N,T = x.shape
        x = x.transpose(1, 2).reshape(B*N, C, T).transpose(1, 2) 
        x, h = self.rnn(x, None)
        x = x.reshape(B, N, T, -1).transpose(2, 3) 
        return x
        
class TemporalTransformerEncoder(nn.Module):
    def __init__(self, args, in_dim, hid_dim):
        super(TemporalTransformerEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.input_proj = nn.Linear(in_dim, hid_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=2, 
            dropout = args.dropout,
            dim_feedforward=hid_dim * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
    def forward(self, x):
        B, C, N, T = x.shape
        x = x.transpose(1, 2).reshape(B*N, C, T).transpose(1, 2)
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.reshape(B, N, T, -1).transpose(2, 3)
        return x

class MultiEncoders(nn.Module):
    def __init__(self, args, in_dim, hid_dim, out_dim):
        super(MultiEncoders, self).__init__()
        self.encoder1 = MultiCNNEncoder(1, args, in_dim, hid_dim, out_dim,
                                        height=34, kernel=3, stride=1, layers=3, tag="Encoder1", linear=True)
        self.encoder2 = MultiCNNEncoder(1, args, in_dim, hid_dim, out_dim,
                                        height=34, kernel=5, stride=2, layers=3,tag="Encoder2",linear=True)
        self.width_len = self.encoder1.width_len + self.encoder2.width_len
        
    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x)
        return torch.cat([x1, x2], dim=-1)
    
class MultiCNNEncoder(nn.Module):
    def __init__(self, cnn_num, args, in_dim, hid_dim, out_dim, height, kernel=3, 
                 stride=1, layers=2,tag=None, linear=False):
        super(MultiCNNEncoder, self).__init__()
        self.cnn_num = cnn_num
        self.cnns = nn.ModuleList()
        self.shared = True
        self.linear = linear
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        for _ in range(self.cnn_num):
            self.cnns.append(CNN1d(in_dim, hid_dim, out_dim, height, kernel,
                                          stride, layers, args.dropout,tag=tag, linear=linear))
        self.width_len = self.cnns[-1].width_len
        
    def forward(self, x):
        B,C,N,T = x.shape
        DLog.debug('mul in shape:', x.shape)
        node_emb = []
        
        if self.shared:
            x = x.transpose(1, 2).reshape(B*N, C, T)
            for i, cnn in enumerate(self.cnns):
                node_emb_tmp = cnn(x)
                node_emb.append(node_emb_tmp)
            node_embs = torch.stack(node_emb, dim=0).max(dim=0).values
            if self.linear:
                node_embs = node_embs.reshape(B, N, -1)
            else:
                node_embs = node_embs.reshape(B, N, self.out_dim, -1)
        else:
            assert N == len(self.cnns)
            for i, cnn in enumerate(self.cnns):
                node_emb.append(cnn(x[:,:,i,:])) 
            node_embs = torch.stack(node_emb, dim=1)  
            
        return node_embs
        
class GNNDecoder(nn.Module):
    def __init__(self, N, args, in_dim, out_dim):
        super(GNNDecoder, self).__init__()
        self.gnns = nn.ModuleList()
        self.args = args
        self.N = N
        if args.gnn_downsample_dim > 0:
            self.downsample = nn.Linear(in_dim, args.gnn_downsample_dim)
            self.gnn_in_dim = args.gnn_downsample_dim
        else:
            self.gnn_in_dim = in_dim
            self.downsample = None
            
        self.gnns.append(GraphConv(N, self.gnn_in_dim, args.gnn_hid_dim, args.dropout))
        for _ in range(args.gnn_layer_num-2):
            self.gnns.append(GraphConv(N, args.gnn_hid_dim, args.gnn_hid_dim, args.dropout))
        self.gnns.append(GraphConv(N, args.gnn_hid_dim, out_dim, args.dropout))

        self.pooling = args.gnn_pooling not in ['None','0',0,'none']
        if args.gnn_pooling == 'att':
            self.g_pooling = AttGraphPooling(args, N, out_dim, out_dim)
        elif args.gnn_pooling == 'cpool':
            K = 3
            self.g_pooling = CompactPooling(args, K, N)
        elif args.gnn_pooling == 'cat':
            self.g_pooling = CatPooling()
        else:
            self.g_pooling = GateGraphPooling(args, N)
        
        self.adj_w = nn.Parameter(torch.Tensor(N, N).cuda()) 
        self.reset_parameters()

    def forward(self, adj, x):
        B = x.shape[0]

        if self.downsample is not None:
            x = self.downsample(x)

        if adj is None:
            adj = self.adj_w
        origin_x = x
        for gnn in self.gnns:
            x = gnn(x, adj)
        
        if self.pooling:
            x = self.g_pooling(x)
            x = x.reshape(B, -1)

        if self.args.gnn_res:
            x = torch.cat([x, origin_x], dim=2)
        DLog.debug('gnn decoder out shape:', x.shape)
        return x

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.adj_w)
        
class SpatialDecoder(nn.Module):
    def __init__(self, args, gnn_decoder=None, cnn_decoder=None):
        super(SpatialDecoder, self).__init__()
        self.args = args
        self.gnn_decoder = gnn_decoder
        self.cnn_decoder = cnn_decoder

    def forward(self, adj, x):
        if isinstance(adj, list):
            x1 = []
            for t in range(len(adj)):
                xt = self.gnn_decoder(adj[t], x[:,t, :, :])
                x1.append(xt)
            x1 = torch.stack(x1, dim=1)
            x1 = torch.flatten(x1, start_dim=1)
        else:
            x1 = self.gnn_decoder(adj, x[:,-1, :, :]) 
        
        DLog.debug('DecoderAdapter x1 shape:', x1.shape)
        if self.cnn_decoder is not None:
            x = x.transpose(3, 1)
            x2 = self.cnn_decoder(x)
            DLog.debug('DecoderAdapter x2 shape:', x2.shape)
            x = torch.cat([x1, x2], dim=1)
        DLog.debug('SpatialDecoder out shape:', x.shape)
        return x

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, out_put, electrode_positions=None, pos_encoding_type='learnable',
                 num_heads=8, num_layers=2, dim_feedforward=2048):
        super(TransformerEncoderModel, self).__init__()

        self.pos_encoding_type = pos_encoding_type
        if electrode_positions is not None:
            self.register_buffer('positions', electrode_positions)  # (N, 3)

        if pos_encoding_type == 'learnable':
            self.position_embedding = nn.Linear(3, input_dim)
        elif pos_encoding_type == 'sincos':
            self.position_embedding = SinusoidalPositionalEncoding3D(input_dim)
        elif pos_encoding_type == 'rope':
            self.rope = RotaryPositionalEncoding3D(input_dim)
        
        self.input_norm = nn.LayerNorm(input_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.pool_weights = nn.Parameter(torch.Tensor(1, 1, input_dim))
        nn.init.xavier_normal_(self.pool_weights)
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_put)
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.input_norm(x)

        if hasattr(self, 'positions'):
            if self.pos_encoding_type == 'learnable':
                pos_emb = self.position_embedding(self.positions)  
                x = x + pos_emb.unsqueeze(0)
            elif self.pos_encoding_type == 'sincos':
                pos_emb = self.position_embedding(self.positions)  
                x = x + pos_emb.unsqueeze(0)
            elif self.pos_encoding_type == 'rope':
                x = self.rope(x, self.positions)
        
        x = self.transformer_encoder(x)  
        weights = F.softmax(self.pool_weights, dim=-1)  
        x = torch.sum(x * weights, dim=1)  
        return self.fc(x)

class transformer_cnn(nn.Module):
    def __init__(self, args, transformer_decoder=None, cnn_decoder=None):
        super(transformer_cnn, self).__init__()
        self.args = args
        self.transformer_decoder = transformer_decoder
        self.cnn_decoder = cnn_decoder
    
    def forward(self,adj, x):
        x1 = x[:, -1, :, :]
        x1 = self.transformer_decoder(x1)
        x = x.transpose(3, 1)
        x2 = self.cnn_decoder(x)
        x = torch.cat([x1, x2], dim=1)
        return x

class SinusoidalPositionalEncoding3D(nn.Module):
    """3D Sine-Cosine Positional Encoding"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.dims_per_coord = [d_model // 3 + (1 if i < d_model % 3 else 0) for i in range(3)]
        
        self.div_terms = []
        for i, dim in enumerate(self.dims_per_coord):
            div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim))
            self.div_terms.append(div_term.unsqueeze(0))
            self.register_buffer(f'div_term_{i}', div_term.unsqueeze(0)) 

    def forward(self, positions):
        pe = torch.zeros(positions.size(0), self.d_model)
        position = positions.float()
        
        start = 0
        for i in range(3):
            dim = self.dims_per_coord[i]
            end = start + dim
            pe[:, start:end] = self._encode_dim(position[:, i], self.div_terms[i], dim)
            start = end
            
        return pe 

    def _encode_dim(self, coord, div_term, dim):
        pe = torch.zeros(coord.size(0), dim)
        scaled_coord = coord.unsqueeze(-1) * div_term.to(coord.device)
        pe[..., 0::2] = torch.sin(scaled_coord[..., :dim//2])
        if dim % 2 == 0:
            pe[..., 1::2] = torch.cos(scaled_coord[..., :dim//2])
        else:
            pe[..., 1::2] = torch.cos(scaled_coord[..., :dim//2])[..., :-1]
        return pe

class RotaryPositionalEncoding3D(nn.Module):
    """3D Rotary Positional Encoding (RoPE)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        base_dim = dim // 3
        remainder = dim % 3
        self.dims = [base_dim + (1 if i < remainder else 0) for i in range(3)]
        
        self.inv_freq_x = 1.0 / (10000 ** (torch.arange(0, self.dims[0], 2).float() / self.dims[0]))
        self.inv_freq_y = 1.0 / (10000 ** (torch.arange(0, self.dims[1], 2).float() / self.dims[1]))
        self.inv_freq_z = 1.0 / (10000 ** (torch.arange(0, self.dims[2], 2).float() / self.dims[2]))

    def _apply_rotary(self, x, positions):
        B, N, D = x.shape
        x_coord = positions[:, 0].unsqueeze(-1)  
        y_coord = positions[:, 1].unsqueeze(-1)  
        z_coord = positions[:, 2].unsqueeze(-1)  
        
        freqs_x = torch.einsum("n,d->nd", x_coord, self.inv_freq_x) 
        freqs_y = torch.einsum("n,d->nd", y_coord, self.inv_freq_y) 
        freqs_z = torch.einsum("n,d->nd", z_coord, self.inv_freq_z) 
        
        emb = torch.cat([
            freqs_x.repeat(1, 2),  
            freqs_y.repeat(1, 2),
            freqs_z.repeat(1, 2)
        ], dim=-1)[:, :self.dim] 
        
        cos = torch.cos(emb).unsqueeze(0).unsqueeze(2) 
        sin = torch.sin(emb).unsqueeze(0).unsqueeze(2) 
        
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated * sin)

    def forward(self, x, positions):
        return self._apply_rotary(x, positions)