import argparse
import pickle
import sys, os
from collections import Counter
import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch import nn
import torch.nn.functional as F

DEVICE=torch.device('cuda' if torch.cuda.is_available else 'cpu')

def set_grad(m, requires_grad):
    for p in m.parameters():
        p.requires_grad = requires_grad

def freeze_module(m):
    set_grad(m, False)

def unfreeze_module(m):
    set_grad(m, True)

class DaoLogger:
    def __init__(self) -> None:
        self.debug_mode = False
        self.log_file = None

    def init(self, args, log_file_path=None):
        self.args = args
        self.debug_mode = args.debug
        if log_file_path:
            self.log_file_path = log_file_path
            self.log_file = open(log_file_path, 'a')
            self.terminal = sys.stdout
            sys.stdout = self

    def _write_to_file(self, message):
        if self.log_file_path:
            self.log_file.write(message + '\n')
            self.log_file.flush()

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        if self.log_file:
            self._write_to_file(message.strip())

    def flush(self):
        pass 

    def log(self, *paras):
        message = '[DLOG] ' + ' '.join(map(str, paras))
        print(message)

    def debug(self, *paras):
        if self.debug_mode:
            message = '[DEBUG] ' + ' '.join(map(str, paras))
            print(message)

    def close(self):
        if self.log_file:
            self.log_file.close()
        sys.stdout = self.terminal  

DLog = DaoLogger()
     
def normalize(data, fill_zeroes=True):
    '''
    Only norm numpy type data with last dimension.
    '''
    mean = np.mean(data)
    std = np.std(data)
    if fill_zeroes:
        mask = (data == 0)
        data[mask] = mean
    return (data - mean) / std

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def sym_norm_lap(adj):
    N = adj.shape[0]
    adj_norm = sym_adj(adj)
    L = np.eye(N) - adj_norm
    return L

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    """
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt).toarray()
    normalized_laplacian = sp.eye(adj.shape[0]) - np.matmul(np.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def generate_rand_adj(p, N):
    return np.random.rand(N, N)

def load_eeg_adj(adj_filename, adjtype=None):
    if 'npy' in adj_filename:
        adj = np.load(adj_filename)
    else:
        adj = np.genfromtxt(adj_filename, delimiter=',')
    adj_mx = np.asarray(adj)
    if adjtype in ["scalap", 'laplacian']:
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "sym_norm_lap":
        adj = [sym_norm_lap(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adjtype == "origin":
        return adj_mx
    return adj[0]

def calc_eeg_accuracy(preds, labels):
    """
    Accuracy calculation
    """
    num = preds.size(0)
    preds_b = preds.argmax(dim=1).squeeze()
    labels = labels.squeeze()
    ones = torch.zeros(num)
    ones[preds_b == labels] = 1
    acc = torch.sum(ones) / num
    return acc

def calc_metrics_eeg(preds, labels, criterion):
    labels = labels.squeeze()
    loss = criterion(preds, labels)
    return loss

class Trainer:
    def __init__(self, args, model, optimizer=None, scaler=None, criterion=nn.MSELoss(), sched=None):
        self.model = model
        self.args = args
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.clip = args.clip
        self.lr_decay_rate = args.lr_decay_rate
        self.epochs = args.epochs
        self.scheduler = sched

    def lr_schedule(self):
        self.scheduler.step()

    def train(self, input_data, target, epoch=-1):
        self.model.train()
        self.optimizer.zero_grad()

        output, vae_loss = self.model(input_data, 'train')
        output = output.squeeze()
        
        task_loss = calc_metrics_eeg(output, target, self.criterion)
        total_loss = task_loss + self.args.vae_weight * vae_loss 

        total_loss.backward(retain_graph=True)
        self.optimizer.step()

        torch.cuda.empty_cache()
        return task_loss.item(), output.detach(), vae_loss.item()

    def eval(self, input_data, target):
        self.model.eval()
        output, vae_loss = self.model(input_data)
        output = output.squeeze()
        loss = calc_metrics_eeg(output, target, self.criterion) 
        torch.cuda.empty_cache()
        return loss.item(), output.detach(), vae_loss.item()

class Trainer_simulation:
    def __init__(self, args, model, optimizer=None, scaler=None, criterion=nn.MSELoss(), sched=None):
        self.model = model
        self.args = args
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.clip = args.clip
        self.lr_decay_rate = args.lr_decay_rate
        self.epochs = args.epochs
        self.scheduler = sched

    def lr_schedule(self):
        self.scheduler.step()

    def forward_source(self, input_data):
        """Forward pass for source data."""
        output, kld_loss = self.model(input_data)
        return output, kld_loss
    
    def forward_est(self, input_data):
        """Forward pass for estimated data with dimension mapping."""
        input_data = input_data.permute(0, 1, 3, 2)
        mapped_input = self.model.map_layer(input_data)  # Map 198 to 66 dim
        mapped_input = mapped_input.permute(0, 1, 3, 2) 
        output, kld_loss= self.model(mapped_input)
        return output, kld_loss
        
    def train(self, input_data_source, input_data_est, epoch=-1):
        self.model.train()
        self.optimizer.zero_grad()

        source_output, kld_loss = self.forward_source(input_data_source)
        est_output, kld_loss = self.forward_est(input_data_est)

        recon_loss = self.criterion(source_output, est_output)
        total_loss = recon_loss + self.args.kld_weight * kld_loss
        
        total_loss.backward()
        self.optimizer.step()
        
        return (total_loss.item(), recon_loss.item(), kld_loss.item()), (source_output.detach(), est_output.detach())

    def eval(self, input_data_source, input_data_est):
        self.model.eval()
        source_output, kld_loss = self.forward_source(input_data_source)
        est_output, kld_loss = self.forward_est(input_data_est)

        recon_loss = self.criterion(source_output, est_output)
        total_loss = recon_loss + self.args.kld_weight * kld_loss

        return (total_loss.item(), recon_loss.item(), kld_loss.item()), (source_output.detach(), est_output.detach())