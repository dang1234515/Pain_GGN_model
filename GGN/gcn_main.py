import pickle
import pandas as pd
from sklearn.metrics import r2_score
import collections
import numpy as np
import os, sys
import time
import argparse
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
from torch import optim
from torch import nn
import torch
import networkx as nx
import json

# Local imports
from ggn import GGN, GGN_simulation
from eeg_util import DLog, Trainer, Trainer_simulation
import eeg_util

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class SeqDataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=False):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size) + 1
        if (self.num_batch - 1) * self.batch_size == self.size:
            self.num_batch -= 1

        print('Num batches:', self.num_batch)
        xs = torch.Tensor(xs).to(DEVICE)
        self.xs = xs

        if args.task != 'regression':
            ys = torch.LongTensor(ys).to(DEVICE)
        else:
            ys = torch.tensor(ys, dtype=torch.float32).to(DEVICE)
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            start_ind = 0
            end_ind = 0
            while self.current_ind < self.num_batch and start_ind <= end_ind and start_ind <= self.size:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1
        return _wrapper()

class SeqDataLoader_reg(object):
    def __init__(self, xs, ys, labels, batch_size, pad_with_last_sample=False):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size) + 1
        if (self.num_batch - 1) * self.batch_size == self.size:
            self.num_batch -= 1

        print('Num batches:', self.num_batch)
        self.xs = torch.Tensor(xs).to(DEVICE)
        self.ys = torch.tensor(ys, dtype=torch.float32).to(DEVICE)
        self.labels = torch.LongTensor(labels).to(DEVICE)

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys, labels = self.xs[permutation], self.ys[permutation], self.labels[permutation]
        self.xs = xs
        self.ys = ys
        self.labels = labels

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            start_ind = 0
            end_ind = 0
            while self.current_ind < self.num_batch and start_ind <= end_ind and start_ind <= self.size:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                label_i = self.labels[start_ind: end_ind, ...]
                yield x_i, y_i, label_i
                self.current_ind += 1
        return _wrapper()

class SeqDataLoader_simulation(object):
    def __init__(self, xs, batch_size, pad_with_last_sample=False):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size) + 1
        if (self.num_batch - 1) * self.batch_size == self.size:
            self.num_batch -= 1

        print('Num batches:', self.num_batch)
        self.xs = torch.Tensor(xs).to(DEVICE)

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        self.xs = self.xs[permutation]

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            start_ind = 0
            end_ind = 0
            while self.current_ind < self.num_batch and start_ind <= end_ind and start_ind <= self.size:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                yield x_i
                self.current_ind += 1
        return _wrapper()

def load_pain_classification_data(args):
    feature = np.load(os.path.join(args.data_path, "pain_x.npy"))
    label = np.load(os.path.join(args.data_path, "pain_y.npy"))
    print('Loaded pain data, shape:', feature.shape, label.shape)

    if args.load_dataset:
        if args.cross_subject:
            train_indices = np.load('train_indices_all.npy')
            test_indices = np.load('test_indices_all.npy')
        else:
            train_indices = np.load(f'train_test_split/train_ind{args.data}.npy')
            test_indices = np.load(f'train_test_split/test_ind{args.data}.npy')

        np.random.shuffle(train_indices)    
        np.random.shuffle(test_indices)

        train_x, test_x = feature[train_indices], feature[test_indices]
        train_y, test_y = label[train_indices], label[test_indices]

        # Reshape to B, T, N, C -> B, C, N, T
        # Assuming input is (B, T, N, C)
        train_x = train_x.transpose(0, 3, 2, 1)
        test_x = test_x.transpose(0, 3, 2, 1)

        print('After transpose:', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        return [train_x, test_x], [train_y, test_y]

    # Legacy shuffle logic if not loading indices
    shuffled_index = np.random.permutation(np.arange(feature.shape[0]))
    feature = feature[shuffled_index]
    label = label[shuffled_index]
    
    # Simple split
    split = int(0.8 * len(feature))
    train_x, test_x = feature[:split], feature[split:]
    train_y, test_y = label[:split], label[split:]

    train_x = train_x.transpose(0, 3, 2, 1)
    test_x = test_x.transpose(0, 3, 2, 1)
    return [train_x, test_x], [train_y, test_y]

def load_pain_regression_data(args):
    feature = np.load(os.path.join(args.data_path, 'pain_x.npy'))
    category_label = np.load(os.path.join(args.data_path, 'pain_y.npy'))
    regression_label = np.load(os.path.join(args.data_path, f'pain_{args.score}.npy'))
    regression_label = regression_label.astype(np.float32)

    if args.reg_normalized:
        if args.score == 'VAS':
            regression_label = regression_label/10
        else:
            regression_label = regression_label/100
    
    print('Data loaded, shapes:', feature.shape, category_label.shape, regression_label.shape)

    if args.load_dataset:
        if args.cross_subject:
            train_indices = np.load('train_indices_reg.npy')
            test_indices = np.load('test_indices_reg.npy')
        else:
            train_indices = np.load('train_indices_reg_no_cross.npy')
            test_indices = np.load('test_indices_reg_no_cross.npy')

        # Filter for pain patients (assuming 0 and 1 are pain categories)
        train_indices = train_indices[np.isin(category_label[train_indices],[0, 1])]
        test_indices = test_indices[np.isin(category_label[test_indices],[0, 1])]

        train_x, test_x = feature[train_indices], feature[test_indices]
        train_y, test_y = regression_label[train_indices], regression_label[test_indices]
        train_label, test_label = category_label[train_indices], category_label[test_indices]

        # Transpose to (B, C, N, T)
        train_x = train_x.transpose(0, 3, 2, 1)
        test_x = test_x.transpose(0, 3, 2, 1)

        print('Transposed shapes:', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        return [train_x, test_x], [train_y, test_y], [train_label, test_label]

    # Stratified split if not using predefined indices
    unique_labels = np.unique(category_label)
    train_x, train_y, test_x, test_y = [], [], [], []
    train_label, test_label = [], []
    
    for label in unique_labels:
        indices = np.where(category_label == label)[0]
        np.random.shuffle(indices)
        split_index = int(0.7 * len(indices))
        train_indices, test_indices = indices[:split_index], indices[split_index:]
        
        train_x.append(feature[train_indices])
        train_y.append(regression_label[train_indices])
        train_label += [label] * len(train_indices)
        test_x.append(feature[test_indices])
        test_y.append(regression_label[test_indices])
        test_label += [label] * len(test_indices)
    
    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    train_label = np.array(train_label)
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)
    test_label = np.array(test_label)

    train_x = train_x.transpose(0, 3, 2, 1)
    test_x = test_x.transpose(0, 3, 2, 1)

    print('Transposed shapes (Random Split):', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    return [train_x, test_x], [train_y, test_y], [train_label, test_label]

def load_pain_simulation_data(args, source_name="", est_name=""):
    source_data = np.load(os.path.join(args.source_data_path, f"pain_x{source_name}.npy"))
    est_data = np.load(os.path.join(args.est_data_path, f"pain_x{est_name}.npy"))
    print('Loaded source and est data, shapes:', source_data.shape, est_data.shape)
    
    assert source_data.shape[0] == est_data.shape[0], "Source and Est sample counts mismatch"

    total_samples = source_data.shape[0]
    test_size = int(total_samples / 3)
    
    train_source = source_data[test_size:]
    test_source = source_data[:test_size]
    
    train_est = est_data[test_size:]
    test_est = est_data[:test_size]
    
    # Reshape (B, C, N, T)
    train_source = train_source.transpose(0, 3, 2, 1)
    test_source = test_source.transpose(0, 3, 2, 1)
    train_est = train_est.transpose(0, 3, 2, 1)
    test_est = test_est.transpose(0, 3, 2, 1)

    return [train_source, test_source], [train_est, test_est]

def normalize_features(features):
    """Inplace normalization"""
    for i in range(len(features)):
        # (B, C, N, T) - Norm over features dim (C is usually channel/freq, but normalize per sample/node/freq?)
        # Original code normalized last dimension after transpose? 
        # Here we assume (B, C, N, T), we iterate over T (last dim)
        for j in range(features[i].shape[-1]):
             features[i][..., j] = eeg_util.normalize(features[i][..., j])

def generate_dataloader(features, labels, args, reg_labels=None):
    cates = ['train', 'val', 'test']
    datasets = dict()

    if args.task == 'simulation':
        datasets = {'source': {}, 'est': {}}
        for i in range(len(features)):
            datasets['source'][f'{cates[i]}_loader'] = SeqDataLoader_simulation(features[i], args.batch_size)
        if len(features) < 3:
            datasets['source']['test_loader'] = datasets['source']['val_loader']

        for i in range(len(labels)):
            datasets['est'][f'{cates[i]}_loader'] = SeqDataLoader_simulation(labels[i], args.batch_size)
        if len(labels) < 3:
            datasets['est']['test_loader'] = datasets['est']['val_loader']
        return datasets

    if args.task == 'regression':
        for i in range(len(features)):
            datasets[cates[i] + '_loader'] = SeqDataLoader_reg(features[i], labels[i], reg_labels[i], args.batch_size)
        if len(features) < 3:
            datasets['test_loader'] = SeqDataLoader_reg(features[-1], labels[-1], reg_labels[-1], args.batch_size)
        return datasets 

    for i in range(len(features)):
        datasets[cates[i] + '_loader'] = SeqDataLoader(features[i], labels[i], args.batch_size)
    if len(features) < 3:
        datasets['test_loader'] = SeqDataLoader(features[-1], labels[-1], args.batch_size)

    return datasets

def init_adjs(args, index=0):
    adjs = []
    if args.adj_type == 'rand10':
        adj_mx = eeg_util.generate_rand_adj(0.1*(index+1), N=args.N)
    elif args.adj_type == 'er':
        adj_mx = nx.to_numpy_array(nx.erdos_renyi_graph(args.N, 0.1*(index+1)))
    else:
        adj_mx = eeg_util.load_eeg_adj(args.adj_file, args.adj_type)
    
    adj = torch.from_numpy(adj_mx).float().to(DEVICE)
    adjs.append(adj)
    return adjs

def chose_model(args, adjs):
    if args.task == 'classification' or args.task == 'regression':
        adj = adjs[0]
        model = GGN(adj, args)
    elif args.task == 'simulation':
        adj = adjs[0]
        model = GGN_simulation(adj, args)
    else:
        model = None
        print('No model found!')
    return model

def init_trainer(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    def lr_adjust(epoch):
        if epoch < 30:
            return 1
        return args.lr_decay_rate ** ((epoch - 19) / 3 + 1)

    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_adjust)
    
    if args.task == 'simulation':
        trainer = Trainer_simulation(args, model, optimizer, criterion=nn.MSELoss(), sched=lr_sched)
    elif args.task == 'regression':
        if args.reg_loss.upper() == 'HUBER':
            loss = nn.HuberLoss(delta=1.0, reduction='mean')
        else:
            loss = nn.MSELoss()
        trainer = Trainer(args, model, optimizer, criterion=loss, sched=lr_sched)
    else:
        # Classification Loss
        # Weights for imbalanced classes (Example)
        weights = None 
        criter = nn.CrossEntropyLoss(weight=weights)
        trainer = Trainer(args, model, optimizer, criterion=criter, sched=lr_sched)
    return trainer

def train_eeg(args, datasets, index=0):
    dt = time.strftime("%m_%d_%H_%M", time.localtime())
    log_dir = "./logs/"+args.server_tag+"/" + dt
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    adjs = init_adjs(args, index)
    model = chose_model(args, adjs)
    if args.cuda:
        model.cuda()

    trainer = init_trainer(model, args)
    
    best_val_acc = 0
    start_time = time.time()
    basedir, file_tag = os.path.split(args.best_model_save_path)
    model_save_path = os.path.join(basedir, file_tag)
    
    for e in range(args.epochs):
        datasets['train_loader'].shuffle()
        train_loss, train_loss_vae = [], []
        train_preds = []

        for i, (input_data, target) in enumerate(datasets['train_loader'].get_iterator()):
            loss, preds, vae_loss = trainer.train(input_data, target)
            train_loss.append(loss)
            train_loss_vae.append(vae_loss)
            train_preds.append(preds)
        
        val_loss, val_loss_vae = [], []
        val_preds = []
    
        for j, (input_data, target) in enumerate(datasets['val_loader'].get_iterator()):
            loss, preds, vae_loss  = trainer.eval(input_data, target)
            val_loss.append(loss)
            val_preds.append(preds)
            val_loss_vae.append(vae_loss)   

        train_preds = torch.cat(train_preds, dim=0)
        val_preds = torch.cat(val_preds, dim=0)
        
        train_acc = eeg_util.calc_eeg_accuracy(train_preds, datasets['train_loader'].ys).item()
        val_acc = eeg_util.calc_eeg_accuracy(val_preds, datasets['val_loader'].ys).item()

        if e % 5 == 0:
            print(f'Epoch {e}: Train Loss {np.mean(train_loss):.4f}, Val Acc {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print("Update best model, epoch: ", e)
            torch.save(trainer.model.state_dict(), model_save_path)

        trainer.lr_schedule()

    return best_val_acc

def train_eeg_reg(args, datasets, index=0):
    adjs = init_adjs(args, index)
    model = chose_model(args, adjs)
    if args.cuda:
        model.cuda()
    
    trainer = init_trainer(model, args)
    best_val_loss = float('inf')
    start_time = time.time()
    model_save_path = args.best_model_save_path
    
    for e in range(args.epochs):
        train_loss = []
        for i, (input_data, target, label) in enumerate(datasets['train_loader'].get_iterator()):
            loss, preds, _ = trainer.train(input_data, target)
            train_loss.append(loss)

        val_loss = []
        for j, (input_data, target, label) in enumerate(datasets['val_loader'].get_iterator()):
            loss, preds, _  = trainer.eval(input_data, target)
            val_loss.append(loss)

        curr_val_loss = np.mean(val_loss)
        if e % 5 == 0:
            print(f'Epoch {e}: Train Loss {np.mean(train_loss):.4f}, Val Loss {curr_val_loss:.4f}')

        if curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss
            print("Update best model, epoch: ", e)
            torch.save(model.state_dict(), model_save_path)
        
        trainer.lr_schedule()

    print('Training finished, time cost:', time.time() - start_time)
    
    # Testing
    test_model = chose_model(args, adjs)
    test_model.load_state_dict(torch.load(model_save_path))
    test_model.cuda()
    trainer.model = test_model

    test_loss, test_preds, test_targets = [], [], []

    for i, (input_data, target, label) in enumerate(datasets['test_loader'].get_iterator()):
        loss, preds, _ = trainer.eval(input_data, target)
        if isinstance(loss, torch.Tensor):
            test_loss.append(loss.detach().cpu().numpy())
        else:
            test_loss.append(loss)
        test_preds.append(preds.cpu().numpy())
        test_targets.append(target.cpu().numpy())

    test_targets = np.concatenate(test_targets, axis=0)
    test_preds = np.concatenate(test_preds, axis=0)

    if args.score == 'VAS':
        test_targets = test_targets * 10
        test_preds = test_preds * 10
    else:
        test_targets = test_targets * 100
        test_preds = test_preds * 100
   
    r2 = r2_score(test_targets, test_preds)
    print(f"Test R^2: {r2:.4f}, Test Loss: {np.mean(test_loss):.4f}")
    
    # Save results
    results_df = pd.DataFrame({'target': test_targets.flatten(), 'pred': test_preds.flatten()})
    results_df.to_excel('testing_predictions.xlsx', index=False)

    return np.mean(test_loss)

def train_eeg_simulation(args, datasets, index=0):
    adjs = init_adjs(args, index)
    model = chose_model(args, adjs)
    if args.cuda: model.cuda()

    trainer = init_trainer(model, args)
    best_val_loss = float('inf')
    model_save_path = f"{args.best_model_save_path}_{index}"
    
    for e in range(args.epochs):
        train_loss_mse = []
        for src, est in zip(datasets['source']['train_loader'].get_iterator(), datasets['est']['train_loader'].get_iterator()):
            (total, recon, kld), _ = trainer.train(src, est)
            train_loss_mse.append(recon)
        
        val_loss_mse = []
        for src, est in zip(datasets['source']['val_loader'].get_iterator(), datasets['est']['val_loader'].get_iterator()):
            (total, recon, kld), _ = trainer.eval(src, est)
            val_loss_mse.append(recon)

        curr_val_mse = np.mean(val_loss_mse)
        if e % 5 == 0:
            print(f'Epoch {e}: Train MSE {np.mean(train_loss_mse):.4f}, Val MSE {curr_val_mse:.4f}')

        if curr_val_mse < best_val_loss:
            best_val_loss = curr_val_mse
            print("Update best model")
            torch.save(trainer.model.state_dict(), model_save_path)

        trainer.lr_schedule()

    return best_val_loss

def multi_train(args):
    if args.task == 'simulation':
        source_data, est_data = load_pain_simulation_data(args)
        normalize_features(source_data)
        normalize_features(est_data)
        datasets = generate_dataloader(source_data, est_data, args)
        for i in range(args.runs):
            train_eeg_simulation(args, datasets, i)
        return

    if args.task == 'regression':
        xs, ys, labels = load_pain_regression_data(args)
        normalize_features(xs)
        datasets = generate_dataloader(xs, ys, args, labels)
        for i in range(args.runs):
            train_eeg_reg(args, datasets, i)
        return

    # Classification
    xs, ys = load_pain_classification_data(args)
    normalize_features(xs)
    datasets = generate_dataloader(xs, ys, args)
    for i in range(args.runs):
        train_eeg(args, datasets, i)

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2004)
parser.add_argument('--server_tag', type=str, default='pain_exp')
parser.add_argument('--task', type=str, default='regression', help='classification/regression/simulation')
parser.add_argument('--dataset', type=str, default='TUH')
parser.add_argument('--load_dataset', action='store_true', default=True)
parser.add_argument('--data', type=str, default='4')
parser.add_argument('--data_path', type=str, default='./data/pain')
parser.add_argument('--adj_file', type=str, default='adjs/raw_adj.npy')
parser.add_argument('--adj_type', type=str, default='er')
parser.add_argument('--source_data_path', type=str, default='./data/pain')
parser.add_argument('--est_data_path', type=str, default='./data/est')
parser.add_argument('--cross_subject', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--lr_decay_rate', type=float, default=0.92)
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--gnn_hid_dim', type=int, default=32)
parser.add_argument('--gnn_out_dim', type=int, default=16)
parser.add_argument('--gnn_layer_num', type=int, default=2)
parser.add_argument('--gnn_pooling', type=str, default='gate')
parser.add_argument('--agg_type', type=str, default='gate')
parser.add_argument('--gnn_adj_type', type=str, default='None')
parser.add_argument('--gnn_downsample_dim', type=int, default=0)
parser.add_argument('--gnn_res', action='store_true')
parser.add_argument('--encoder', type=str, default='rnn')
parser.add_argument('--feature_len', type=int, default=75)
parser.add_argument('--time_len', type=int, default=38)
parser.add_argument('--encoder_hid_dim', type=int, default=256)
parser.add_argument('--bidirect', action='store_true', default=True)
parser.add_argument('--decoder', type=str, default='lgg_cnn')
parser.add_argument('--decoder_hid_dim', type=int, default=512)
parser.add_argument('--decoder_out_dim', type=int, default=32)
parser.add_argument('--cut_encoder_dim', type=int, default=0)
parser.add_argument('--lgg', type=bool, default=True)
parser.add_argument('--lgg_mode', type=str, default='gumble')
parser.add_argument('--lgg_time', action='store_true')
parser.add_argument('--lgg_warmup', type=int, default=5)
parser.add_argument('--lgg_tau', type=float, default=0.1)
parser.add_argument('--lgg_hid_dim', type=int, default=64)
parser.add_argument('--lgg_k', type=int, default=5)
parser.add_argument('--kld_weight', type=float, default=0.1)
parser.add_argument('--vae_weight', type=float, default=0.1)
parser.add_argument('--N', type=int, default=127)
parser.add_argument('--score', type=str, default='NRS')
parser.add_argument('--reg_loss', type=str, default='weighted_mse')
parser.add_argument('--reg_normalized', type=bool, default=True)
parser.add_argument('--predict_class_num', type=int, default=1)
parser.add_argument('--predictor_hid_dim', type=int, default=256)
parser.add_argument('--predictor_num', type=int, default=3)
parser.add_argument('--pos', type=str, default=None)
parser.add_argument('--pe_type', type=str, default='learnable')
parser.add_argument('--best_model_save_path', type=str, default='best_models/pain_model.pth')
parser.add_argument('--log_file_path', type=str, default='logs/pain_log.txt')
parser.add_argument('--testing', action='store_true')
parser.add_argument('--clip', type=int, default=3)

args = parser.parse_known_args()[0]

if __name__ == "__main__":
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    eeg_util.DLog.init(args, args.log_file_path)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()

    if args.testing:
        print('Testing Mode')
        # Add testing logic here if needed, utilizing data loaders and loading the best model
        pass
    else:
        dt = time.strftime('%Y%m%d', time.localtime(time.time()))
        print(f"Starting training for task: {args.task}")
        multi_train(args)
        
    print('Execution finished.')