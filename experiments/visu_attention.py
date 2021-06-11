# -*- coding: utf-8 -*-
import argparse
import numpy as np
import networkx as nx
import os
import torch

from matplotlib.gridspec import GridSpec
from torch_geometric import datasets
# from torch_geometric.data import DataLoader
from torch_geometric.utils.convert import to_networkx
from transformer.models import DiffGraphTransformer, GraphTransformer
from transformer.data import GraphDataset
from transformer.data_io import display_multilayer_attns_and_graph, save_multilayer_attns, make_attns_gif
from transformer.position_encoding import LapEncoding, POSENCODINGS
from transformer.utils import count_parameters

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

'''
Script for computing and visualizing attention scores.
'''


def load_args():
    parser = argparse.ArgumentParser(
        description='Transformer visualization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default="Mutagenicity",
                        help='name of dataset')
    parser.add_argument('--nb-heads', type=int, default=4)
    parser.add_argument('--nb-layers', type=int, default=3)
    parser.add_argument('--dim-hidden', type=int, default=64)
    parser.add_argument('--pos-enc', choices=[None,
                        'diffusion', 'pstep', 'adj'],
                        default='diffusion')
    parser.add_argument('--lappe', action='store_true',
                        help='use laplacian PE')
    parser.add_argument('--lap-dim', type=int, default=8,
                        help='dimension for laplacian PE')
    parser.add_argument('--pe-size', type=int, default=10,
                        help='dimension of the pe for lappe and graphwave')
    parser.add_argument('--p', type=int, default=1,
                        help='p step random walk kernel')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='bandwidth for the diffusion kernel')
    parser.add_argument('--normalization', choices=[None, 'sym', 'rw'],
                        default=None, help='normalization for Laplacian')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--outdir', type=str, default='../pretrained_models',
                        help='output path')
    parser.add_argument('--gcn', action='store_true',
                        help='use matrix multiplication')
    parser.add_argument('--batch-norm', action='store_true',
                        help='use batch norm instead of layer norm')
    parser.add_argument('--degree', action='store_true',
                        help='use inverse degree as factor in self-attn')
    parser.add_argument('--zero-diag', action='store_true',
                        help='zero diagonal for PE matrix')
    parser.add_argument('--fold-idx', type=int, default=1,
                        help='indices for the train/test datasets')
    parser.add_argument('--idx-sample', type=int, default=0,
                        help='idx of sample to visualize')
    parser.add_argument('--idx-layer', type=int, default=0,
                        help='idx of layer to visualize')
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()

    return args


def main():
    global args
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    data_path = '../dataset/TUDataset'

    dset_name = args.dataset
    if dset_name == 'PTC-MR':
        dset_name = 'PTC'

    dset = datasets.TUDataset(data_path, args.dataset)
    n_tags = None  # dset[0].num_node_features
    nb_class = dset.num_classes

    if args.dataset == 'Mutagenicity':
        nb_samples = 4337
        end_train = int(0.8 * 4337)
        end_val = int(0.9 * 4337)
        idxs = np.arange(nb_samples)
        train_fold_idx = torch.from_numpy(idxs[:end_train].astype(int))
        val_fold_idx = torch.from_numpy(idxs[end_train:end_val].astype(int))
        test_fold_idx = torch.from_numpy(idxs[end_val:].astype(int))
    else:
        idx_path = '../dataset/{}/10fold_idx/inner_folds/{}-{}-{}.txt'
        test_idx_path = '../dataset/{}/10fold_idx/test_idx-{}.txt'

        inner_idx = 1
        train_fold_idx = torch.from_numpy(np.loadtxt(
            idx_path.format(
                    args.dataset, 'train_idx', args.fold_idx, inner_idx
                    )).astype(int))
        val_fold_idx = torch.from_numpy(np.loadtxt(
            idx_path.format(
                    args.dataset, 'val_idx', args.fold_idx, inner_idx
                    )).astype(int))
        test_fold_idx = torch.from_numpy(np.loadtxt(
            test_idx_path.format(args.dataset, args.fold_idx)).astype(int))

    train_dset = GraphDataset(dset[train_fold_idx], n_tags, degree=args.degree)
    input_size = train_dset.input_size()
    print(len(train_dset))
    print(train_dset[0])
    val_dset = GraphDataset(dset[val_fold_idx], n_tags, degree=args.degree)

    if not os.path.exists("../cache/pe/{}".format(args.dataset)):
        try:
            os.makedirs("../cache/pe/{}".format(args.dataset))
        except Exception:
            pass

    pos_encoder = None
    if args.pos_enc is not None:
        pos_encoding_method = POSENCODINGS.get(args.pos_enc, None)
        pos_encoding_params_str = ""
        if args.pos_enc == 'diffusion':
            pos_encoding_params = {
                'beta': args.beta
            }
            pos_encoding_params_str = args.beta
        elif args.pos_enc == 'pstep':
            pos_encoding_params = {
                'beta': args.beta,
                'p': args.p
            }
            pos_encoding_params_str = "{}_{}".format(args.p, args.beta)
        else:
            pos_encoding_params = {}

        if pos_encoding_method is not None:
            pos_cache_path = '../cache/pe/{}/{}_{}_{}.pkl'.format(
                args.dataset, args.pos_enc, args.normalization,
                pos_encoding_params_str)
            pos_encoder = pos_encoding_method(
                pos_cache_path, normalization=args.normalization,
                zero_diag=args.zero_diag, **pos_encoding_params)

        print("Position encoding...")
        pos_encoder.apply_to(dset, split='all')
        # pos_encoder.apply_to(val_dset, split='val')
        train_dset.pe_list = [dset.pe_list[i] for i in train_fold_idx]
        val_dset.pe_list = [dset.pe_list[i] for i in val_fold_idx]

    if args.lappe and args.lap_dim > 0:
        lap_pos_encoder = LapEncoding(args.lap_dim, normalization='sym')
        lap_pos_encoder.apply_to(train_dset)
        lap_pos_encoder.apply_to(val_dset)

    if args.pos_enc is not None:
        model = DiffGraphTransformer(in_size=input_size,
                                     nb_class=nb_class,
                                     d_model=args.dim_hidden,
                                     dim_feedforward=2*args.dim_hidden,
                                     dropout=args.dropout,
                                     nb_heads=args.nb_heads,
                                     nb_layers=args.nb_layers,
                                     # gcn=args.gcn,
                                     batch_norm=args.batch_norm,
                                     lap_pos_enc=args.lappe,
                                     lap_pos_enc_dim=args.lap_dim)
    else:
        model = GraphTransformer(in_size=input_size,
                                 nb_class=nb_class,
                                 d_model=args.dim_hidden,
                                 dim_feedforward=2*args.dim_hidden,
                                 dropout=args.dropout,
                                 nb_heads=args.nb_heads,
                                 nb_layers=args.nb_layers)
    # load model and put it in eval mode
    if args.use_cuda:
        to_load = torch.load(os.path.join(args.outdir, 'model.pkl'))
    else:
        to_load = torch.load(os.path.join(args.outdir, 'model.pkl'),
                             map_location=torch.device('cpu'))
    print('Incoming model: ', to_load['args'])
    model.load_state_dict(to_load['state_dict'])
    if args.use_cuda:
        model.cuda()
    model.eval()
    print("Total number of parameters: {}".format(count_parameters(model)))

    test_dset = GraphDataset(dset[test_fold_idx], n_tags, degree=args.degree)
    if pos_encoder is not None:
        test_dset.pe_list = [dset.pe_list[i] for i in test_fold_idx]

    if args.lappe and args.lap_dim > 0:
        lap_pos_encoder.apply_to(test_dset)

    print("Visualizing...")

    # get the desired sample.
    sample = train_dset[args.idx_sample]

    attns = []

    # add hook for attention.
    def get_attns(module, input, output):
        attns.append(output[1])
    for i in range(args.nb_layers):
        model.encoder.layers[i].self_attn.register_forward_hook(get_attns)

    # inference, with correct prediction.
    collate_fn = train_dset.collate_fn()
    data, mask, pe, lap_pe, degree, label = collate_fn([sample])
    with torch.no_grad():
        output = model(data, mask, pe, lap_pe, degree)
    pred = output.data.argmax(dim=-1)
    print('Ground truth: ', label, 'Prediction: ', pred)

    # plot attentions and graph.
    # display_multilayer_attns_and_graph(args, sample, attns, pe=pe)
    make_attns_gif(args, sample, attns, pe=pe)


if __name__ == "__main__":
    main()
