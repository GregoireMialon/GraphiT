import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from matplotlib.gridspec import GridSpec
from torch_geometric.utils.convert import to_networkx

ATOMS = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S',
         8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}

ATOM_COLORS = {'C': 'dodgerblue', 'O': 'firebrick', 'Cl': 'mediumseagreen',
               'H': 'dimgrey', 'N': 'mediumorchid', 'F': 'orange',
               'S': 'lightseagreen', 'P': 'olive', 'Br': 'chocolate',
               'Na': 'plum'}


def plot_attns(dataset, sample, attns, layer, pe):
    '''
    This function displays the content of the
    4 attention heads for a given layer.
    '''
    if dataset == 'Mutagenicity':
        labels = sample.x.argmax(dim=1).numpy()
    f = plt.figure()
    gs = GridSpec(2, 2, figure=f)
    sample_nx = to_networkx(sample)
    attns = attns[layer][0]
    pe = pe[0]
    for i in range(len(attns)):
        f.add_subplot(gs[i])
        plt.imshow(attns[i])
        plt.colorbar()
    plt.figure()
    plt.imshow(pe)
    plt.colorbar()
    plt.figure()
    labels_dict = {}
    color_map = []
    for node in sample_nx.nodes:
        labels_dict[node] = str(node) + ' (' + ATOMS[labels[node]] + ')'
        color_map.append(ATOM_COLORS[ATOMS[labels[node]]])
    # nx.draw(sample_nx, with_labels=True, arrows=False)
    nx.draw(sample_nx, labels=labels_dict, node_color=color_map, arrows=False)
    plt.show()
    return


def save_multilayer_attns(args, sample, attns, pe):
    '''
    This function saves a figure of the averaged
    content of attention scores (per head) for each
    layer.
    '''
    pe = pe[0]
    # attn_labels = {0: (None, None),
    #             1: ([1, 12], ['C', 'O']),
    #             2: ([3, 4, 5, 6, 7, 15], ['N', 'S', 'N', 'N', 'N', 'Cl'])}
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    for i in range(len(attns)):
        fig = plt.figure(figsize=(4, 4), dpi=300)
        ax = fig.add_subplot()
        ax.tick_params(direction='out')
        ax_r = ax.secondary_yaxis('right')
        ax_t = ax.secondary_xaxis('top')
        ax_r.tick_params(axis='y', direction='in')
        ax_t.tick_params(axis='x', direction='inout')
        # plt.xticks(attn_labels[i][0], attn_labels[i][1])
        # plt.yticks(attn_labels[i][0], attn_labels[i][1])
        avg_attns = attns[i][0].sum(dim=0) / len(attns[i][0])
        plt.imshow(avg_attns)
        plt.title(f"Layer {i + 1}", fontsize=20)
        # plt.show()
        plt.savefig(f"attns_{args.idx_sample}_{i + 1}_t.pdf",
                    format="pdf")
    return


def display_multilayer_attns_and_graph(args, sample, attns, pe):
    '''
    This function displays a figure of the averaged content
    of attention scores (per head) for each layer) as well
    as the corresponding molecule.
    '''
    if args.dataset == 'Mutagenicity':
        labels = sample.x.argmax(dim=1).numpy()
    f = plt.figure()
    gs = GridSpec(1, len(attns), figure=f)
    sample_nx = to_networkx(sample)
    sample_nx = to_networkx(sample, to_undirected=True)
    pe = pe[0]

    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    for i in range(len(attns)):
        ax = f.add_subplot(gs[i])
        ax.tick_params(direction='out')
        ax_r = ax.secondary_yaxis('right')
        ax_t = ax.secondary_xaxis('top')
        ax_r.tick_params(axis='y', direction='in')
        ax_t.tick_params(axis='x', direction='inout')
        avg_attns = attns[i][0].sum(dim=0) / len(attns[i][0])
        plt.imshow(avg_attns)
        plt.title(f"Layer {i + 1}", fontsize=25)
    plt.figure()
    labels_dict = {}
    color_map = []
    for node in sample_nx.nodes:
        labels_dict[node] = str(node) + ' (' + ATOMS[labels[node]] + ')'
        color_map.append(ATOM_COLORS[ATOMS[labels[node]]])
    nx.draw(sample_nx, labels=labels_dict, node_color=color_map,
            pos=nx.kamada_kawai_layout(sample_nx, weight=None),
            arrows=False)
    plt.show()
    return


def make_attns_gif(args, sample, attns, pe):
    '''
    '''
    if args.dataset == 'Mutagenicity':
        labels = sample.x.argmax(dim=1).numpy()
    sample_nx = to_networkx(sample)
    sample_nx = to_networkx(sample, to_undirected=True)
    pe = pe[0]
    
    avg_attns = attns[2][0].sum(dim=0) / len(attns[2][0])
    
    labels_dict = {}
    for node in sample_nx.nodes:
        labels_dict[node] = ATOMS[labels[node]]
    # color_map = []
    vmin = 0
    vmax = 0.1
    for node_idx in range(len(avg_attns)):
        plt.figure()
        color_map = torch.clone(avg_attns[node_idx]).tolist()
        color_map[node_idx] = -1 
        cmap = plt.get_cmap('viridis')
        cmap.set_under('blue')
        for node in sample_nx.nodes:
            labels_dict[node] = ATOMS[labels[node]] # + str(node)
            # color_map.append(ATOM_COLORS[ATOMS[labels[node]]])
        # labels_dict[node_idx] = ATOMS[labels[node]]
        # plt.title(f"Attention from {labels_dict[node_idx]} atom", fontdict = {'fontsize' : 20})
        plt.figtext(0.35, 1.12, f"Attention from", fontsize=20, color='k', ha ='center', va='top')
        plt.figtext(0.6, 1.12, f"{labels_dict[node_idx]} atom", fontsize=20, color='b', ha ='center', va='top')
        # plt.figtext(0.50, 0.96, ' vs ', fontsize=20, color='k', ha ='center')	
        nx.draw(sample_nx, labels=labels_dict, node_color=color_map,
                pos=nx.kamada_kawai_layout(sample_nx, weight=None),
                font_color='white',
                font_size=15,
                edge_color="grey",
                vmin=vmin,
                vmax=vmax,
                width=2,
                node_size=400,
                cmap=cmap,
                arrows=False)
        # plt.show()
        if node_idx < 10:
            node_idx = str(0) + str(node_idx)
        # plt.figure()
        # plt.imshow(avg_attns)
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        # sm._A = []
        # plt.colorbar(sm)
        plt.savefig(f"../figures/gif/attns_{node_idx}_{args.idx_sample}.jpg",
		    bbox_inches='tight',
                    format="jpg")
        # plt.show()
    return


