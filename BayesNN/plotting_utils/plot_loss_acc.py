import os
import pandas as pd
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("whitegrid")

from json_utils import get_train_loss, get_test_acc, get_loss_acc_json
from csvutils import read_csv_loss_acc
from txt_utils import parse_bayesbinn_log_file, parse_md_tanh_log_file

plt.rc('font', family='serif')

dataset='CIFAR10'
network = 'VGG16'

method_lists = {}
#method_lists ['acc'] = ['FENBPNA', 'FENBP', 'BAYESBINN', 'BC', 'MDTANH']
method_lists ['acc'] = ['FENBPNA', 'FENBP', 'BAYESBINN', 'MDTANH', 'BC']
method_lists ['loss'] =  ['FENBPNA', 'FENBP', 'BAYESBINN']
#method_lists ['loss'] =  ['FENBPNA', 'FENBP', 'BAYESBINN']

method_legend = {'FENBPNA': 'Ours (w/o annealing)', 
        'FENBP': 'Ours (w/ annealing)',
        'BAYESBINN': 'BayesBiNN',
        'BC': 'BinaryConnect',
        'STE': 'STE',
        'MDTANH': 'MD-tanh'}
method_colors = {
        'FENBPNA': 'red',
        'FENBP': 'blue', 
        'BAYESBINN': 'green',
        'BC': 'black',
        'STE': 'black',
        'MDTANH': '#B22400'
        }

method_line_styles = {
        'FENBPNA': 'solid',
        'FENBP': 'solid', 
        'BAYESBINN': 'dashed',
        'BC': 'dashed',
        'STE': 'dashed',
        'MDTANH': 'dotted'
        }

json_methods=['FENBP', 'FENBPNA', 'BAYESBINN', 'STE']
plot_types=['acc', 'loss']
# plot_types=['loss']
y_labels = {'loss': 'Training Loss', 'acc': 'Testing Accuracy'}

for plot_type in plot_types:
    plt.figure(figsize=(6,4))
    max_epochs = 200

    method_list = method_lists[plot_type]

    for method in method_list:
        # data_path = '/home/intellhave/Work/RunResults/SelectedRuns/{}/{}/{}/'.format(dataset, network, method)
        data_path = '/media/intellhave/Data/BINN_Results/SelectedRuns/{}/{}/{}/'.format(dataset, network, method)

        train_loss, train_acc = [], []
        if method in json_methods: 
            file_name = 'train_hist_100.json'
            txt_file_name = 'log.txt'
            file_path = os.path.join(data_path, file_name)
            txt_file_path = os.path.join(data_path, txt_file_name)
            #train_loss, test_acc = get_loss_acc_json(file_path)
            if os.path.exists(txt_file_path):
                train_loss, test_acc = parse_bayesbinn_log_file(txt_file_path)
            else:
                train_loss, test_acc = get_loss_acc_json(file_path)
        else:
            txt_file_name = 'log.txt'
            csv_file_name = 'results.csv'
            txt_file_path = os.path.join(data_path, txt_file_name)
            csv_file_path = os.path.join(data_path, csv_file_name)
            if plot_type=='loss':
                train_loss, test_acc = parse_md_tanh_log_file(txt_file_path)
            else:
                train_loss, test_acc = read_csv_loss_acc(csv_file_path)

        train_loss, test_acc = list(train_loss), list(test_acc)
        xdata = list(range(min(max_epochs, len(train_loss)))) 

        plot_data = train_loss if plot_type=='loss' else test_acc

        plt.plot(xdata, plot_data[0:len(xdata)], 
                label='{}'.format(method_legend[method]), 
                linestyle=method_line_styles[method],
                color = method_colors[method],
                lw = 3 )

        plt.ylabel(y_labels[plot_type], fontsize=14); 
        plt.xlabel('{Epoch}', fontsize=14); 
        plt.legend(fontsize=12); plt.tight_layout()
        plt.savefig('./figs/{}/{}_{}_{}_e{}.pdf'.format(dataset,dataset, network, plot_type,max_epochs))

# bayes_file='./data/cifar10_vgg_bayes/dicts/train_hist_200.json'
# fenbp_file='./data/cifar10_vgg_fenbp/dicts/train_hist_200.json'
# md_file='/home/intellhave/Work/RunResults/Alvis/out/CIFAR10/VGG16/TANH_PROJECTION_STE/ADAM_lr0.001_bts1.05_lrs0.5_bti500_seed_100/2021-10-28_00-59-05/results.csv'
# # bc_file = 

# bayes_train_loss = get_train_loss(bayes_file)
# bayes_test_acc = get_test_acc(bayes_file)

# fenbp_train_loss = get_train_loss(fenbp_file)
# fenbp_test_acc = get_test_acc(fenbp_file)

# md_tanh_loss, md_tanh_acc = read_csv_loss_acc(md_file)

# import pdb; pdb.set_trace()

# # plt.figure(figsize=(8,5))
# # plt.plot(list(range(len(bayes_train_loss))), bayes_train_loss, label='BayesBiNN - Loss', lw = 3 )
# # plt.plot(list(range(len(fenbp_train_loss))), fenbp_train_loss, label='FenBP - Loss', lw=3 )
# # plt.ylabel("Training Loss"); plt.xlabel("Epoch"); plt.legend(); plt.tight_layout()
# # plt.savefig('cifar10_vgg_train_loss.png')

# plt.figure(figsize=(8,5))
# plt.plot(list(range(len(bayes_train_loss))), bayes_test_acc, label='BayesBiNN - Acc', lw = 3 )
# plt.plot(list(range(len(fenbp_train_loss))), fenbp_test_acc, label='FenBP - Acc', lw=3 )
# plt.plot(list(range(len(md_tanh_acc))), md_tanh_acc, label='MD - Acc', lw=3 )
# plt.ylabel("Accuracy "); plt.xlabel("Epoch"); plt.legend(); plt.tight_layout()
# plt.savefig('cifar10_vgg_acc.png')



