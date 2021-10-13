import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("whitegrid")


network = 'resnet'
depth = 20
if network=='resnet':
    network = 'resnet{}'.format(depth)

dataset = 'cifar10'
# methods = ['FenBP_Prox', 'PQ', 'ST']
methods = ['FenBP', 'PQ', 'ST']
#methods = ['FenBP', 'PQ']
data_frames = []
plot_size=300

for method in methods:
    # result_path = 'results/{}_{}/results.csv'.format(network, method)
    result_path = 'results/{}_{}_{}/results.csv'.format(network, dataset, method)
    df = pd.read_csv(result_path)
    data_frames.append(df)
    best_train_acc = max(100 - df.train_error1[0:plot_size])
    best_val_acc = max(100 - df.val_error1_bin[0:plot_size])
    print('{} - Best train acc: {}'.format(method,best_train_acc))
    print('{} - Best train acc: {}'.format(method, best_val_acc))

# Plot training loss

plt.figure(figsize=(8,5))
for i, method in enumerate(methods):
    df =  data_frames[i]
    plt.plot(df.epoch[0:plot_size], df.train_loss[0:plot_size], label="{} - Training loss".format(method), lw=3)

plt.ylabel("Training Loss"); plt.xlabel("Epoch"); plt.legend(); plt.tight_layout()
plt.savefig('figs/train_loss.png')


# Plot validation loss
plt.figure(figsize=(8,5))
for i, method in enumerate(methods):
    df =  data_frames[i]
    plt.plot(df.epoch[0:plot_size], df.val_loss[0:plot_size], label="{} - Validation loss".format(method), lw=3)

plt.ylabel("Validation Loss"); plt.xlabel("Epoch"); plt.legend(); plt.tight_layout()
plt.savefig('figs/val_loss.png')

# Plot accuracy 
plt.figure(figsize=(8,5))
for i, method in enumerate(methods):
    df =  data_frames[i]
    plt.plot(df.epoch[0:plot_size], 100-df.val_error1_bin[0:plot_size], label="{} - Top-1 Acc".format(method), lw=3)

plt.ylabel("Validation Loss"); plt.xlabel("Epoch"); plt.legend(); plt.tight_layout()
plt.savefig('figs/acc.png')




