import json
import pdb
import matplotlib.pyplot as plt


def get_train_loss(data_file):
    f = open(data_file, 'r')
    data = json.load(f)
    return data['train_loss']

def get_test_acc(data_file):
    f = open(data_file, 'r')
    data = json.load(f)
    return data['test_acc']


bayes_file='./cifar10_vgg_bayes/dicts/train_hist_200.json'
fenbp_file='./cifar10_vgg_fenbp/dicts/train_hist_200.json'

bayes_train_loss = get_train_loss(bayes_file)
bayes_test_acc = get_test_acc(bayes_file)

fenbp_train_loss = get_train_loss(fenbp_file)
fenbp_test_acc = get_test_acc(fenbp_file)
plt.figure(figsize=(8,5))
plt.plot(list(range(len(bayes_train_loss))), bayes_train_loss, label='BayesBiNN - Loss', lw = 3 )
plt.plot(list(range(len(fenbp_train_loss))), fenbp_train_loss, label='FenBP - Loss', lw=3 )
plt.ylabel("Training Loss"); plt.xlabel("Epoch"); plt.legend(); plt.tight_layout()
plt.savefig('cifar10_vgg_train_loss.png')

plt.figure(figsize=(8,5))
plt.plot(list(range(len(bayes_train_loss))), bayes_test_acc, label='BayesBiNN - Acc', lw = 3 )
plt.plot(list(range(len(fenbp_train_loss))), fenbp_test_acc, label='FenBP - Acc', lw=3 )
plt.ylabel("Accuracy "); plt.xlabel("Epoch"); plt.legend(); plt.tight_layout()
plt.savefig('cifar10_vgg_acc.png')






# data_file = './train_hist_100.json'
# f = open(data_file, 'r')
# data = json.load(f)
# pdb.set_trace()

# print(data)


