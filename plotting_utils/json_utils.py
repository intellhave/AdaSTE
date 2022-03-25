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

def get_loss_acc_json(data_file):
    f = open(data_file, 'r')
    data = json.load(f)
    return data['train_loss'], data['test_acc']








