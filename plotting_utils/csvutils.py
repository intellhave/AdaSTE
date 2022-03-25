import pandas as pd

def read_csv_loss_acc(file_path):
    df = pd.read_csv(file_path)
    return df.train_loss, df.best_val_acc1

