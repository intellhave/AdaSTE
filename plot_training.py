import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("whitegrid")

# csvpath = "results/resnet20_FenBP_e600/results.csv"
# csvpath = "results/resnet20_FenBP/results.csv"
# csvpath = "results/resnet20_FenBP_1em7/results.csv"
# csvpath = "results/resnet20_FenBP_1em9/results.csv"
# csvpath = "results/resnet20_FenBP_1em6_normeta/results.csv"
# csvpath = "results/resnet20_FenBP_1em6_f4/results.csv"
# csvpath = "results/resnet/results.csv"
# csvpath = "results/simple_mlp_MNIST/results.csv"
# csvpath = "results/resnet20_FenBP/results.csv"
# csvpath = "results/resnet18_prox_Adam_run_0/results.csv"
csvpath = "results/resnet20_bc_Adam_run_0/results.csv"
# csvpath = "results/simple_mlp_MNIST/results.csv"
# csvpath = "results/simple_mlp_MNIST/results.csv"

df = pd.read_csv(csvpath)

plt.figure(figsize=(8,5))
plt.plot(df.epoch, df.train_loss, label="train", lw=3)
plt.plot(df.epoch, df.val_loss, label="val", lw=3)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig('figs/loss.png')


best_train_acc = max(100-df.train_error1)
best_val_acc = max(100-df.val_error1)
print('Best train accuracy: ', best_train_acc)
print('Best val accuracy: ', best_val_acc)

plt.figure(figsize=(8,5))
plt.plot(df.epoch, 100-df.train_error1, label="train", lw=3)
plt.plot(df.epoch, 100-df.val_error1, label="val", lw=3)
plt.plot(df.epoch, 100-df.train_error5, label="train_5", lw=3)
plt.plot(df.epoch, 100-df.val_error5, label="val_5", lw=3)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig('figs/acc.png')
