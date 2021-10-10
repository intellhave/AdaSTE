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
csvpath_fenbp = "results/resnet20_FenBP/results.csv"
# csvpath = "results/resnet18_prox_Adam_run_0/results.csv"
# csvpath_st = "results/resnet20_bc_Adam_run_0/results.csv"
csvpath_st = "results/resnet20_PQ/results.csv"
# csvpath = "results/simple_mlp_MNIST/results.csv"
# csvpath = "results/simple_mlp_MNIST/results.csv"

df_fenbp = pd.read_csv(csvpath_fenbp)
df_st = pd.read_csv(csvpath_st)

best_train_acc_fenbp = max(100-df_fenbp.train_error1)
best_val_acc_fenbp = max(100-df_fenbp.val_error1)
print('FENBP Best train accuracy: ', best_train_acc_fenbp)
print('FENBP Best val accuracy: ', best_val_acc_fenbp)

best_train_acc_st = max(100-df_st.train_error1)
best_val_acc_st = max(100-df_st.val_error1)
print('ST Best train accuracy: ', best_train_acc_st)
print('ST Best val accuracy: ', best_val_acc_st)

plot_size = 300

plt.figure(figsize=(8,5))
plt.plot(df_fenbp.epoch[0:plot_size], df_fenbp.train_loss[0:plot_size], label="FENBP_train", lw=3)
plt.plot(df_st.epoch[0:plot_size], df_st.train_loss[0:plot_size], label="ST_train", lw=3)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig('figs/train_loss.png')


plt.figure(figsize=(8,5))
plt.plot(df_fenbp.epoch[0:plot_size], df_fenbp.val_loss[0:plot_size], label="FENBP_val", lw=3)
plt.plot(df_st.epoch[0:plot_size], df_st.val_loss[0:plot_size], label="ST_val", lw=3)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig('figs/val_loss.png')


plt.figure(figsize=(8,5))
plt.plot(df_fenbp.epoch[0:plot_size], 100-df_fenbp.val_error1[0:plot_size], label="FENBP", lw=3)
plt.plot(df_st.epoch[0:plot_size], 100-df_st.val_error1[0:plot_size], label="ST", lw=3)
# plt.plot(df.epoch, 100-df.train_error5, label="train_5", lw=3)
# plt.plot(df.epoch, 100-df.val_error5, label="val_5", lw=3)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig('figs/acc.png')
