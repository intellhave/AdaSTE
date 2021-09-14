import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("whitegrid")

csvpath = "results/resnet20_FenBP/results.csv"
csvpath = "results/resnet20_PQ/results.csv"

df = pd.read_csv(csvpath)

plt.figure(figsize=(8,5))
plt.plot(df.epoch, df.train_loss, label="train", lw=3)
plt.plot(df.epoch, df.val_loss, label="val", lw=3)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(df.epoch, 100-df.train_error1, label="train", lw=3)
plt.plot(df.epoch, 100-df.val_error1, label="val", lw=3)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.show()