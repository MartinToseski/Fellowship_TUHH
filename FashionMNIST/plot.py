import pandas as pd
import matplotlib.pyplot as plt

log_path = "logs/LeNet5_v1/version_39/metrics.csv"
df = pd.read_csv(log_path)

train_df = df.dropna(subset=["train_loss"])

plt.figure()
plt.plot(train_df["step"], train_df["train_loss"])
plt.title("Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.savefig("train_loss.png")
plt.close()


val_df = df.dropna(subset=["val_loss"])

plt.figure()
plt.plot(val_df["epoch"], val_df["val_loss"], marker="o")
plt.title("Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("val_loss.png")
plt.close()


val_df = df.dropna(subset=["val_acc"])

plt.figure()
plt.plot(val_df["epoch"], val_df["val_acc"], marker="o")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("val_acc.png")
plt.close()

test_df = df.dropna(subset=["test_acc"])

print(test_df[["test_acc", "test_loss"]])