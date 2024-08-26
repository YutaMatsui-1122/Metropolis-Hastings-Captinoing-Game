import numpy as np
import matplotlib.pyplot as plt

# plot test loss

test_file_pathes = [
    "models/adapter_tuning/fine_tune_prefix_test_loss.npy",
    "models/decoder_tuning/fine_tune_prefix_test_loss.npy",
    "cc3m2coco-official/2/clipcap_test_loss.npy",
]

for test_file_path in test_file_pathes:
    test_loss = np.load(test_file_path)
    plt.plot(test_loss)
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss")
plt.legend(["Adapter Tuning", "Decoder Tuning", "LoRA"])
plt.savefig("test_loss.png")    

# plot train loss

train_file_pathes = [
    "models/adapter_tuning/fine_tune_prefix_loss.npy",
    "models/decoder_tuning/fine_tune_prefix_loss.npy",
    "cc3m2coco-official/2/clipcap_loss.npy",
]

plt.clf()

for train_file_path in train_file_pathes:
    train_loss = np.load(train_file_path)
    plt.plot(train_loss, alpha=0.5)
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Train Loss")
plt.legend(["Adapter Tuning", "Decoder Tuning", "LoRA"])
plt.savefig("train_loss.png")