import re
import matplotlib.pyplot as plt

# Path to your log file
log_file = 'epochs.txt'

# Data containers
epochs = []
accuracy = []
val_accuracy = []
loss = []
val_loss = []
min_epoch=100
max_epoch=300

with open(log_file, 'r') as file:
    lines = file.readlines()

    for i in range(len(lines)):
        if lines[i].startswith("Epoch"):
            epoch_num = int(lines[i].split()[1].split('/')[0])
            if min_epoch <= epoch_num <= max_epoch:
                match = re.search(r'accuracy: ([\d.]+) - loss: ([\d.]+) - val_accuracy: ([\d.]+) - val_loss: ([\d.]+)', lines[i+1])
                if match:
                    epochs.append(epoch_num)
                    accuracy.append(float(match.group(1)))
                    loss.append(float(match.group(2)))
                    val_accuracy.append(float(match.group(3)))
                    val_loss.append(float(match.group(4)))

# Plot Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, label='Train Accuracy', marker='o', linewidth=1)
plt.plot(epochs, val_accuracy, label='Validation Accuracy', marker='o', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy (Epochs 100–700)')
plt.legend()
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='Train Loss', marker='o', linewidth=1)
plt.plot(epochs, val_loss, label='Validation Loss', marker='o', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss (Epochs 100–700)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
