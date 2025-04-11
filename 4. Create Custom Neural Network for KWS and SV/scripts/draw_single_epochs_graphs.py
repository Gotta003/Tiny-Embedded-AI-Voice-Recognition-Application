import re
import matplotlib.pyplot as plt

# Ask user for epoch range
min_epoch = int(input("Enter minimum epoch: "))
max_epoch = int(input("Enter maximum epoch: "))

# Path to your log file
log_file='epochs.txt'
#log_file = 'epochs_dnn.txt'

# Data containers
epochs = []
accuracy = []
val_accuracy = []
loss = []
val_loss = []

# Read and parse the log
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

# Create 2x2 subplot layout
plt.figure(figsize=(14, 10))

# 1. Training Accuracy
plt.subplot(2, 2, 1)
plt.plot(epochs, accuracy, color='blue', marker='o', linewidth=1)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

# 2. Validation Accuracy
plt.subplot(2, 2, 2)
plt.plot(epochs, val_accuracy, color='green', marker='o', linewidth=1)
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

# 3. Training Loss
plt.subplot(2, 2, 3)
plt.plot(epochs, loss, color='red', marker='o', linewidth=1)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# 4. Validation Loss
plt.subplot(2, 2, 4)
plt.plot(epochs, val_loss, color='orange', marker='o', linewidth=1)
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.suptitle(f'Training Metrics from Epoch {min_epoch} to {max_epoch}', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
