import numpy as np

weights = np.load("weights.npy", allow_pickle=True).item()
print("Weights:")
for key, value in weights.items():
    print(f"{key}: Shape={value['dequantized'].shape}, Data={value['dequantized']}")

biases = np.load("biases.npy", allow_pickle=True).item()
print("\nBiases:")
for key, value in biases.items():
    print(f"{key}: Shape={value['dequantized'].shape}, Data={value['dequantized']}")
