import numpy as np
import os
import matplotlib.pyplot as plt

def verify_npy_files(directory="npy"):
    """Verify contents of .npy files in a directory."""
    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    
    if not npy_files:
        print(f"No .npy files found in {directory}!")
        return

    print(f"Found {len(npy_files)} .npy files in {directory}:\n")
    
    for file in sorted(npy_files):
        filepath = os.path.join(directory, file)
        data = np.load(filepath)
        
        print(f"{file}:")
        print(f"  - Shape: {data.shape}")
        print(f"  - Dtype: {data.dtype}")
        print(f"  - Min: {np.min(data):.4f}, Max: {np.max(data):.4f}, Mean: {np.mean(data):.4f}")
        print(f"  - Size: {data.size} elements\n")

        if len(data.shape) == 4 and 'conv' in file.lower():
            print("  Visualizing first filter of 4D weights...")
            plt.figure(figsize=(3, 3))
            plt.imshow(data[0, :, :, 0], cmap='viridis') 
            plt.colorbar()
            plt.title(f"{file} (first filter)")
            plt.show()

if __name__ == "__main__":
    verify_npy_files()
