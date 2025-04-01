import numpy as np
import os
import matplotlib.pyplot as plt

def verify_npz_files(directory="/content/dataset/user_0_organized/npz_features"):
    """Verify contents of .npz files in a directory."""
    npz_files = [f for f in os.listdir(directory) if f.endswith('.npz')]
    
    if not npz_files:
        print(f"No .npz files found in {directory}!")
        return

    print(f"Found {len(npz_files)} .npz files in {directory}:\n")
    
    for file in sorted(npz_files):
        filepath = os.path.join(directory, file)
        data = np.load(filepath)
        
        print(f"File: {file}")
        print(f"Number of arrays stored: {len(data.files)}\n")
        
        for array_name in data.files:
            array_data = data[array_name]
            
            print(f"  Array: '{array_name}'")
            if(array_name=='features'):
              print(f"    - Shape: {array_data.shape}")
              print(f"    - Dtype: {array_data.dtype}")
              print(f"    - Min: {np.min(array_data):.4f}, Max: {np.max(array_data):.4f}, Mean: {np.mean(array_data):.4f}")
              print(f"    - Size: {array_data.size} elements\n")

        print("-" * 50 + "\n")

if __name__ == "__main__":
    verify_npz_files()
