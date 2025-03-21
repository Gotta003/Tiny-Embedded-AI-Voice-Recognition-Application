import numpy as np
import matplotlib.pyplot as plt

# Load the spectrogram data from the file
spectrogram = np.loadtxt("spectrogram.txt")

# Plot the spectrogram
plt.figure(figsize=(10, 6))
plt.imshow(spectrogram.T, aspect="auto", origin="lower", cmap="viridis")
plt.colorbar(label="Log Magnitude")
plt.xlabel("Time (frames)")
plt.ylabel("Frequency (Mel bins)")
plt.title("Log Mel Spectrogram")
plt.show()
