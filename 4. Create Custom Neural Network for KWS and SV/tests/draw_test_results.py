import pandas as pd
import matplotlib.pyplot as plt

# I tuoi dati
data = {
    "Neurons": [192, 192, 192, 192, 192, 192, 256, 256, 256, 256, 256, 256],
    "Threshold": [0.8, 0.85, 0.9, 0.8, 0.85, 0.9, 0.8, 0.85, 0.9, 0.8, 0.85, 0.9],
    "Label": ["Me", "Me", "Me", "Others", "Others", "Others", "Me", "Me", "Me", "Others", "Others", "Others"],
    "Sheila Found": [256, 256, 256, 1858, 1858, 1858, 256, 256, 256, 1858, 1858, 1858],
    "Matteo Recognized CONV": [190, 98, 34, 0, 0, 0, 190, 98, 34, 0, 0, 0],
    "Matteo Recognized DENSE": [242, 128, 15, 119, 10, 0, 243, 193, 69, 80, 8, 0]
}

df = pd.DataFrame(data)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # NO sharey
labels = ['Me', 'Others']
samples = {'Me': 256, 'Others': 1858}
colors = {192: 'skyblue', 256: 'orange'}

for ax, label in zip(axes, labels):
    sub = df[df['Label'] == label]
    for neurons in [192, 256]:
        part = sub[sub['Neurons'] == neurons]
        ax.plot(part['Threshold'], part['Matteo Recognized DENSE'], marker='o', label=f'DENSE - {neurons} neurons', color=colors[neurons])
    # CONV come riferimento
    ax.plot(part['Threshold'], part['Matteo Recognized CONV'], marker='x', linestyle='--', color='gray', label='CONV (reference)')
    ax.set_title(f'Label: {label} (Input={samples[label]})')
    ax.set_xlabel('Threshold')
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend()
    ax.set_ylabel('Matteo Recognized')

plt.suptitle('Performance SV - DENSE vs CONV', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
