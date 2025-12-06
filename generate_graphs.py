import matplotlib.pyplot as plt
import numpy as np

# Data extracted from training logs
# Epoch 5 Step-wise Loss (Sampled every 100 steps)
steps = [600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400]
losses = [2.78, 6.94, 11.44, 7.43, 9.74, 4.44, 3.87, 5.38, 4.82, 3.47, 1.11, 11.91, 10.05, 1.36, 6.50, 2.95, 15.65, 5.45, 7.34]

# Epoch Averages (Reconstructed)
epochs = [1, 5]
train_avg_losses = [59.88, 6.63] # Epoch 1 from earlier log, Epoch 5 from final log
val_losses = [12.71, 8.24] # Epoch 1 from earlier log, Epoch 5 from final log

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Training Stability (Epoch 5)
ax1.plot(steps, losses, marker='o', linestyle='-', color='#2563eb', label='Step Loss')
ax1.axhline(y=6.63, color='#16a34a', linestyle='--', label='Epoch Average (6.63)')
ax1.set_title('Training Stability (Epoch 5)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Step', fontsize=12)
ax1.set_ylabel('MSE Loss', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# Plot 2: Convergence (Start vs End)
bar_width = 0.35
index = np.arange(len(epochs))

bar1 = ax2.bar(index, train_avg_losses, bar_width, label='Training Loss', color='#2563eb', alpha=0.8)
bar2 = ax2.bar(index + bar_width, val_losses, bar_width, label='Validation Loss', color='#dc2626', alpha=0.8)

ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('MSE Loss', fontsize=12)
ax2.set_title('Model Convergence (Epoch 1 vs 5)', fontsize=14, fontweight='bold')
ax2.set_xticks(index + bar_width / 2)
ax2.set_xticklabels(['Epoch 1', 'Epoch 5'])
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for i, v in enumerate(train_avg_losses):
    ax2.text(i, v + 1, str(v), ha='center', fontweight='bold')
for i, v in enumerate(val_losses):
    ax2.text(i + bar_width, v + 1, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('performance_analysis.png', dpi=300)
print("Graph generated: performance_analysis.png")
