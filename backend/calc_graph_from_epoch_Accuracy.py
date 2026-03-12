import matplotlib.pyplot as plt
import numpy as np

accuracy_values = [0.8704, 0.8811, 0.8880, 0.8911, 0.8896, 0.8942, 0.8949, 0.8965, 
                   0.8965, 0.8949, 0.8972, 0.8972, 0.8965, 0.8949, 0.8926, 0.8988, 
                   0.9049, 0.9049, 0.9034, 0.9026, 0.9018, 0.9011, 0.9018, 0.8995, 
                   0.9041, 0.8995, 0.9018, 0.9057, 0.9034, 0.9049, 0.9026, 0.9018, 
                   0.9018, 0.9026, 0.9041, 0.9026, 0.9018, 0.9018, 0.9011, 0.9026]

epochs = range(1, len(accuracy_values) + 1)

# Set dynamic limits to zoom in on the relevant range
y_min = min(accuracy_values) - 0.005
y_max = max(accuracy_values) + 0.005

plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracy_values, marker='o', linestyle='-', color='#1f77b4', markersize=4, label='Accuracy')

# Optional: Add a trendline to see the general direction
z = np.polyfit(epochs, accuracy_values, 3)
p = np.poly1d(z)
plt.plot(epochs, p(epochs), "r--", label='Trendline', alpha=0.8)

plt.title('Model Accuracy per Epoch', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(y_min, y_max)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()
plt.savefig('/home/abk/abk/projects/Major-project-basic-ui/backend/Results/accuracy_plot.png')