import matplotlib.pyplot as plt
import numpy as np


points = [(0, 0), (1, 1), (2, 0)]
classes = [-1, -1, +1]

for point, cls in zip(points, classes):
    if cls == -1:
        plt.scatter(point[0], point[1], color='red', label='Class -1' if 'Class -1' not in plt.gca().get_legend_handles_labels()[1] else "")
    else:
        plt.scatter(point[0], point[1], color='blue', label='Class +1' if 'Class +1' not in plt.gca().get_legend_handles_labels()[1] else "")
plt.show()

for point, cls in zip(points, classes):
    if cls == -1:
        plt.scatter(point[0], point[1], color='red', label='Class -1' if 'Class -1' not in plt.gca().get_legend_handles_labels()[1] else "")
    else:
        plt.scatter(point[0], point[1], color='blue', label='Class +1' if 'Class +1' not in plt.gca().get_legend_handles_labels()[1] else "")

x_values = np.linspace(-1, 3, 100)
y_values = x_values-1 
plt.plot(x_values, y_values, color='green', label='Decision Boundary')


y_values_pos = x_values - 2 
plt.plot(x_values, y_values_pos, color='grey', linestyle='--')


y_values_neg = x_values  
plt.plot(x_values, y_values_neg, color='grey', linestyle='--')


plt.xlabel('X1')
plt.ylabel('x')
plt.legend()
plt.grid(True)
plt.title('SVM Decision Boundary and Points')
plt.show()