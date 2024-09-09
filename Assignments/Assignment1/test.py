import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 4]
# Create sample data for the contour plot
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Create the contour plot
plt.contour(X, Y, Z)

# Set limits to crop the map
# plt.xlim(-3, 3)  # Crop the x-axis
# plt.ylim(-3, 3)  # Crop the y-axis

plt.title('Cropped Contour Map')
plt.savefig('test.png')
