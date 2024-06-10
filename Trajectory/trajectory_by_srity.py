import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg

import cv2 as cv

# Load the background image
img = cv.imread("img_1.jpg")
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert BGR to RGB
print(img_rgb.shape)

# Example data for x, y, and z positions for multiple trajectories
x_positions = np.linspace(0, 22, 100)  # Assuming 22 yards for a cricket pitch length
y_positions = [np.linspace(-1, 1, 100) for _ in range(3)]  # Example y positions, replace with actual data
z_positions = [4 * np.sin(np.pi * x_positions / 22 + i * np.pi / 6) for i in range(3)]  # Example z positions

colors = ['blue', 'red', 'yellow']  # Colors for different trajectories

# Create the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define the coordinates of the plane where the image will be plotted
X, Y = np.meshgrid(np.linspace(0, 22, img.shape[1]), np.linspace(-3, 3, img.shape[0]))
Z = np.zeros_like(X)

# Plot the background image on the XY plane
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=img/255, shade=False)

# Plot the 3D trajectories
for i in range(len(y_positions)):
    ax.plot(x_positions, y_positions[i], z_positions[i], color=colors[i], linewidth=2, label=f'Trajectory {i+1}')

# Labeling the axes
ax.set_xlabel('X position (yards)')
ax.set_ylabel('Y position (width)')
ax.set_zlabel('Z position (height)')

# Adding a title and legend
ax.set_title('3D Trajectory of a Cricket Ball')
ax.legend()

# Set the limits and view angle
ax.set_xlim(0, 22)
ax.set_ylim(-3, 3)
ax.set_zlim(0, 5)
ax.view_init(elev=20, azim=60)

# Display the plot
plt.show()
