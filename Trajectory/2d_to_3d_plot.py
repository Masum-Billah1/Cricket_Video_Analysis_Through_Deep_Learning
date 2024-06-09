# Importing necessary libraries
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# Creating a figure
fig = plt.figure(figsize=(10, 7))

# Creating 3D axes
ax = fig.add_subplot(111, projection='3d')

# Defining all 3 axes
z = np.linspace(0, 100, 33)  # num represents number of points
x = [40, 40, 40, 40, 40, 39, 38, 37, 36, 35, 34, 32, 30, 29, 28, 26, 24, 22, 19, 16, 13, 10, 8, 6, 4, 0, 4, 8, 10, 13,
     16, 18, 20]
y = [40, 40, 40, 40, 40, 39, 38, 37, 36, 35, 34, 32, 30, 29, 28, 26, 24, 22, 19, 18, 15, 10, 8, 6, 4, 0, 4, 8, 10, 13,
     16, 18, 20]

# Plotting the trajectory line
ax.plot3D(z, x, y, 'red', label='Ball Trajectory')

# Adding image on XY plane
img = plt.imread('Images/cricket_pitch.jpeg')  # Change to the path of your image
x_img = np.linspace(0, 100, img.shape[1])
y_img = np.linspace(0, 100, img.shape[0])
x_img, y_img = np.meshgrid(x_img, y_img)

# Normalize image values
img_normalized = img.astype(float) / 255.0

# Plot the image on the XY plane
ax.plot_surface(x_img, y_img, np.zeros_like(x_img), rstride=1, cstride=1, facecolors=img_normalized, alpha=0.5)

# Adding labels and title
ax.set_xlabel('Z (Distance)')
ax.set_ylabel('X (Width)')
ax.set_zlabel('Y (Height)')
ax.set_title('3D Ball Trajectory with Cricket Pitch Background')

# Adding legend
ax.legend()

# Adjusting view angle for better visualization
ax.view_init(elev=20, azim=-45)

# Show plot
plt.show()
