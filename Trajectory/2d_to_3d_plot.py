# importing mplot3d toolkits, numpy and matplotlib
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()

# syntax for 3-D projection
ax = plt.axes(projection='3d')

# defining all 3 axes
z = np.linspace(0, 100, 33)  # num represents number of points
x = [40, 40, 40, 40, 40, 39, 38, 37, 36, 35, 34, 32, 30, 29, 28, 26, 24, 22, 19, 16, 13, 10, 8, 6, 4, 0, 4, 8, 10, 13,
     16, 18, 20]
y = [40, 40, 40, 40, 40, 39, 38, 37, 36, 35, 34, 32, 30, 29, 28, 26, 24, 22, 19, 18, 15, 10, 8, 6, 4, 0, 4, 8, 10, 13,
     16, 18, 20]

# Plotting the line
ax.plot3D(z, x, y, 'red')

# Adding image on XY plane
img = plt.imread('Images/cricket_pitch.jpeg')  # Change 'your_image_path.png' to the path of your image
x_img = np.linspace(0, 100, img.shape[1])
y_img = np.linspace(0, 100, img.shape[0])
x_img, y_img = np.meshgrid(x_img, y_img)

# Normalize image values
img_normalized = img.astype(float) / 255.0

ax.plot_surface(x_img, y_img, np.zeros_like(x_img), rstride=1, cstride=1, facecolors=img_normalized)

ax.set_title('3D line plot with Image on XY plane')
plt.show()
