import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib._png import read_png
import cv2 as cv

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set labels for axes
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Load the image

# Define data for the plot
x = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120])
y = np.ones(15)
z = np.array([50, 45, 40, 35, 30, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])

fn = get_sample_data('Images/pitch.jpeg', asfileobj=False)
img = read_png(fn)

# Plot the line
plt.subplot(1,1,1)
ax.plot(x, y, z, c='red')
ax.plot_surface(x, y, np.sin(0.02*X)*np.sin(0.02*Y), rstride=2, cstride=2,facecolors=img)

# Set the image as the background on the XY plane
plt.subplot(1,1,1)
image = plt.imread('Images/pitch.jpeg')
fig, ax = plt.subplots()
ax.imshow(image, extent=[-5, 80, -5, 30])
# ax.scatter(TMIN, PRCP, color="#ebb734")
plt.show()

