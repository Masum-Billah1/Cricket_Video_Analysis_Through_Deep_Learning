import numpy as np
import matplotlib as npl
import matplotlib.pyplot as plt
import cv2 as cv

from mpl_toolkits.mplot3d import Axes3D

image = cv.imread('Images/Output.jpeg')

fig=plt.figure()
ax=plt.axes(111,projection='3d')

ax.set_xlabel('X Axis')

ax.set_ylabel('Y Axis')

ax.set_zlabel('Z Axis')


# x=np.arange(0,50,0.3)
x = np.array([50,55,60,65,70, 75, 80, 85, 90, 95, 100,105,110,115,120])
# y = np.array([10,10,20,25,30, 35, 30, 25, 20, 15, 10])
# y=np.arange(0,50,0.3)
y = np.ones(15)
z = np.array([50,45,40,35,30, 25, 30, 35, 40, 45, 50,55,60,65,70])


ax.plot(x,y,z,c='red')
ax.set_facecolor("green")

# ax.grid(False)
# plt.axis('off')
plt.show()