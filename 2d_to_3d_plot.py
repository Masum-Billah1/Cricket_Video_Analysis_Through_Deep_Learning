# importing mplot3d toolkits, numpy and matplotlib
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()

# syntax for 3-D projection
ax = plt.axes(projection ='3d')

# defining all 3 axis
z = np.linspace(0, 100, 33) #num represents number of point
print(z)
x = [40,40,40,40,40,39,38,37,36,35,34,32,30,29,28,26,24,22,19,16,13,10,8,6,4,0,4,8,10,13,16,18,20]
y = [40,40,40,40,40,39,38,37,36,35,34,32,30,29,28,26,24,22,19,18,15,10,8,6,4,0,4,8,10,13,16,18,20]
print(len(x))

# plotting
ax.plot3D(z,x, y, 'green')
ax.set_title('3D line plot ')
plt.show()
