from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
dim = 3
points = [3, 6, 7, 5, 3, 5, 6, 2, 9, 1, 2, 7, 0, 9, 3, 6, 0, 6, 2, 6, 1, 8, 7, 9, 2, 0, 2, 3, 7, 5, 9, 2, 2, 8, 9, 7, 3, 6, 1, 2, 9, 3, 1, 9, 4, 7, 8, 4, 5, 0, 3, 6, 1, 0, 6, 3, 2, 0, 6, 1, 5, 5, 4, 7, 6, 5, 6, 9, 3, 7, 4, 5, 2, 5, 4, 7, 4, 4, 3, 0, 7, 8, 6, 8, 8, 4, 3, 1, 4, 9, 2, 0, 6, 8, 9, 2, 6, 6, 4, 9, 5, 0, 4, 8, 7, 1, 7, 2, 7, 2, 2, 6, 1, 0, 6, 1, 5, 9, 4, 9, 0, 9, 1, 7, 7, 1, 1, 5, 9, 7, 7, 6, 7, 3, 6, 5, 6, 3, 9, 4, 8, 1, 2, 9, 3, 9, 0, 8, 8, 5, 0, 9, 6, 3, 8, 5, 6, 1, 1, 5, 9, 8, 4, 8, 1, 0, 3, 0, 4, 4, 4, 4, 7, 6, 3, 1, 7, 5, 9, 6, 2, 1, 7, 8, 5, 7, 4, 1, 8, 5, 9, 7, 5, 3, 8, 8, 3, 1, 8, 9]
points_to_centroids = [2,2,2,2,0,2,0,2,0,0,2,1,0,0,0,1,2,2,2,0,2,1,1,2,0,2,2,2,2,2,2,1,1,1,1,0,2,2,2,2,0,1,0,1,2,1,2,2,0,1,0,0,2,1,0,0,2,1,2,1,2,2,2,1,2,2]

for i in range(int(len(points)/dim)):
  if(points_to_centroids[i] == 0):
    ax.scatter(points[i], points[i+1], points[i+2], marker='o', color='red')
  elif(points_to_centroids[i] == 1):
    ax.scatter(points[i], points[i+1], points[i+2], marker='o', color='green')
  elif(points_to_centroids[i] == 2):
    ax.scatter(points[i], points[i+1], points[i+2], marker='o', color='blue')



ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

#2d

p2c = [1,0,1,2,0,1,1,1,1,1,1,1,2,2,0,0,1,1,1,2,0,1,1,0,2,1,2,2,2,2,1,1,1,1,1,1,1,1,2,2,1,1,0,2,1,2,1,0,1,1,2,1,0,0,0,1,2,2,1,1,1,1,0,1,1,1,0,1,2,0,0,1,1,1,0,1,2,0,2,1,1,1,2,2,2,2,1,2,0,0,2,1,1,2,0,1,2,1,2]
points = [3, 6, 7, 5, 3, 5, 6, 2, 9, 1, 2, 7, 0, 9, 3, 6, 0, 6, 2, 6, 1, 8, 7, 9, 2, 0, 2, 3, 7, 5, 9, 2, 2, 8, 9, 7, 3, 6, 1, 2, 9, 3, 1, 9, 4, 7, 8, 4, 5, 0, 3, 6, 1, 0, 6, 3, 2, 0, 6, 1, 5, 5, 4, 7, 6, 5, 6, 9, 3, 7, 4, 5, 2, 5, 4, 7, 4, 4, 3, 0, 7, 8, 6, 8, 8, 4, 3, 1, 4, 9, 2, 0, 6, 8, 9, 2, 6, 6, 4, 9, 5, 0, 4, 8, 7, 1, 7, 2, 7, 2, 2, 6, 1, 0, 6, 1, 5, 9, 4, 9, 0, 9, 1, 7, 7, 1, 1, 5, 9, 7, 7, 6, 7, 3, 6, 5, 6, 3, 9, 4, 8, 1, 2, 9, 3, 9, 0, 8, 8, 5, 0, 9, 6, 3, 8, 5, 6, 1, 1, 5, 9, 8, 4, 8, 1, 0, 3, 0, 4, 4, 4, 4, 7, 6, 3, 1, 7, 5, 9, 6, 2, 1, 7, 8, 5, 7, 4, 1, 8, 5, 9, 7, 5, 3, 8, 8, 3, 1, 8, 9]
N = 200

x = np.array(points[::2])
y = np.array(points[1::2])
colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
x_a = []
x_b = []
y_a = []
y_b = []
for i in range(len(p2c)):
  if p2c[i] == 1:
    plt.scatter(x[i], y[i], c='red')
  elif p2c[i] == 0:
    plt.scatter(x[i], y[i], c='blue')
  else: 
    plt.scatter(x[i], y[i], c='green')

plt.show()