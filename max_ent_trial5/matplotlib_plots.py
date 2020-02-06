import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.preprocessing import normalize


trajectories_z0 = [[0, 0, 0, 0] for i in range(27)]
default_points = np.array([-0.5, 0, 0.5])
i = 0
for z in default_points:
  for y in default_points:
    for x in default_points:
      trajectories_z0[i] = [x, y, z, 0.]
      i += 1

weights = np.array(
  [
    0.90759588,
    4.77365172,
    0.8771005,
    0.43324404,
    6.33486176,
    0.87008563,
    0.60639639,
    4.76273361,
    0.17210085,
    0.02747048,
    1.08775725,
    0.46268714,
    0.39770698,
    0.42458757,
    0.4958771,
    0.82629336,
    4.68860071,
    0.20892782,
    0.77898644,
    3.32710488,
    0.22604133,
    0.7976902,
    0.7246625,
    0.60428786,
    0.78655751,
    4.74956047,
    0.20839334
  ]
)

Mat = weights.reshape((3, 3, 3), order='F')
mFlat = Mat.flatten()
print "mflat is ", mFlat
norm = mFlat/np.linalg.norm(mFlat)
print "norm is ", norm
norm_thresh = np.zeros(27)
for i in range(len(norm)):
  if norm[i] >=0.2:
    norm_thresh[i] = 1
  else:
    norm_thresh[i] = 0.1

print "norm thresh is ", norm_thresh
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(27):
  color = 'red'
  label = 'z = -0.5'
  if i >= 9 and i < 18:
    color = 'blue'
    label = 'z = 0'
  elif i >= 18:
    color = 'green'
    label = 'z = 0.5'
    # print(i)
  trajectories_z0[i][3] = norm_thresh[i]
  ax.scatter(
    trajectories_z0[i][0],
    trajectories_z0[i][1],
    trajectories_z0[i][2],
    s=trajectories_z0[i][3] * 100,
    marker="o",
    c=color
  )
ax.legend(loc=8, framealpha=1, fontsize=8)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
plt.title('Distribution of Reward Function for a custom trajectory')
fig.savefig("grie_plot")