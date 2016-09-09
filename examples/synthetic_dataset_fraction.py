import sys

gh_root= '/ihome/sfalkner/repositories/github/'
sys.path.extend([gh_root + 'RoBO/', gh_root + 'HPOlibConfigSpace/'])
sys.path.extend([gh_root + 'HPOlib/'])


import numpy as np

import hpolib.benchmarks.synthetic_functions as hpobench


import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D



# Let's use the 1d Forrester function and add some artifical, dateset size dependend noise
f  = hpobench.SyntheticNoiseAndCost(hpobench.Forrester(), 0, 2, 0.1, 1, 2, 2)



# grid for plotting
X = np.linspace(0,1,50)
d = np.linspace(0,1,50)
X, D = np.meshgrid(X, d)


# compute target values and costs
T = []
C = []

for x,d in zip(X.flatten(), D.flatten()):
	mew = f.objective_function([x], dataset_fraction=d)
	T.append(mew['function_value'])
	C.append(mew['cost'])
T = np.array(T).reshape(X.shape)
C = np.array(C).reshape(X.shape)

# make some descent looking plots
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.set_title('The function in the \'extended space\'')
ax.set_xlabel('x')
ax.set_ylabel('dataset fraction')
ax.set_zlabel('f(x,dataset_fraction')
surf = ax.plot_surface(X, D, T, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


ax = fig.add_subplot(122, projection='3d')
ax.set_title('The associated cost')
ax.set_xlabel('x')
ax.set_ylabel('dataset fraction')
ax.set_zlabel('cost')
surf = ax.plot_surface(X, D, C, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.show()
