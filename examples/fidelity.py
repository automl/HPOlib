import sys

import numpy as np

import hpolib.benchmarks.synthetic_functions as hpobench


import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D



# Let's use the 1d Forrester function and add some artifical, dateset size dependend noise
f  = hpobench.Forrester()



# grid for plotting
X = np.linspace(0,1,50)
fids = np.linspace(0,1,50)
X, F = np.meshgrid(X, fids)


# compute target values
T = []

for x,g in zip(X.flatten(), F.flatten()):
	mew = f.objective_function([x], fidelity=g)
	T.append(mew['function_value'])
T = np.array(T).reshape(X.shape)

# make some descent looking plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('The Forrester function with an artificial fidelity dimension')
ax.set_xlabel('x')
ax.set_ylabel('fidelity')
ax.set_zlabel('f(x,dataset_fraction')
surf = ax.plot_surface(X, F, T, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.show()
