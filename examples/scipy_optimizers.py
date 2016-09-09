import sys

gh_root= '/ihome/sfalkner/repositories/github/'
sys.path.extend([gh_root + 'RoBO/', gh_root + 'HPOlibConfigSpace/'])
sys.path.extend([gh_root + 'HPOlib/'])


import numpy as np
import scipy.optimize as spopt


import hpolib.benchmarks.synthetic_functions as hpobench


h3 = hpobench.Hartmann3()
h6 = hpobench.Hartmann6()
b = hpobench.Branin()

for f in [b,h3, h6]:

	info = f.get_meta_information()

	print("="*50)
	print(info['name'])
	
	bounds = np.array(info['bounds'])
	
	res = spopt.minimize(f, (bounds[:,0]+bounds[:,1])/2, bounds=bounds, method='SLSQP')

	assert(np.allclose(res.fun, info['f_opt'] ))

	for o in info['optima']:
		print("There is an optimum at \t x={} with f(x) = {}".format(o, info['f_opt']))
	print("scipy.optimize found \t x={} with f(x) = {}".format(res.x, res.fun))
