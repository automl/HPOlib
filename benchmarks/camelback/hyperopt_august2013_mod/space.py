from hyperopt import hp

space = {'x': hp.uniform('x', -2, 2),
         'y': hp.uniform('y', -1, 1)}