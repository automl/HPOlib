from hyperopt import hp

space = {'x': hp.uniform('x', -5, 10),
         'y': hp.uniform('y', 0, 15) }