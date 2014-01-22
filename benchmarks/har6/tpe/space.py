from hyperopt import hp
space = {"x": hp.uniform('x', 0, 1),
         "y": hp.uniform('y', 0, 1),
         "z": hp.uniform('z', 0, 1),
         "a": hp.uniform('a', 0, 1),
         "b": hp.uniform('b', 0, 1),
         "c": hp.uniform('c', 0, 1)}

