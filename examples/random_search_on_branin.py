import numpy as np

from hpolib.benchmarks.synthetic_functions import Branin


b = Branin()

values = []

cs = b.get_configuration_space()

for i in range(1000):
    configuration = cs.sample_configuration()
    rval = b.objective_function(configuration)
    loss = rval['function_value']
    values.append(loss)

print(np.min(values))
