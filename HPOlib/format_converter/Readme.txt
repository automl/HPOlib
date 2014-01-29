Before translating any search spaces, reading this is highly recommended:

    * GENERAL
    * TPE->SMAC

+++++++++++++++++++++++++++++++++++++++

GENERAL
    * Param names to avoid

########################
# Param names to avoid #
########################

Due to be able to translate hyperopt priors some param names will cause unexpected behaviour.
Avoid using the following strings as parts of parameter names:

LOG
LOG2
LOG10
Q1, Q2, ..., Q999

also avoid leading underscores

+++++++++++++++++++++++++++++++++++++++

HYPEROPT -> SMAC

Content:
    * Search space declaration
    * Categorical parameter
    * Different keys in dicts
    * Priors

############################
# Search space declaration #
############################

A search space needs to be python script with a variable called 'space' which contains the
whole search space. All other variables are ignored.

#########################
# Categorical parameter #
#########################

Categorical string values, like in
---------------------------------------
space = hp.choice('case', [
    {'case': 'x'},
    {'case': 'y'}, ])
---------------------------------------
which would be for smac:
---------------------------------------
case {'x', 'y'} [x]
---------------------------------------
will be translated to ints, which needs to be taken into account when using an algorithm wrapper
---------------------------------------
case {0, 1} [0]
---------------------------------------

###########################
# Different keys in dicts #
###########################

When translating hyperopt search spaces the following needs to be considered.
Having this search space
---------------------------------------
from hyperopt import hp

c = hp.randint('c', 10)

space = hp.choice('case', [
    {'user_var': 'x',
     'x': hp.normal('x', 0, 1),
     'c': c},
    {'user_var': 'y',
     'y': hp.uniform('y', 1, 3),
     'c': c}])
---------------------------------------
with this tree structure
---------------------------------------
            case
[user_var=x]/   \[user_var=y]
           / \ / \
          x   c   y
---------------------------------------
Parameter 'user_var' will not show up in the SMAC search space as it is not necessary.
It will be translated to:
---------------------------------------
case {0, 1} [0]
x [-3, 3] [0]
x | case in {0}
c {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} [0]
c | case in {0,1}
y [1, 3] [2.0]
y | case in {1}
---------------------------------------
with this tree structure
---------------------------------------
        case
     [0]/   \[1]
       / \ / \
      x   c   y
---------------------------------------
which is basically the same search space with all strings transformed to numbers and no 'user_var'

###################
# Hyperopt priors #
###################

The hyperopt priors will be translated like in the following table:

hp.choice('x', 'optionA', 'optionB')        | x {optionA, optionB} [optionA]
hp.pchoice('x', [(0.02, 0), (0.98, 1)])     | x {0, 1} [0]
hp.randint('x', 5)                          | x {0, 1, 2, 3, 4} [0]
hp.uniform('x', 0, 5)                       | x [0, 5]
hp.quniform(label, 0, 100, 3)               | Q3_x [0, 100] [50]
hp.loguniform('x', ln(1), ln(1000))         | LOG_x [0, 6.907] [3.4548]
hp.qloguniform('x', ln(1), ln(1000)), 3)    | LOG_Q3_x [0, 6.907] [3.4538]
hp.normal('x', mu, sigma)                   | x [ mu-3*sigma, mu+3*sigma ] [mu]
hp.qnormal('x', mu, sigma, 3)               | Q3_x [ mu-3*sigma, mu+3*sigma ] [mu]
hp.lognormal('x', mu, sigma)                | LOG_x [ mu-3*sigma, mu+3*sigma ] [mu]
hp.qlognormal('x', mu, sigma, 3)            | LOG_Q3_x [ mu-3*sigma, mu+3*sigma ] [mu]
hp.uniform('LOG10_x', log10(0), log10(5))   | x [log10(0), log10(10)]l

*LOG* in the param name: np.exp(float(x))
*Qq* in the param name:  round(float(x)/q)*q

Param names given to the wrapper will be without that extension, e.g. 'LOG_Q16_param' will be 'param'

############
# Pitfalls #
############

Do not use label hyperparameters, these are of kind:
'key': 'value'
They will not be traslated.