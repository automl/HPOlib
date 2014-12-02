from hyperopt import hp
from hyperopt.pyll import scope

space = {'lrate': scope.int(hp.quniform('lrate', -0.50001, 10.49999, 1)),
         'l2_reg': scope.int(hp.quniform('l2_reg', -0.50001, 5.49999, 1)),
         'batchsize': scope.int(hp.quniform('batchsize', -0.50001, 7.49999, 1)),
         'n_epochs': scope.int(hp.quniform('n_epochs', -0.50001, 9.49999, 1))}
