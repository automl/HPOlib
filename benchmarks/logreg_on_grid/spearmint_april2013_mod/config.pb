language: PYTHON
name:     "spearmint_to_HPOlib"

variable {
 name: "lrate"
 type: INT
 size: 1
 min:  0
 max:  10
}

variable {
 name: "l2_reg"
 type: INT
 size: 1
 min:  0
 max:  10
}

variable {
 name: "batchsize"
 type: INT
 size: 1
 min:  0
 max:  7
}

variable {
 name: "n_epochs"
 type: INT
 size: 1
 min:  0
 max:  9
}

