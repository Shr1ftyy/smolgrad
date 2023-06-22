a smol nn lib written in C w/ a scalar valued autograd engine

include the header file from src/NN.h and you're set (⌐■_■)

What you got bro?
- Reverse-mode automatic differentiation over a dynamically built DAG.
- DAG operates over scalar values
- Scalar values implemented via `Value`
``` c
typedef struct Value {
  double val; // double value
  struct Value** children; // array of pointers to child values
  double* local_grads; // accompanying local grads
  int n_children;
} Value;
```
- Scalar Ops on Two `Value`s (`add`, `mul`, `sub`, `sigmoid`, etc.)
- Tensor/Matrix/Vector Ops (coming soon)
- Wrappers for NN stuff (coming soon)

TODO: \
🟡 = in progress, 🟢 = done

- 🟢 write tests for scalar ops
- 🟡 implement backprop
- implement tensor ops

