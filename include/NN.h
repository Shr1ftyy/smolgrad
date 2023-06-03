#ifndef NN
#define NN

#include <math.h>
#include <stdlib.h>

// A scalar value
typedef struct Value {
  double val; // Value
  struct Value** child_values; // accompanying local grads 
  double* local_grads; // accompanying local grads 
  int n_children;
  // bool is_grad; // is backpropable (this has local grad funcs)
} Value;

Value* add(Value* x, Value* y){
  Value** child_values = malloc(sizeof(Value*) * 2);
  double* local_grads = malloc(sizeof(double) * 2);

  child_values[0] = x;
  local_grads[0] = 1;
  child_values[1] = y;
  local_grads[1] = 1;

  Value* newValue = (Value*) malloc(sizeof(Value));
  newValue->val = x->val + y->val;
  newValue->child_values = child_values;
  newValue->local_grads = local_grads;
  newValue->n_children = 2;

  return newValue;
}

Value* sub(Value* x, Value* y){
  Value** child_values = malloc(sizeof(Value*) * 2);
  double* local_grads = malloc(sizeof(double) * 2);

  child_values[0] = x;
  local_grads[0] = 1;
  child_values[1] = y;
  local_grads[1] = 1;

  Value* newValue = (Value*) malloc(sizeof(Value));
  newValue->val = x->val - y->val;
  newValue->child_values = child_values;
  newValue->local_grads = local_grads;
  newValue->n_children = 2;

  return newValue;
}

Value* mul(Value* x, Value* y){
  Value** child_values = malloc(sizeof(Value*) * 2);
  double* local_grads = malloc(sizeof(double) * 2);

  child_values[0] = x;
  local_grads[0] = y->val;
  child_values[1] = y;
  local_grads[1] = x->val;

  Value* newValue = (Value*) malloc(sizeof(Value));
  newValue->val = x->val * y->val;
  newValue->child_values = child_values;
  newValue->local_grads = local_grads;
  newValue->n_children = 2;

  return newValue;
}

Value* sigmoid(Value* x){
  Value** child_values = malloc(sizeof(Value*) * 1);
  double* local_grads = malloc(sizeof(double) * 1);

  child_values[0] = x;
  local_grads[0] = ( 1/(1+exp(-(x->val))) ) * ( 1 - (1/(1+exp(-(x->val)))) );

  Value* newValue = (Value*) malloc(sizeof(Value));
  newValue->val = 1/(1+exp(-(x->val)));
  newValue->child_values = child_values;
  newValue->local_grads = local_grads;
  newValue->n_children = 1;

  return newValue;
}

Value* relu(Value* x){
  Value** child_values = malloc(sizeof(Value*) * 1);
  double* local_grads = malloc(sizeof(double) * 1);

  child_values[0] = x;
  local_grads[0] = x->val < 0 ? 0 : 1;

  Value* newValue = (Value*) malloc(sizeof(Value));
  newValue->val = (x->val > 0) ? x->val : 0;
  newValue->child_values = child_values;
  newValue->local_grads = local_grads;
  newValue->n_children = 1;

  return newValue;
}

#endif // NN