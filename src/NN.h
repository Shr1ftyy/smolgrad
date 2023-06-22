#ifndef NN
#define NN

#include <math.h>
#include <stdlib.h>

//// TYPES /////

// A scalar value
typedef struct Variable
{
  double val;                 // double value
  struct Variable **children; // array of pointers to child variables
  double *local_grads;        // accompanying local grads
  int n_children;
} Variable;

// A "tensor"
typedef struct Tensor
{
  Variable **variables; // array of variables, indexing is determined by shape
  size_t *shape;        // shape of tensor
  size_t shape_size;    // length of shape array
  int *strides;         // for indexing in each dim.
} Tensor;

typedef struct VariablesGradAllocator
{
  Variable *independent_var; // independent var whose gradient we are calculating
  Variable **dependent_vars; // Variables whose values that have to be updated
  Variable **gradients;
  size_t size;
} VariablesGradAllocator;

// Initialize variables
void init_var(Variable *var, double value)
{
  var->val = value;
  var->children = NULL;
  var->local_grads = 0;
  var->n_children = 0;
}

// Initialize gradient buffer
void init_grad_bufs(VariablesGradAllocator **var_grad_allocator, size_t size)
{
  for (size_t alloc_idx = 0; alloc_idx < size; alloc_idx++)
  {
    var_grad_allocator[alloc_idx] = (VariablesGradAllocator *)malloc(sizeof(VariablesGradAllocator));
    var_grad_allocator[alloc_idx]->independent_var = NULL;
    var_grad_allocator[alloc_idx]->dependent_vars = NULL;
    var_grad_allocator[alloc_idx]->gradients = NULL;
    var_grad_allocator[alloc_idx]->size = 0;
  }
}

void free_variable(Variable *var)
{
  if (var->local_grads != NULL)
  {
    free(var->local_grads);
    var->local_grads = NULL;
  }
  if (var->children != NULL)
  {
    free(var->children);
    var->children = NULL;
  }
  var = NULL;
}

// free a variable and all of its children
void free_from_variable(Variable *var)
{
  if (var == NULL)
  {
    return;
  }

  for (int i = 0; i < var->n_children; i++)
  {
    free_from_variable(var->children[i]);
  }

  free_variable(var);

  return;
}

// TODO: any mem leaks?
void free_graph(Variable **outs, size_t n_vars)
{
  for (int i = 0; i < n_vars; i++)
  {
    free_from_variable(outs[i]);
  }
}

// index a tensor
int tensor_index(Tensor *tensor, int *indices)
{
  int index = 0;
  for (int i = 0; i < tensor->shape_size; ++i)
  {
    index += tensor->strides[i] * indices[i];
  }
  return index;
}

// initializing an empty tensor with shape
Tensor *init_tensor(size_t shape_size, size_t *shape)
{
  Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
  tensor->shape_size = shape_size;
  tensor->shape = (size_t *)malloc(shape_size * sizeof(size_t));
  tensor->strides = (int *)malloc(shape_size * sizeof(int));

  int total_size = 1;
  for (int i = 0; i < shape_size; ++i)
  {
    tensor->shape[i] = shape[i];
    tensor->strides[i] = total_size;
    total_size *= shape[i];
  }

  tensor->variables = (Variable **)malloc(total_size * sizeof(Variable *));
  return tensor;
}

//// SCALAR OPS /////

// Builds the compute graph for addition between scalar variables
Variable *add(Variable *x, Variable *y)
{
  // Allocate memory for children and local gradients
  Variable **children = malloc(sizeof(Variable *) * 2);
  double *local_grads = malloc(sizeof(double) * 2);

  // Set up children and local gradients
  children[0] = x;
  local_grads[0] = 1;
  children[1] = y;
  local_grads[1] = 1;

  // Create a new Variable with the computed value and connections
  Variable *newValue = (Variable *)malloc(sizeof(Variable));
  newValue->val = x->val + y->val;
  newValue->children = children;
  newValue->local_grads = local_grads;
  newValue->n_children = 2;

  return newValue;
}

// Builds the compute graph for subtraction between scalar variables
Variable *sub(Variable *x, Variable *y)
{
  // Allocate memory for children and local gradients
  Variable **children = malloc(sizeof(Variable *) * 2);
  double *local_grads = malloc(sizeof(double) * 2);

  // Set up children and local gradients
  children[0] = x;
  local_grads[0] = 1;
  children[1] = y;
  local_grads[1] = 1;

  // Create a new Variable with the computed value and connections
  Variable *newValue = (Variable *)malloc(sizeof(Variable));
  newValue->val = x->val - y->val;
  newValue->children = children;
  newValue->local_grads = local_grads;
  newValue->n_children = 2;

  return newValue;
}

// Builds the compute graph for multiplication between scalar variables
Variable *mul(Variable *x, Variable *y)
{
  // Allocate memory for children and local gradients
  Variable **children = malloc(sizeof(Variable *) * 2);
  double *local_grads = malloc(sizeof(double) * 2);

  // Set up children and local gradients
  children[0] = x;
  local_grads[0] = y->val;
  children[1] = y;
  local_grads[1] = x->val;

  // Create a new Variable with the computed value and connections
  Variable *newValue = (Variable *)malloc(sizeof(Variable));
  newValue->val = x->val * y->val;
  newValue->children = children;
  newValue->local_grads = local_grads;
  newValue->n_children = 2;

  return newValue;
}

// Builds the compute graph for the sigmoid function on a scalar variable
Variable *sigmoid(Variable *x)
{
  // Allocate memory for children and local gradients
  Variable **children = malloc(sizeof(Variable *) * 1);
  double *local_grads = malloc(sizeof(double) * 1);

  // Set up children and local gradients
  children[0] = x;
  double sigmoid_val = 1 / (1 + exp(-(x->val)));
  local_grads[0] = sigmoid_val * (1 - sigmoid_val);

  // Create a new Variable with the computed value and connections
  Variable *newValue = (Variable *)malloc(sizeof(Variable));
  newValue->val = sigmoid_val;
  newValue->children = children;
  newValue->local_grads = local_grads;
  newValue->n_children = 1;

  return newValue;
}

// Builds the compute graph for the ReLU function on a scalar variable
Variable *relu(Variable *x)
{
  // Allocate memory for children and local gradients
  Variable **children = malloc(sizeof(Variable *) * 1);
  double *local_grads = malloc(sizeof(double) * 1);

  // Set up children and local gradients
  children[0] = x;
  local_grads[0] = x->val < 0 ? 0 : 1;

  // Create a new Variable with the computed value and connections
  Variable *newValue = (Variable *)malloc(sizeof(Variable));
  newValue->val = (x->val > 0) ? x->val : 0;
  newValue->children = children;
  newValue->local_grads = local_grads;
  newValue->n_children = 1;

  return newValue;
}

// Helper function for get_gradients. Computes grads of all children of a Variable
void compute_grads(VariablesGradAllocator *var_grad_store, Variable *independent_var, int path_value)
{
  // TODO: Unnecessary?
  if (independent_var == NULL || var_grad_store == NULL)
  {
    return;
  }

  for (int child_index = 0; child_index < independent_var->n_children; child_index++)
  {
    Variable *var = independent_var->children[child_index];
    int path_to_child_val = var->local_grads != NULL ? path_value * var->local_grads[child_index] : path_value;

    // Store pointer(s) to dependent variable(s).
    // These will be used later to apply gradients to them
    if (var_grad_store->size <= child_index)
    {
      var_grad_store->dependent_vars = (Variable **)realloc(var_grad_store->dependent_vars, child_index + 1);
      var_grad_store->gradients = (Variable **)realloc(var_grad_store->gradients, child_index + 1);
    }

    // TODO: I'm using "+=" here, are the values for the ints here 0 after malloc?
    // Because if they aren't, this will break 💀
    var_grad_store->dependent_vars[child_index] = var;

    if (var_grad_store->gradients[child_index] == NULL)
    {
      Variable *new_var = (Variable *)malloc(sizeof(Variable));
      Variable **new_vars_arr = (Variable **)malloc(sizeof(Variable **));
      init_var(new_var, 0);
      new_vars_arr[0] = new_var;
      var_grad_store->gradients[child_index] = new_vars_arr[0];
      var_grad_store->size++;
    }
    else
    {
      var_grad_store->gradients[child_index]->val += path_to_child_val;
    }

    compute_grads(var_grad_store, var, path_to_child_val);
  }
}

// Compute gradients of outs, store pointers to Values with
// val ∂ outs/ ∂w_i in buffer, where w_i is a node in the
// compute DAG which produces outputs outs
void get_gradients(VariablesGradAllocator **grad_buf, Variable **independent_vars, int n_vars)
{
  for (int root_index = 0; root_index < n_vars; root_index++)
  {
    Variable *indep_var = independent_vars[root_index];

    grad_buf = (VariablesGradAllocator **)realloc(grad_buf, (root_index + 1) * sizeof(VariablesGradAllocator *));
    grad_buf[root_index]->independent_var = independent_vars[root_index];

    compute_grads(grad_buf[root_index], indep_var, 1);
  }
}

//// TENSOR OPS /////
// TODO: write tensor ops

#endif // NN