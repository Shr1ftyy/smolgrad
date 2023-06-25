#ifndef NN
#define NN

#include "hashmap/hashmap.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

//// GLOBALS ////
uint64_t NN_VAR_ID = 0;

//// TYPES /////

// A scalar value
typedef struct Variable
{
    double val;                 // double value
    struct Variable **children; // array of pointers to child variables
    double *local_grads;        // accompanying local grads
    int n_children;             // number of children
    bool can_grad;              // can we perform gradient updates on the value during backpropagation?
    uint64_t id;                // unique id (used for hashing)
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
    struct hashmap *dep_grad_map;
    Variable **independent_vars; // independent var whose gradient we are calculating
    Variable **dependent_vars;   // independent var whose gradient we are calculating
    size_t n_indep_vars;
    size_t n_dep_vars;
} VariablesGradAllocator;

// Initialize variables
void init_var(Variable *var, double value, bool grad)
{
    var->val = value;
    var->children = NULL;
    var->local_grads = NULL;
    var->n_children = 0;
    var->can_grad = grad;
    var->id = NN_VAR_ID;
    NN_VAR_ID++;
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

void print_variable(const Variable *var)
{
    printf("val: %lf, n_children: %d, children: %p, local_grads: %p\n", var->val,
           var->n_children, (void *)var->children, (void *)var->local_grads);
}

// Hash function for Variable struct
uint64_t var_hash(const void *item, uint64_t seed0, uint64_t seed1)
{
    const Variable *variable = (const Variable *)item;
    return hashmap_sip(&(variable->id), sizeof(variable->id), seed0, seed1);
    // return variable->id; // TODO: Use the id field as the hash???
}

// Key comparison function for Variable struct
int var_compare(const void *a, const void *b, void *udata)
{
    const Variable *var_a = (const Variable *)a;
    const Variable *var_b = (const Variable *)b;
    if (var_a->id < var_b->id)
    {
        return -1;
    }
    else if (var_a->id > var_b->id)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

bool var_iter(const void *item, void *var_data)
{
    const Variable *var = item;
    print_variable(var);
    return true;
}

// Initialize gradient buffer
void init_grad_alloc(VariablesGradAllocator *var_grad_allocator)
{
    VariablesGradAllocator temp = {0};
    *(var_grad_allocator) = temp;
    var_grad_allocator->independent_vars = (Variable **)malloc(sizeof(Variable *));
    var_grad_allocator->dependent_vars = (Variable **)malloc(sizeof(Variable *));
    struct hashmap *map = hashmap_new(sizeof(Variable *), 0, 0, 0, var_hash,
                                      var_compare, NULL, NULL);
    if (map == NULL)
    {
        // TODO LOL, is his bad practice?
        printf("failed to initialize map!");
    }
    var_grad_allocator->dep_grad_map = map;
}

// Free gradient buffer
void free_grad_alloc(VariablesGradAllocator *var_grad_allocator, size_t size)
{
    for (size_t idx = 0; idx < var_grad_allocator->n_indep_vars; idx++)
    {
        free_from_variable(var_grad_allocator->independent_vars[idx]);
    }

    // TODO: does this not already free all all of the dependent_vars?
    hashmap_free(var_grad_allocator->dep_grad_map);

    for (size_t idx = 0; idx < var_grad_allocator->n_dep_vars; idx++)
    {
        free_from_variable(var_grad_allocator->dependent_vars[idx]);
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
    Tensor *tensor = malloc(sizeof tensor);
    tensor->shape_size = shape_size;
    tensor->shape = malloc(shape_size * sizeof(size_t));
    tensor->strides = malloc(shape_size * sizeof(int));

    int total_size = 1;
    for (int i = 0; i < shape_size; ++i)
    {
        tensor->shape[i] = shape[i];
        tensor->strides[i] = total_size;
        total_size *= shape[i];
    }

    tensor->variables = malloc(total_size * sizeof(Variable *));
    return tensor;
}

//// SCALAR OPS /////

// Builds the compute graph for addition between scalar variables
Variable *add(Variable *x, Variable *y)
{
    // Allocate memory for children and local gradients
    Variable **children = malloc((sizeof *children) * 2);
    double *local_grads = malloc((sizeof *local_grads) * 2);

    // Set up children and local gradients
    children[0] = x;
    local_grads[0] = 1;
    children[1] = y;
    local_grads[1] = 1;

    // Create a new Variable with the computed value and connections
    Variable *newValue = malloc(sizeof *newValue);
    init_var(newValue, x->val + y->val, false);
    newValue->children = children;
    newValue->local_grads = local_grads;
    newValue->n_children = 2;

    return newValue;
}

// Builds the compute graph for subtraction between scalar variables
Variable *sub(Variable *x, Variable *y)
{
    // Allocate memory for children and local gradients
    Variable **children = malloc((sizeof *children) * 2);
    double *local_grads = malloc((sizeof *local_grads) * 2);

    // Set up children and local gradients
    children[0] = x;
    local_grads[0] = 1;
    children[1] = y;
    local_grads[1] = 1;

    // Create a new Variable with the computed value and connections
    Variable *newValue = malloc(sizeof *newValue);
    init_var(newValue, x->val - y->val, false);
    newValue->children = children;
    newValue->local_grads = local_grads;
    newValue->n_children = 2;

    return newValue;
}

// Builds the compute graph for multiplication between scalar variables
Variable *mul(Variable *x, Variable *y)
{
    // Allocate memory for children and local gradients
    Variable **children = malloc((sizeof *children) * 2);
    double *local_grads = malloc((sizeof *local_grads) * 2);

    // Set up children and local gradients
    children[0] = x;
    local_grads[0] = y->val;
    children[1] = y;
    local_grads[1] = x->val;

    // Create a new Variable with the computed value and connections
    Variable *newValue = malloc(sizeof *newValue);
    init_var(newValue, x->val * y->val, false);
    newValue->children = children;
    newValue->local_grads = local_grads;
    newValue->n_children = 2;

    return newValue;
}

// Builds the compute graph for the sigmoid function on a scalar variable
Variable *sigmoid(Variable *x)
{
    // Allocate memory for children and local gradients
    Variable **children = malloc((sizeof *children) * 1);
    double *local_grads = malloc((sizeof *local_grads) * 1);

    // Set up children and local gradients
    children[0] = x;
    double sigmoid_val = 1 / (1 + exp(-(x->val)));
    local_grads[0] = sigmoid_val * (1 - sigmoid_val);

    // Create a new Variable with the computed value and connections
    Variable *newValue = malloc(sizeof *newValue);
    init_var(newValue, sigmoid_val, false);
    newValue->children = children;
    newValue->local_grads = local_grads;
    newValue->n_children = 1;

    return newValue;
}

// Builds the compute graph for the ReLU function on a scalar variable
Variable *relu(Variable *x)
{
    // Allocate memory for children and local gradients
    Variable **children = malloc((sizeof *children) * 1);
    double *local_grads = malloc((sizeof *local_grads) * 1);

    // Set up children and local gradients
    children[0] = x;
    local_grads[0] = x->val < 0 ? 0 : 1;

    // Create a new Variable with the computed value and connections
    Variable *newValue = malloc(sizeof *newValue);
    init_var(newValue, (x->val > 0) ? x->val : 0, false);
    newValue->children = children;
    newValue->local_grads = local_grads;
    newValue->n_children = 1;

    return newValue;
}

Variable *power(Variable *x, double n)
{
    // Allocate memory for children and local gradients
    Variable **children = malloc((sizeof *children) * 1);
    double *local_grads = malloc((sizeof *local_grads) * 1);

    // Set up children and local gradients
    children[0] = x;
    local_grads[0] = n * pow(x->val, n - 1);

    // Create a new Variable with the computed value and connections
    Variable *newValue = malloc(sizeof *newValue);
    init_var(newValue, pow(x->val, n), false);
    newValue->children = children;
    newValue->local_grads = local_grads;
    newValue->n_children = 1;

    return newValue;
}

// Helper function for get_gradients. Computes grads of all children of a
// Variable
void compute_grads(VariablesGradAllocator *grad_alloc,
                   Variable *independent_var, double path_value)
{
    // TODO: Unnecessary?
    if (independent_var == NULL || grad_alloc == NULL || independent_var->children == NULL)
    {
        return;
    }

    for (int child_idx = 0; child_idx < independent_var->n_children;
         child_idx++)
    {
        Variable *dependent_var = independent_var->children[child_idx];
        double path_to_child_val =
            path_value * independent_var->local_grads[child_idx];

        uint64_t dep_hash = var_hash(dependent_var, 0, 0);
        Variable v = {0};
        Variable *g_var = (Variable *)hashmap_get_with_hash(grad_alloc->dep_grad_map, &v, dep_hash);

        if (g_var == NULL)
        {
            if (!dependent_var->can_grad)
            {
                break;
            }

            Variable *new_var = malloc(sizeof *new_var);
            init_var(new_var, path_to_child_val, false);
            const void *set = hashmap_set_with_hash(grad_alloc->dep_grad_map, new_var, dep_hash);
            if (set == NULL)
            {
                size_t count = hashmap_count(grad_alloc->dep_grad_map);
#ifdef DEBUG
                printf("successfully inserted new grad at hash: %llu\n", dep_hash);
                printf("# items in hash: %lu\n", count);
#endif
                grad_alloc->dependent_vars = (Variable **)realloc(grad_alloc->dependent_vars, count * sizeof(Variable *));
                grad_alloc->dependent_vars[count - 1] = dependent_var;
                grad_alloc->n_dep_vars++;
            }
#ifdef DEBUG
            else
            {

                printf("insert failed at hash: %llu\n", dep_hash);
            }
#endif
        }
        else
        {
#ifdef DEBUG

            printf("val: %f ", path_to_child_val);
            printf("\nis already occupying hash: %llu\n", dep_hash);
#endif
            g_var->val += path_to_child_val;
        }

        compute_grads(grad_alloc, dependent_var, path_to_child_val);
    }
#ifdef DEBUG
    printf("____________________\n");
#endif
}

// Compute gradients of outs, store pointers to Values with
// val ∂ outs/ ∂w_i in buffer, where w_i is a node in the
// compute DAG which produces outputs outs
void get_gradients(VariablesGradAllocator *grad_alloc,
                   Variable **independent_vars, int n_vars)
{
    if(grad_alloc == NULL || independent_vars == NULL){
        return;
    }
    
    for (int root_index = 0; root_index < n_vars; root_index++)
    {
        Variable *indep_var = independent_vars[root_index];

        grad_alloc->independent_vars =
            (Variable **)realloc(grad_alloc->independent_vars, (root_index + 1) * sizeof(Variable *));
        grad_alloc->independent_vars[root_index] = indep_var;
        grad_alloc->n_indep_vars++;

        compute_grads(grad_alloc, indep_var, 1.0);
    }
}

//// TENSOR OPS /////
// TODO: write tensor ops

#endif // NN