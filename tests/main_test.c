#ifndef DEBUG
#define DEBUG 1
#endif

#include "NN.h"
#include <stdio.h>

void print_vars(Variable *node)
{
    if (node == NULL)
    {
        return;
    }

    for (int i = 0; i < node->n_children; i++)
    {
        print_vars(node->children[i]);
    }

    print_variable(node);
    for (int idx = 0; idx < node->n_children; idx++)
    {
        printf("%.3f, ", node->local_grads[idx]);
    }
    printf("\n----------------\n");

    return;
}

int main()
{
    printf("yo\n");
    printf("let's go\n");



    Variable x0; 
    Variable x1; 
    Variable x2; 

    init_var(&x0, 0.5, false);
    init_var(&x1, 0.3, false);
    init_var(&x2, 0.1, false);

    Variable *add_res = add(&x0, &x1);
    Variable *add_res_1 = add(&x1, &x2);
    Variable *mul_res = mul(add_res, add_res_1);
    Variable *final_res = sigmoid(mul_res);
    Variable *final_res_1 = relu(mul_res);

    mul_res->can_grad = true;
    add_res->can_grad = true;
    add_res_1->can_grad = true;

    Variable *expected = (Variable*)malloc(sizeof(Variable));

    Variable *loss = sub(final_res, expected);

    Variable **independent_vars = (Variable **)malloc(2 * sizeof(Variable *));
    independent_vars[0] = final_res;
    independent_vars[1] = final_res_1;

    printf("final_res: %lf\n", final_res->val);
    printf("final_res_1: %lf\n", final_res_1->val);

    // dfs all children
    printf("\n### final_res: ###\n");
    print_vars(final_res);
    printf("\n### final_res_1: ###\n");
    print_vars(final_res_1);

    VariablesGradAllocator *grad_alloc =
        (VariablesGradAllocator *)malloc(sizeof(VariablesGradAllocator));
    init_grad_alloc(grad_alloc);
    get_gradients(grad_alloc, independent_vars, 2);

    printf("### grad_alloc: ###\n");
    for (int idx = 0; idx < grad_alloc->n_indep_vars; idx++)
    {
        printf("hashmap size: %zu\n", hashmap_count(grad_alloc->dep_grad_map));

        Variable *indep_var = grad_alloc->independent_vars[idx];

        size_t iter = 0;
        Variable *item;
        int idx = 0;

        while (hashmap_iter(grad_alloc->dep_grad_map, &iter, &item))
        {
            printf("%i, %f, %p, %f, %f\n", idx, indep_var->val, grad_alloc->dependent_vars[idx], grad_alloc->dependent_vars[idx]->val, item->val);
            idx++;
        }
        printf("-----------------\n");
    }

    free_from_variable(final_res);
    return 0;
}