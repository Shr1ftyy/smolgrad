#include "NN.h"
#include <stdio.h>

void print_vars(Variable* node){
  if(node == NULL){
    return;
  }

  for(int i=0; i<node->n_children; i++){
    print_vars(node->children[i]);
  }

  printf("%lf, %p, %p, %d\n", node->val, node->local_grads, node->children, node->n_children);

  return;
}


int main() {
  printf("yo\n");
  printf("let's go\n");

  Variable** children = NULL;
  double* grads = NULL;

  Variable x0 = {0.5, children, grads, 0};
  Variable x1 = {0.7, children, grads, 0};

  Variable* add_res = add(&x0, &x1);
  Variable* mul_res = mul(&x0, add_res);
  Variable* final_res = sigmoid(mul_res);
  Variable* final_res_1 = relu(mul_res);

  Variable* loss = sub(final_res, final_res_1);

  Variable** independent_vars = (Variable**)malloc(2*sizeof(Variable*));
  independent_vars[0] = final_res;
  independent_vars[1] = final_res_1;

  printf("final_res: %lf\n", final_res->val);
  printf("final_res_1: %lf\n", final_res_1->val);

  // dfs all children 
  printf("\n### vals: ###\n");
  print_vars(final_res);


  VariablesGradAllocator** grad_buf = (VariablesGradAllocator**)malloc(2*sizeof(VariablesGradAllocator*));
  init_grad_bufs(grad_buf, 2);
  get_gradients(grad_buf, independent_vars, 2);

  printf("### grad_buf: ###\n");
  for(int i=0; i<2; i++){
    VariablesGradAllocator* grad = grad_buf[i];
    for(int j=0; j<grad->size; j++){
      printf("%f, %f, %f, %zu\n", grad->independent_var->val, grad->dependent_vars[j]->val, grad->gradients[j]->val, grad->size);
    }
    printf("-----------------\n");
  }

  // free_from_variable(final_res);
  return 0;
}