#include "NN.h"
#include <stdio.h>

void dfs_vals(Value* node){
  if(node == NULL){
    return;
  }

  for(int i=0; i<node->n_children; i++){
    dfs_vals(node->child_values[i]);
  }

  printf("%lf\n", node->val);

  return;
}

void dfs_grads(Value* node){
  if(node == NULL){
    return;
  }

  for(int i=0; i<node->n_children; i++){
    dfs_grads(node->child_values[i]);
  }

  printf("%lf\n", *(node->local_grads));

  return;
}


int main() {
  printf("yo\n");
  printf("let's go\n");

  Value** children = malloc(sizeof(Value*));
  double* grads = malloc(sizeof(double));
  children[0] = NULL;

  Value x0 = {0.5, children, grads};
  Value x1 = {0.7, children, grads};

  Value* add_res = add(&x0, &x1);
  Value* mul_res = mul(&x0, add_res);
  Value* sigmoid_res = sigmoid(mul_res);
  Value* final_res = relu(sigmoid_res);

  printf("result: %lf\n", final_res->val);

  // dfs all children 
  printf("\n### vals: ###\n");
  dfs_vals(final_res);
  printf("\n### grads: ###\n");
  dfs_grads(final_res);

}

/*
  a = 4
  b = 3
  c = a + b  # = 4 + 3 = 7
  d = a * c  # = 4 * 7 = 28
*/