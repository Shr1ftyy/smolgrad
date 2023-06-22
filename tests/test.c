#include "NN.h"
#include <assert.h>
#include <stdio.h>

int main() {
  printf("yo\n");
  printf("let's go\n");

  Variable **children = malloc(sizeof(Variable *));
  double *grads = malloc(sizeof(double));

  Variable x = {3, children, grads};
  Variable y = {2, children, grads};
  Variable *add_res = add(&x, &y);
  Variable *sub_res = sub(&x, &y);
  Variable *mul_res = mul(&x, &y);

  printf("%f + ", x.val);
  printf("%f = ", y.val);
  printf("%f\n", add_res->val);

  printf("\ngrads:\n");
  for (int i = 0; i < add_res->n_children; i++) {
    printf("%f\n", add_res->local_grads[i]);
  }

  printf("\nchildren values:\n");
  for (int i = 0; i < add_res->n_children; i++) {
    printf("%f\n", add_res->children[i]->val);
  }

  printf("%f - ", x.val);
  printf("%f = ", y.val);
  printf("%f\n", sub_res->val);

  printf("\ngrads:\n");
  for (int i = 0; i < sub_res->n_children; i++) {
    printf("%f\n", sub_res->local_grads[i]);
  }

  printf("\nchildren values:\n");
  for (int i = 0; i < sub_res->n_children; i++) {
    printf("%f\n", sub_res->children[i]->val);
  }

  printf("%f * ", x.val);
  printf("%f = ", y.val);
  printf("%f\n", mul_res->val);

  printf("\ngrads:\n");
  for (int i = 0; i < mul_res->n_children; i++) {
    printf("%f\n", mul_res->local_grads[i]);
  }

  printf("\nchildren values:\n");
  for (int i = 0; i < mul_res->n_children; i++) {
    printf("%f\n", add_res->children[i]->val);
  }

  return 0;
}

/*
  a = 4
  b = 3
  c = a + b  # = 4 + 3 = 7
  d = a * c  # = 4 * 7 = 28
*/