#include "NN.h"
#include "unity/unity.h"

void setUp(void) {
  // Set up initial state before each test
}

void tearDown(void) {
  // Clean up after each test
}

// test
void test_VariableInitialization(void) {
  Variable x;
  init_var(&x, 3.0);
  TEST_ASSERT_EQUAL_DOUBLE(3.0, x.val);
  TEST_ASSERT_NULL(x.children);
  TEST_ASSERT_NULL(x.local_grads);
  TEST_ASSERT_EQUAL_INT(0, x.n_children);
}

void test_Addition(void) {
  Variable x, y;
  init_var(&x, 3.0);
  init_var(&y, 2.0);

  Variable *add_res = add(&x, &y);
  TEST_ASSERT_NOT_NULL(add_res);
  TEST_ASSERT_EQUAL_DOUBLE(5.0, add_res->val);
  TEST_ASSERT_EQUAL_MEMORY(&x, add_res->children[0], sizeof(Variable));
  TEST_ASSERT_EQUAL_MEMORY(&y, add_res->children[1], sizeof(Variable));

  printf("before free\n");
  free_from_variable(add_res);
  printf("after free\n");
}

void test_Subtraction(void) {
  Variable x, y;
  init_var(&x, 3.0);
  init_var(&y, 2.0);

  Variable *sub_res = sub(&x, &y);
  TEST_ASSERT_NOT_NULL(sub_res);
  TEST_ASSERT_EQUAL_DOUBLE(1.0, sub_res->val);
  TEST_ASSERT_EQUAL_MEMORY(&x, sub_res->children[0], sizeof(Variable));
  TEST_ASSERT_EQUAL_MEMORY(&y, sub_res->children[1], sizeof(Variable));

  free_from_variable(sub_res);
}

void test_Multiplication(void) {
  Variable x, y;
  init_var(&x, 3.0);
  init_var(&y, 2.0);

  Variable *mul_res = mul(&x, &y);
  TEST_ASSERT_NOT_NULL(mul_res);
  TEST_ASSERT_EQUAL_DOUBLE(6.0, mul_res->val);
  TEST_ASSERT_EQUAL_MEMORY(&x, mul_res->children[0], sizeof(Variable));
  TEST_ASSERT_EQUAL_MEMORY(&y, mul_res->children[1], sizeof(Variable));

  free_from_variable(mul_res);
}

void test_Sigmoid(void) {
  Variable x;
  init_var(&x, 2.0);

  Variable *sigmoid_res = sigmoid(&x);
  TEST_ASSERT_NOT_NULL(sigmoid_res);
  TEST_ASSERT_EQUAL_DOUBLE(0.8807970779778823, sigmoid_res->val);
  TEST_ASSERT_EQUAL_MEMORY(&x, sigmoid_res->children[0], sizeof(Variable));

  free_from_variable(sigmoid_res);
}

void test_ReLU(void) {
  Variable x;
  init_var(&x, -2.0);

  Variable *relu_res = relu(&x);
  TEST_ASSERT_NOT_NULL(relu_res);
  TEST_ASSERT_EQUAL_DOUBLE(0.0, relu_res->val);
  TEST_ASSERT_EQUAL_MEMORY(&x, relu_res->children[0], sizeof(Variable));

  free_from_variable(relu_res);
}

int main(void) {
  UNITY_BEGIN();
  RUN_TEST(test_VariableInitialization);
  RUN_TEST(test_Addition);
  RUN_TEST(test_Subtraction);
  RUN_TEST(test_Multiplication);
  RUN_TEST(test_Sigmoid);
  RUN_TEST(test_ReLU);
  UNITY_END();

  return 0;
}
