#include "NN.h"
#include "unity/unity.h"

void setUp(void)
{
    // Set up initial state before each test
}

void tearDown(void)
{
    // Clean up after each test
}

// test
void test_VariableInitialization(void)
{
    Variable x;
    init_var(&x, 3.0, true);
    TEST_ASSERT_EQUAL_DOUBLE(3.0, x.val);
    TEST_ASSERT_NULL(x.children);
    TEST_ASSERT_NULL(x.local_grads);
    TEST_ASSERT_EQUAL_INT(0, x.n_children);
}

void test_Addition(void)
{
    Variable x, y;
    init_var(&x, 3.0, true);
    init_var(&y, 2.0, true);

    Variable *add_res = add(&x, &y);
    TEST_ASSERT_NOT_NULL(add_res);
    TEST_ASSERT_EQUAL_DOUBLE(5.0, add_res->val);
    TEST_ASSERT_EQUAL_MEMORY(&x, add_res->children[0], sizeof(Variable));
    TEST_ASSERT_EQUAL_MEMORY(&y, add_res->children[1], sizeof(Variable));

    free_from_variable(add_res);
}

void test_Subtraction(void)
{
    Variable x, y;
    init_var(&x, 3.0, true);
    init_var(&y, 2.0, true);

    Variable *sub_res = sub(&x, &y);
    TEST_ASSERT_NOT_NULL(sub_res);
    TEST_ASSERT_EQUAL_DOUBLE(1.0, sub_res->val);
    TEST_ASSERT_EQUAL_MEMORY(&x, sub_res->children[0], sizeof(Variable));
    TEST_ASSERT_EQUAL_MEMORY(&y, sub_res->children[1], sizeof(Variable));

    free_from_variable(sub_res);
}

void test_Multiplication(void)
{
    Variable x, y;
    init_var(&x, 3.0, true);
    init_var(&y, 2.0, true);

    Variable *mul_res = mul(&x, &y);
    TEST_ASSERT_NOT_NULL(mul_res);
    TEST_ASSERT_EQUAL_DOUBLE(6.0, mul_res->val);
    TEST_ASSERT_EQUAL_MEMORY(&x, mul_res->children[0], sizeof(Variable));
    TEST_ASSERT_EQUAL_MEMORY(&y, mul_res->children[1], sizeof(Variable));

    free_from_variable(mul_res);
}

void test_Sigmoid(void)
{
    Variable x;
    init_var(&x, 2.0, true);

    Variable *sigmoid_res = sigmoid(&x);
    TEST_ASSERT_NOT_NULL(sigmoid_res);
    TEST_ASSERT_EQUAL_DOUBLE(0.8807970779778823, sigmoid_res->val);
    TEST_ASSERT_EQUAL_MEMORY(&x, sigmoid_res->children[0], sizeof(Variable));

    free_from_variable(sigmoid_res);
}

void test_ReLU(void)
{
    Variable x;
    init_var(&x, -2.0, true);

    Variable *relu_res = relu(&x);
    TEST_ASSERT_NOT_NULL(relu_res);
    TEST_ASSERT_EQUAL_DOUBLE(0.0, relu_res->val);
    TEST_ASSERT_EQUAL_MEMORY(&x, relu_res->children[0], sizeof(Variable));

    free_from_variable(relu_res);
}

void test_Power(void)
{
    Variable x;
    init_var(&x, 3.0, true);

    Variable *pow_res = power(&x, 2);
    TEST_ASSERT_NOT_NULL(pow_res);
    TEST_ASSERT_EQUAL_DOUBLE(9.0, pow_res->val);
    TEST_ASSERT_EQUAL_MEMORY(&x, pow_res->children[0], sizeof(Variable));

    free_from_variable(pow_res);
}

// basically tests compute_grads, since get_gradients just wraps it...
// TODO: Make this test better in the future...
void test_get_gradients(void)
{
    VariablesGradAllocator grad_alloc;
    init_grad_alloc(&grad_alloc);

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

    // the gradients for these will be tracked
    mul_res->can_grad = true;
    add_res->can_grad = true;
    add_res_1->can_grad = true;

    Variable **independent_vars = (Variable **)malloc(2 * sizeof(Variable *));
    independent_vars[0] = final_res;
    independent_vars[1] = final_res_1;

    int n_vars = 2;

    // Test case 1: testing for when grad alloc and/or independent_vars is NULL
    get_gradients(NULL, independent_vars, n_vars);
    get_gradients(&grad_alloc, NULL, n_vars);
    get_gradients(NULL, NULL, n_vars);


    // Test case 2: checking for expected behaviour
    get_gradients(&grad_alloc, independent_vars, n_vars);

    // check sizes of arrays
    TEST_ASSERT_EQUAL_size_t(n_vars, grad_alloc.n_indep_vars); // 2
    TEST_ASSERT_EQUAL_size_t(3, grad_alloc.n_dep_vars); // 3

    for (int idx = 0; idx < grad_alloc.n_indep_vars; idx++)
    {
        printf("hashmap size: %zu\n", hashmap_count(grad_alloc.dep_grad_map));

        Variable *indep_var = grad_alloc.independent_vars[idx];

        size_t iter = 0;
        Variable *item;
        int idx = 0;

        while (hashmap_iter(grad_alloc.dep_grad_map, &iter, &item))
        {
            printf("%i, %f, %p, %f, %f\n", idx, indep_var->val, grad_alloc.dependent_vars[idx], grad_alloc.dependent_vars[idx]->val, item->val);
            idx++;
        }
        printf("-----------------\n");
    }

}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_VariableInitialization);
    RUN_TEST(test_Addition);
    RUN_TEST(test_Subtraction);
    RUN_TEST(test_Multiplication);
    RUN_TEST(test_Sigmoid);
    RUN_TEST(test_ReLU);
    RUN_TEST(test_Power);
    RUN_TEST(test_get_gradients);
    UNITY_END();

    return 0;
}
