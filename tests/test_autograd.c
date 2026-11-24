#include "test_common.h"
#include "neurochart.h"

#include <string.h>

static NC_Tensor *matrix(NC_Arena *arena, size_t r, size_t c,
                         bool requires_grad, const float *values)
{
    const size_t shape[2] = { r, c };
    NC_Tensor *t = nc_tensor_create(arena, shape, 2, requires_grad);
    if (t != NULL && values != NULL) {
        memcpy(t->data, values, t->size * sizeof(float));
    }
    return t;
}

static NC_Tensor *chain_forward(NC_Tape *tape, NC_Arena *scratch,
                                NC_Tensor *w, NC_Tensor *b,
                                const NC_Tensor *x, NC_Tensor **hidden)
{
    nc_autograd_tape_reset(tape);
    NC_Tensor *z = nc_nn_matmul(tape, scratch, x, w);
    z = nc_nn_add(tape, scratch, z, b);
    *hidden = nc_nn_activation(tape, scratch, z, NC_OP_RELU);
    return *hidden;
}

static float sum_values(const NC_Tensor *t)
{
    float sum = 0.0f;
    for (size_t i = 0; i < t->size; ++i) {
        sum += t->data[i];
    }
    return sum;
}

static void test_tape_basics(void)
{
    NC_Arena model = nc_arena_init(64 * 1024);
    NC_Arena scratch = nc_arena_init(256 * 1024);
    NC_Tape *tape = nc_autograd_tape_init(&model, 8);
    NC_CHECK(tape != NULL);
    NC_CHECK(tape->capacity == 8 && tape->count == 0);

    const float av[] = { 1, 2, 3, 4 };
    const float bv[] = { 5, 6, 7, 8 };
    NC_Tensor *a = matrix(&model, 2, 2, true, av);
    NC_Tensor *b = matrix(&model, 2, 2, true, bv);
    NC_Tensor *c = nc_nn_matmul(tape, &scratch, a, b);
    NC_CHECK(c != NULL);
    NC_CHECK(tape->count == 1);
    NC_CHECK_NEAR(c->data[0], 19.0f, 1e-6f);
    NC_CHECK_NEAR(c->data[3], 50.0f, 1e-6f);

    for (size_t i = 0; i < c->size; ++i) c->grad[i] = 1.0f;
    nc_autograd_backward(tape);
    /* d(sum(A@B))/dA = soma das linhas de B */
    NC_CHECK_NEAR(a->grad[0], 11.0f, 1e-5f);
    NC_CHECK_NEAR(a->grad[3], 15.0f, 1e-5f);
    /* d/dB = soma das colunas de A */
    NC_CHECK_NEAR(b->grad[0], 4.0f, 1e-5f);
    NC_CHECK_NEAR(b->grad[3], 6.0f, 1e-5f);

    nc_autograd_tape_reset(tape);
    NC_CHECK(tape->count == 0);

    nc_arena_free(&scratch);
    nc_arena_free(&model);
}

static void test_numerical_gradient(void)
{
    NC_Arena model = nc_arena_init(128 * 1024);
    NC_Arena scratch = nc_arena_init(512 * 1024);
    NC_Tape *tape = nc_autograd_tape_init(&model, 16);
    NC_CHECK(tape != NULL);

    const float wv[] = {
         0.1f, -0.2f,  0.3f,  0.4f,
        -0.5f,  0.6f, -0.7f,  0.8f,
         0.9f,  1.0f, -1.1f,  1.2f
    };
    const float bv[] = { 0.05f, -0.05f, 0.1f, -0.1f };
    const float xv[] = { 1.0f, -0.5f, 0.25f };
    NC_Tensor *w = matrix(&model, 3, 4, true, wv);
    NC_Tensor *b = matrix(&model, 1, 4, true, bv);
    NC_Tensor *x = matrix(&model, 1, 3, false, xv);
    NC_Tensor *hidden = NULL;

    nc_arena_reset(&scratch);
    chain_forward(tape, &scratch, w, b, x, &hidden);
    NC_CHECK(hidden != NULL);
    for (size_t i = 0; i < hidden->size; ++i) hidden->grad[i] = 1.0f;
    nc_autograd_backward(tape);

    float analytic_w[12], analytic_b[4];
    memcpy(analytic_w, w->grad, sizeof(analytic_w));
    memcpy(analytic_b, b->grad, sizeof(analytic_b));

    const float step = 1e-3f;
    for (size_t i = 0; i < w->size; ++i) {
        const float original = w->data[i];
        w->data[i] = original + step;
        nc_arena_reset(&scratch);
        const float f_plus = sum_values(
            chain_forward(tape, &scratch, w, b, x, &hidden));
        w->data[i] = original - step;
        nc_arena_reset(&scratch);
        const float f_minus = sum_values(
            chain_forward(tape, &scratch, w, b, x, &hidden));
        w->data[i] = original;
        NC_CHECK_NEAR(analytic_w[i], (f_plus - f_minus) / (2.0f * step), 2e-2f);
    }
    for (size_t i = 0; i < b->size; ++i) {
        const float original = b->data[i];
        b->data[i] = original + step;
        nc_arena_reset(&scratch);
        const float f_plus = sum_values(
            chain_forward(tape, &scratch, w, b, x, &hidden));
        b->data[i] = original - step;
        nc_arena_reset(&scratch);
        const float f_minus = sum_values(
            chain_forward(tape, &scratch, w, b, x, &hidden));
        b->data[i] = original;
        NC_CHECK_NEAR(analytic_b[i], (f_plus - f_minus) / (2.0f * step), 2e-2f);
    }

    nc_arena_free(&scratch);
    nc_arena_free(&model);
}

static void test_activations_dropout_and_mse(void)
{
    NC_Arena model = nc_arena_init(128 * 1024);
    NC_Arena scratch = nc_arena_init(256 * 1024);
    NC_Tape *tape = nc_autograd_tape_init(&model, 16);

    const float xv[] = { -1.0f, 0.5f, 2.0f, -0.25f };
    NC_Tensor *x = matrix(&model, 1, 4, true, xv);

    NC_Tensor *relu = nc_nn_activation(tape, &scratch, x, NC_OP_RELU);
    NC_CHECK_NEAR(relu->data[0], 0.0f, 1e-7f);
    NC_CHECK_NEAR(relu->data[2], 2.0f, 1e-7f);
    relu->grad[1] = 1.0f;
    nc_autograd_backward(tape);
    NC_CHECK_NEAR(x->grad[0], 0.0f, 1e-7f);
    NC_CHECK_NEAR(x->grad[1], 1.0f, 1e-7f);

    nc_arena_reset(&scratch);
    nc_autograd_tape_reset(tape);
    const float tv[] = { 0.0f, 2.0f, -1.0f, 4.0f };
    NC_Tensor *pred = matrix(&model, 1, 4, true, tv);
    NC_Tensor *target = matrix(&model, 1, 4, false, xv);
    NC_CHECK_NEAR(nc_compute_mse_loss(pred, target), 30.3125f / 4.0f, 1e-6f);
    nc_autograd_record_mse(tape, pred, target);
    nc_autograd_backward(tape);
    NC_CHECK_NEAR(pred->grad[1], 2.0f * (2.0f - 0.5f) / 4.0f, 1e-6f);

    nc_arena_reset(&scratch);
    nc_autograd_tape_reset(tape);
    nc_rng_seed(11);
    NC_Tensor *dropped = nc_nn_dropout(tape, &scratch, x, 0.5f);
    NC_CHECK(dropped != NULL);
    for (size_t i = 0; i < dropped->size; ++i) {
        NC_CHECK(dropped->data[i] == 0.0f || dropped->data[i] == 2.0f * x->data[i]);
    }
    for (size_t i = 0; i < dropped->size; ++i) dropped->grad[i] = 1.0f;
    memset(x->grad, 0, x->size * sizeof(float));
    nc_autograd_backward(tape);
    /* Com inverted dropout o gradiente é 2 (mantido) ou 0 (descartado). */
    for (size_t i = 0; i < x->size; ++i) {
        NC_CHECK(x->grad[i] == 0.0f || fabsf(x->grad[i] - 2.0f) < 1e-6f);
    }

    nc_arena_free(&scratch);
    nc_arena_free(&model);
}

static void test_review_regressions(void)
{
    NC_Arena model = nc_arena_init(512 * 1024);
    NC_Arena scratch = nc_arena_init(512 * 1024);
    NC_Tape *tape = nc_autograd_tape_init(&model, 16);
    NC_CHECK(tape != NULL);

    /* 1: shapes de matmul incompatíveis devem ser rejeitados sem gravar nó. */
    const float av[] = { 1, 2, 3, 4, 5, 6 };
    const float bv[] = { 1, 2, 3, 4 };
    NC_Tensor *a = matrix(&model, 2, 3, false, av);
    NC_Tensor *b = matrix(&model, 2, 2, false, bv);
    NC_CHECK(nc_nn_matmul(tape, &scratch, a, b) == NULL);
    NC_CHECK(tape->count == 0);

    /* 2: add só permite linhas iguais ou broadcasting de um dos lados. */
    const float a2v[] = { 1, 1, 1, 1 };
    const float b2v[] = { 2, 2, 2, 2, 2, 2 };
    NC_Tensor *a2 = matrix(&model, 2, 2, false, a2v);
    NC_Tensor *b2 = matrix(&model, 3, 2, false, b2v);
    NC_CHECK(nc_nn_add(tape, &scratch, a2, b2) == NULL);
    NC_CHECK(tape->count == 0);

    const float bias_values[] = { 2, 3 };
    NC_Tensor *bias = matrix(&model, 1, 2, true, bias_values);
    NC_Tensor *sum = nc_nn_add(tape, &scratch, a2, bias);
    NC_CHECK(sum != NULL);
    for (size_t i = 0; i < sum->size; ++i) sum->grad[i] = 1.0f;
    nc_autograd_backward(tape);
    NC_CHECK_NEAR(bias->grad[0], 2.0f, 1e-6f); /* soma sobre as 2 linhas */
    NC_CHECK_NEAR(bias->grad[1], 2.0f, 1e-6f);

    nc_autograd_tape_reset(tape);
    NC_Tensor *single = matrix(&model, 1, 2, true, bias_values);
    NC_Tensor *sum2 = nc_nn_add(tape, &scratch, single, b2);
    NC_CHECK(sum2 != NULL);
    for (size_t i = 0; i < sum2->size; ++i) sum2->grad[i] = 1.0f;
    nc_autograd_backward(tape);
    NC_CHECK_NEAR(single->grad[0], 3.0f, 1e-6f);

    /* 3: gravar MSE não pode consumir memória da arena persistente. */
    const float pv[] = { 1, 2, 3, 4 };
    NC_Tensor *pred = matrix(&scratch, 1, 4, true, pv);
    NC_Tensor *target = matrix(&model, 1, 4, false, pv);
    const size_t before = nc_arena_bytes_used(&model);
    for (size_t i = 0; i < 1000; ++i) {
        nc_autograd_tape_reset(tape);
        nc_autograd_record_mse(tape, pred, target);
    }
    NC_CHECK(nc_arena_bytes_used(&model) == before);

    /* target menor que pred: usa min(size) e escala por |pred|. */
    nc_autograd_tape_reset(tape);
    const float zero2[] = { 0, 0 };
    NC_Tensor *small_target = matrix(&model, 1, 2, false, zero2);
    memset(pred->grad, 0, pred->size * sizeof(float));
    nc_autograd_record_mse(tape, pred, small_target);
    nc_autograd_backward(tape);
    NC_CHECK_NEAR(pred->grad[0], 0.5f, 1e-6f); /* 2/|pred| * (1-0) */
    NC_CHECK_NEAR(pred->grad[1], 1.0f, 1e-6f); /* 2/|pred| * (2-0) */
    NC_CHECK(pred->grad[2] == 0.0f && pred->grad[3] == 0.0f);

    /* 4: nenhuma operação aceita fita nula. */
    NC_CHECK(nc_nn_matmul(NULL, &scratch, a, a) == NULL);
    NC_CHECK(nc_nn_add(NULL, &scratch, a2, a2) == NULL);
    NC_CHECK(nc_nn_activation(NULL, &scratch, a2, NC_OP_RELU) == NULL);
    NC_CHECK(nc_nn_dropout(NULL, &scratch, a2, 0.5f) == NULL);

    nc_arena_free(&scratch);
    nc_arena_free(&model);
}

int main(void)
{
    printf("=== test_autograd (Marco 2) ===\n");
    test_tape_basics();
    test_numerical_gradient();
    test_activations_dropout_and_mse();
    test_review_regressions();
    NC_TEST_SUMMARY("test_autograd");
}
