#include "test_common.h"
#include "neurochart.h"

#include <string.h>

static float train_step(NC_Graph *graph, NC_OptimizerAdam *opt,
                        const NC_Tensor *x, const NC_Tensor *target,
                        NC_Arena *scratch)
{
    nc_arena_reset(scratch);
    NC_Tensor *pred = nc_graph_forward(graph, x, scratch);
    if (pred == NULL) {
        return NAN;
    }
    const float loss = nc_compute_mse_loss(pred, target);
    nc_autograd_record_mse(graph->tape, pred, target);
    nc_graph_backward(graph);
    nc_train_backprop_step(graph, opt);
    return loss;
}

static NC_Tensor *make_matrix(NC_Arena *arena, size_t r, size_t c,
                              const float *values)
{
    const size_t shape[2] = { r, c };
    NC_Tensor *t = nc_tensor_create(arena, shape, 2, false);
    if (t != NULL && values != NULL) {
        memcpy(t->data, values, t->size * sizeof(float));
    }
    return t;
}

static void test_zoo_shapes(void)
{
    NC_Arena model = nc_arena_init(4 * 1024 * 1024);

    NC_Graph *p = nc_zoo_create_perceptron(&model, 3, 1);
    NC_CHECK(p != NULL);
    NC_CHECK(nc_graph_cell_count(p) == 2);
    NC_CHECK(p->cells[0].type == NC_CELL_INPUT);
    NC_CHECK(p->cells[1].type == NC_CELL_OUTPUT);
    NC_CHECK(p->cells[1].weights->shape[0] == 3);

    NC_Graph *ff = nc_zoo_create_ff(&model, 4, 5, 2);
    NC_CHECK(ff != NULL);
    NC_CHECK(nc_graph_cell_count(ff) == 3);
    NC_CHECK(ff->cells[1].type == NC_CELL_HIDDEN);
    NC_CHECK(ff->cells[2].weights->shape[0] == 5);

    const size_t dims[3] = { 8, 6, 4 };
    NC_Graph *dff = nc_zoo_create_dff(&model, 10, dims, 3, 1);
    NC_CHECK(dff != NULL);
    NC_CHECK(nc_graph_cell_count(dff) == 5);

    NC_Graph *ae = nc_zoo_create_ae(&model, 6, 3);
    NC_CHECK(ae != NULL && nc_graph_cell_count(ae) == 3);
    NC_CHECK(ae->cells[2].type == NC_CELL_MATCH_INPUT_OUTPUT);
    NC_CHECK(ae->cells[2].weights->shape[1] == 6);

    NC_Graph *dae = nc_zoo_create_dae(&model, 6, 3, 0.2f);
    NC_CHECK(dae != NULL);
    NC_CHECK(dae->cells[0].type == NC_CELL_NOISY_INPUT);
    NC_CHECK_NEAR(dae->cells[0].state.noisy.noise_std, 0.2f, 1e-7f);

    NC_Graph *sae = nc_zoo_create_sae(&model, 6, 3, 0.1f, 0.01f);
    NC_CHECK(sae != NULL && sae->sparse_cell == &sae->cells[1]);
    NC_CHECK_NEAR(sae->sparsity_beta, 0.01f, 1e-7f);

    const size_t residual_dims[3] = { 6, 6, 6 };
    NC_Graph *drn = nc_zoo_create_drn(&model, 4, residual_dims, 3, 1);
    NC_CHECK(drn != NULL);
    bool has_residual = false;
    for (size_t i = 0; i < drn->edge_count; ++i) {
        if (drn->edges[i].type == NC_EDGE_RESIDUAL) has_residual = true;
    }
    NC_CHECK(has_residual);

    nc_arena_free(&model);
}

static void test_forward_and_backward(void)
{
    NC_Arena model = nc_arena_init(1024 * 1024);
    NC_Arena scratch = nc_arena_init(1024 * 1024);
    NC_Graph *ff = nc_zoo_create_ff(&model, 2, 4, 1);
    NC_CHECK(ff != NULL);

    const float xv[] = { 0.0f, 1.0f };
    NC_Tensor *x = make_matrix(&model, 1, 2, xv);
    NC_Tensor *pred = nc_graph_forward(ff, x, &scratch);
    NC_CHECK(pred != NULL);
    NC_CHECK(pred->shape[0] == 1 && pred->shape[1] == 1);
    NC_CHECK(ff->tape->count > 0);

    const float tv[] = { 1.0f };
    NC_Tensor *target = make_matrix(&model, 1, 1, tv);
    nc_autograd_record_mse(ff->tape, pred, target);
    nc_graph_backward(ff);
    NC_CHECK(ff->cells[1].weights->grad != NULL);
    NC_CHECK(ff->cells[2].weights->grad != NULL);
    bool nonzero = false;
    for (size_t i = 0; i < ff->cells[2].weights->size; ++i) {
        if (ff->cells[2].weights->grad[i] != 0.0f) nonzero = true;
    }
    NC_CHECK(nonzero);

    NC_OptimizerAdam opt = { 0.01f, 0.9f, 0.999f, 1e-8f };
    float before[8];
    NC_CHECK(ff->cells[2].weights->size <= 8);
    memcpy(before, ff->cells[2].weights->data,
           ff->cells[2].weights->size * sizeof(float));
    nc_train_backprop_step(ff, &opt);
    bool changed = false;
    for (size_t i = 0; i < ff->cells[2].weights->size; ++i) {
        if (ff->cells[2].weights->data[i] != before[i]) changed = true;
    }
    NC_CHECK(changed);
    NC_CHECK(ff->adam_count > 0);

    nc_arena_free(&scratch);
    nc_arena_free(&model);
}

static void test_xor_training(void)
{
    NC_Arena model = nc_arena_init(4 * 1024 * 1024);
    NC_Arena scratch = nc_arena_init(4 * 1024 * 1024);
    NC_Graph *ff = nc_zoo_create_ff(&model, 2, 8, 1);
    NC_CHECK(ff != NULL);

    const float xv[] = { 0, 0, 0, 1, 1, 0, 1, 1 };
    const float tv[] = { 0, 1, 1, 0 };
    NC_Tensor *x = make_matrix(&model, 4, 2, xv);
    NC_Tensor *target = make_matrix(&model, 4, 1, tv);
    NC_OptimizerAdam opt = { 0.05f, 0.9f, 0.999f, 1e-8f };

    float first = 0.0f, last = 0.0f;
    for (size_t epoch = 0; epoch < 3000; ++epoch) {
        last = train_step(ff, &opt, x, target, &scratch);
        if (epoch == 0) first = last;
    }
    printf("  XOR: loss inicial %.5f -> final %.5f\n", first, last);
    NC_CHECK(isfinite(last));
    NC_CHECK(last < first * 0.2f);
    NC_CHECK(last < 0.05f);

    nc_arena_free(&scratch);
    nc_arena_free(&model);
}

static void test_autoencoder_training(void)
{
    NC_Arena model = nc_arena_init(4 * 1024 * 1024);
    NC_Arena scratch = nc_arena_init(4 * 1024 * 1024);
    const float xv[] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    NC_Tensor *x = make_matrix(&model, 4, 4, xv);
    NC_OptimizerAdam opt = { 0.02f, 0.9f, 0.999f, 1e-8f };

    NC_Graph *architectures[4];
    architectures[0] = nc_zoo_create_ae(&model, 4, 2);
    architectures[1] = nc_zoo_create_dae(&model, 4, 2, 0.1f);
    architectures[2] = nc_zoo_create_sae(&model, 4, 2, 0.1f, 0.001f);
    const size_t dims[2] = { 6, 6 };
    architectures[3] = nc_zoo_create_drn(&model, 4, dims, 2, 4);

    for (size_t a = 0; a < 4; ++a) {
        NC_Graph *g = architectures[a];
        NC_CHECK(g != NULL);
        float first = 0.0f, last = 0.0f;
        for (size_t epoch = 0; epoch < 1500; ++epoch) {
            last = train_step(g, &opt, x, x, &scratch);
            if (epoch == 0) first = last;
        }
        printf("  arq %zu (%zu células): loss %.5f -> %.5f\n",
               a, nc_graph_cell_count(g), first, last);
        NC_CHECK(isfinite(last));
        NC_CHECK(last < first * 0.6f);
    }

    nc_arena_free(&scratch);
    nc_arena_free(&model);
}

int main(void)
{
    printf("=== test_zoo (Marco 2) ===\n");
    test_zoo_shapes();
    test_forward_and_backward();
    test_xor_training();
    test_autoencoder_training();
    NC_TEST_SUMMARY("test_zoo");
}
