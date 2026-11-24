#include "test_common.h"
#include "neurochart.h"

#include <string.h>

static NC_Tensor *make_seq(NC_Arena *arena, size_t T, size_t batch,
                           size_t features, const float *values)
{
    const size_t shape[3] = { T, batch, features };
    NC_Tensor *t = nc_tensor_create(arena, shape, 3, false);
    if (t != NULL && values != NULL) {
        memcpy(t->data, values, t->size * sizeof(float));
    }
    return t;
}

static float sequence_loss(NC_Graph *graph, const NC_Tensor *x, size_t T,
                           NC_Arena *scratch, const float *targets)
{
    nc_arena_reset(scratch);
    nc_sequence_reset_state(graph);
    NC_Tensor **outputs = nc_sequence_forward(graph, x, T, scratch);
    if (outputs == NULL) {
        return NAN;
    }
    float total = 0.0f;
    for (size_t t = 0; t < T; ++t) {
        const size_t s[2] = { 1, 1 };
        NC_Tensor *target = nc_tensor_create(scratch, s, 2, false);
        target->data[0] = targets[t];
        total += nc_compute_mse_loss(outputs[t], target);
    }
    return total;
}

static void seed_gradients(NC_Graph *graph, const NC_Tensor *x, size_t T,
                           NC_Arena *scratch, const float *targets)
{
    nc_arena_reset(scratch);
    nc_sequence_reset_state(graph);
    NC_Tensor **outputs = nc_sequence_forward(graph, x, T, scratch);
    NC_CHECK(outputs != NULL);
    if (outputs == NULL) {
        return;
    }
    for (size_t t = 0; t < T; ++t) {
        const size_t s[2] = { 1, 1 };
        NC_Tensor *target = nc_tensor_create(scratch, s, 2, false);
        target->data[0] = targets[t];
        nc_autograd_record_mse(graph->tape, outputs[t], target);
    }
    nc_sequence_backward(graph);
}

static void test_bptt_gradient(void)
{
    NC_Arena model = nc_arena_init(16 * 1024 * 1024);
    NC_Arena scratch = nc_arena_init(16 * 1024 * 1024);
    nc_rng_seed(11);
    const size_t T = 4;
    NC_Graph *g = nc_zoo_create_rnn(&model, 2, 4, 1);
    NC_CHECK(g != NULL);
    const float xv[] = { 1, 0, 0, 1, 1, 1, 0, 0 };
    const float tv[] = { 0, 1, 0, 1 };
    NC_Tensor *x = make_seq(&model, T, 1, 2, xv);

    seed_gradients(g, x, T, &scratch, tv);
    NC_CHECK(g->tape->count == 32);

    NC_Tensor *w = g->cells[1].weights; /* W_xh */
    float analytic[8];
    memcpy(analytic, w->grad, sizeof(analytic));
    const float step = 1e-3f;
    for (size_t i = 0; i < 8; i += 3) {
        const float original = w->data[i];
        w->data[i] = original + step;
        const float plus = sequence_loss(g, x, T, &scratch, tv);
        w->data[i] = original - step;
        const float minus = sequence_loss(g, x, T, &scratch, tv);
        w->data[i] = original;
        NC_CHECK_NEAR(analytic[i], (plus - minus) / (2.0f * step), 2e-2f);
    }

    /* W_hh também participa do BPTT. */
    NC_Tensor *wh = g->cells[1].recurrent_weights;
    float analytic_h[16];
    memcpy(analytic_h, wh->grad, sizeof(analytic_h));
    for (size_t i = 0; i < 16; i += 5) {
        const float original = wh->data[i];
        wh->data[i] = original + step;
        const float plus = sequence_loss(g, x, T, &scratch, tv);
        wh->data[i] = original - step;
        const float minus = sequence_loss(g, x, T, &scratch, tv);
        wh->data[i] = original;
        NC_CHECK_NEAR(analytic_h[i], (plus - minus) / (2.0f * step), 2e-2f);
    }

    nc_arena_free(&scratch);
    nc_arena_free(&model);
}

static void test_architectures(void)
{
    NC_Arena model = nc_arena_init(32 * 1024 * 1024);
    NC_Arena scratch = nc_arena_init(32 * 1024 * 1024);
    nc_rng_seed(5);
    const size_t T = 4;
    const float xv[] = { 1,0, 0,1, 1,1, 0,0 };
    const float tv[] = { 0,1,0,1 };
    NC_Tensor *x = make_seq(&model, T, 1, 2, xv);

    NC_Graph *nets[4] = {
        nc_zoo_create_rnn(&model, 2, 4, 1),
        nc_zoo_create_lstm(&model, 2, 4, 1),
        nc_zoo_create_gru(&model, 2, 4, 1),
        nc_zoo_create_ntm(&model, 2, 4, 3, 1)
    };
    const char *names[4] = { "RNN", "LSTM", "GRU", "NTM" };

    for (size_t a = 0; a < 4; ++a) {
        NC_Graph *g = nets[a];
        NC_CHECK(g != NULL);
        if (g == NULL) continue;
        nc_arena_reset(&scratch);
        nc_sequence_reset_state(g);
        NC_Tensor **outs = nc_sequence_forward(g, x, T, &scratch);
        NC_CHECK(outs != NULL);
        if (outs == NULL) continue;
        for (size_t t = 0; t < T; ++t) {
            NC_CHECK(outs[t] != NULL);
            NC_CHECK(outs[t]->shape[0] == 1 && outs[t]->shape[1] == 1);
            for (size_t i = 0; i < outs[t]->size; ++i) {
                NC_CHECK(isfinite(outs[t]->data[i]));
            }
        }
        NC_CHECK(g->tape->count > 0);

        seed_gradients(g, x, T, &scratch, tv);
        bool any_gradient = false;
        for (size_t c = 0; c < g->cell_count; ++c) {
            NC_Cell *cell = &g->cells[c];
            NC_Tensor *params[16] = {
                cell->weights, cell->recurrent_weights, cell->bias,
                cell->gate_weights[0], cell->gate_weights[1],
                cell->gate_weights[2], cell->gate_weights[3],
                cell->gate_recurrent[0], cell->gate_recurrent[1],
                cell->gate_recurrent[2], cell->gate_recurrent[3],
                cell->gate_bias[0], cell->gate_bias[1],
                cell->gate_bias[2], cell->gate_bias[3], NULL
            };
            for (size_t p = 0; p < 15; ++p) {
                if (params[p] == NULL || params[p]->grad == NULL) continue;
                for (size_t i = 0; i < params[p]->size; ++i) {
                    if (fabsf(params[p]->grad[i]) > 1e-8f) any_gradient = true;
                }
            }
        }
        NC_CHECK(any_gradient);
        printf("  %s: %zu nós, %zu parâmetros Adam\n",
               names[a], g->tape->count, g->adam_count);
    }

    /* NTM: leitura e escrita por conteúdo produzem estados válidos. */
    NC_Graph *ntm = nets[3];
    nc_arena_reset(&scratch);
    nc_sequence_reset_state(ntm);
    NC_Tensor **ntm_outs = nc_sequence_forward(ntm, x, T, &scratch);
    NC_CHECK(ntm_outs != NULL);
    NC_CHECK(ntm->cells[1].state.ntm.memory_bank != NULL);
    NC_CHECK(ntm->cells[1].state.ntm.read_head != NULL);
    if (ntm->cells[1].state.ntm.read_head != NULL) {
        for (size_t i = 0; i < ntm->cells[1].state.ntm.read_head->size; ++i) {
            NC_CHECK(isfinite(ntm->cells[1].state.ntm.read_head->data[i]));
        }
    }

    nc_arena_free(&scratch);
    nc_arena_free(&model);
}

static void test_sequence_training(void)
{
    NC_Arena model = nc_arena_init(64 * 1024 * 1024);
    NC_Arena scratch = nc_arena_init(64 * 1024 * 1024);
    nc_rng_seed(23);
    const size_t T = 6;
    /* Tarefa de cópia com atraso 1: y[t] = x[t-1][0]. */
    const float xv[] = {
        1, 0,  0, 1,  1, 1,  0, 0,  1, 0,  0, 1
    };
    const float tv[] = { 0, 1, 0, 1, 0, 1 };
    NC_Tensor *x = make_seq(&model, T, 1, 2, xv);
    NC_Graph *g = nc_zoo_create_gru(&model, 2, 8, 1);
    NC_CHECK(g != NULL);
    NC_OptimizerAdam opt = { 0.01f, 0.9f, 0.999f, 1e-8f };

    float first = 0.0f, last = 0.0f;
    for (size_t epoch = 0; epoch < 500; ++epoch) {
        seed_gradients(g, x, T, &scratch, tv);
        last = sequence_loss(g, x, T, &scratch, tv);
        if (epoch == 0) first = last;
        nc_train_backprop_step(g, &opt);
    }
    printf("  GRU (cópia com atraso): loss %.5f -> %.5f\n", first, last);
    NC_CHECK(isfinite(last));
    NC_CHECK(last < first * 0.8f);

    nc_arena_free(&scratch);
    nc_arena_free(&model);
}

int main(void)
{
    printf("=== test_sequence (Marco 4) ===\n");
    test_bptt_gradient();
    test_architectures();
    test_sequence_training();
    NC_TEST_SUMMARY("test_sequence");
}
