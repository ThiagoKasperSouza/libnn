/*
 * lstm_sequence.c — BPTT com LSTM numa tarefa de cópia com atraso (Marco 4).
 * Build: make examples    Execução: ./examples/lstm_sequence
 */
#include <stdio.h>
#include <stdlib.h>

#include "neurochart.h"

static void seed_and_backward(NC_Graph *graph, const NC_Tensor *x, size_t T,
                              NC_Arena *scratch, const float *targets)
{
    nc_arena_reset(scratch);
    nc_sequence_reset_state(graph);
    NC_Tensor **outputs = nc_sequence_forward(graph, x, T, scratch);
    if (outputs == NULL) {
        return;
    }
    for (size_t t = 0; t < T; ++t) {
        const size_t shape[2] = { 1, 1 };
        NC_Tensor *target = nc_tensor_create(scratch, shape, 2, false);
        target->data[0] = targets[t];
        nc_autograd_record_mse(graph->tape, outputs[t], target);
    }
    nc_sequence_backward(graph);
}

int main(void)
{
    printf("=== libnn Marco 4: LSTM com BPTT ===\n");

    NC_Arena model = nc_arena_init(64u * 1024 * 1024);
    NC_Arena scratch = nc_arena_init(64u * 1024 * 1024);
    if (model.buffer == NULL || scratch.buffer == NULL) {
        fprintf(stderr, "Falha ao inicializar arenas.\n");
        nc_arena_free(&model);
        nc_arena_free(&scratch);
        return EXIT_FAILURE;
    }
    nc_rng_seed(2026);

    NC_Graph *net = nc_zoo_create_lstm(&model, 2, 8, 1);
    if (net == NULL) {
        fprintf(stderr, "Falha ao criar a LSTM.\n");
        nc_arena_free(&scratch);
        nc_arena_free(&model);
        return EXIT_FAILURE;
    }

    const size_t T = 6;
    const size_t xs[3] = { T, 1, 2 };
    NC_Tensor *x = nc_tensor_create(&model, xs, 3, false);
    const float xv[] = { 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1 };
    const float tv[] = { 0, 1, 0, 1, 0, 1 }; /* y[t] = x[t-1][0] */
    for (size_t i = 0; i < T * 2; ++i) x->data[i] = xv[i];

    NC_OptimizerAdam opt = { 0.01f, 0.9f, 0.999f, 1e-8f };
    for (size_t epoch = 1; epoch <= 600; ++epoch) {
        seed_and_backward(net, x, T, &scratch, tv);
        nc_train_backprop_step(net, &opt);
        if (epoch % 150 == 0 || epoch == 1) {
            nc_arena_reset(&scratch);
            nc_sequence_reset_state(net);
            NC_Tensor **outs = nc_sequence_forward(net, x, T, &scratch);
            float loss = 0.0f;
            for (size_t t = 0; t < T; ++t) {
                const size_t s[2] = { 1, 1 };
                NC_Tensor *target = nc_tensor_create(&scratch, s, 2, false);
                target->data[0] = tv[t];
                loss += nc_compute_mse_loss(outs[t], target);
            }
            printf("  época %4zu/%d — loss %.6f\n", epoch, 600, loss);
        }
    }

    nc_arena_reset(&scratch);
    nc_sequence_reset_state(net);
    NC_Tensor **outs = nc_sequence_forward(net, x, T, &scratch);
    printf("  previsões: ");
    for (size_t t = 0; t < T; ++t) printf("%.3f ", outs[t]->data[0]);
    printf("\n  alvos:     ");
    for (size_t t = 0; t < T; ++t) printf("%.3f ", tv[t]);
    printf("\n");

    nc_graph_destroy(net);
    nc_arena_free(&scratch);
    nc_arena_free(&model);
    printf("=== Concluído sem vazamentos. ===\n");
    return EXIT_SUCCESS;
}
