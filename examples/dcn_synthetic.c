/*
 * dcn_synthetic.c — Deep Convolutional Network com dados sintéticos (Marco 3).
 *
 * O plano cita mnist_dcn.c; como a libnn não tem dependências de I/O de
 * datasets, este exemplo usa padrões 8x8 gerados em memória.
 * Build: make examples    Execução: ./examples/dcn_synthetic
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "neurochart.h"

int main(void)
{
    printf("=== libnn Marco 3: DCN em padrões sintéticos ===\n");

    NC_Arena model = nc_arena_init(16u * 1024 * 1024);
    NC_Arena scratch = nc_arena_init(16u * 1024 * 1024);
    if (model.buffer == NULL || scratch.buffer == NULL) {
        fprintf(stderr, "Falha ao inicializar arenas.\n");
        nc_arena_free(&model);
        nc_arena_free(&scratch);
        return EXIT_FAILURE;
    }
    nc_rng_seed(2026);

    NC_Graph *net = nc_zoo_create_dcn(&model, 1, 8, 8, 2);
    if (net == NULL) {
        fprintf(stderr, "Falha ao criar a DCN.\n");
        nc_arena_free(&scratch);
        nc_arena_free(&model);
        return EXIT_FAILURE;
    }

    /* Classe 0: faixa horizontal; classe 1: faixa vertical. */
    float pixels[4][64] = { { 0 } };
    for (size_t i = 0; i < 64; ++i) {
        pixels[0][i] = (i / 8 >= 3 && i / 8 <= 4) ? 1.0f : 0.0f;
        pixels[1][i] = (i % 8 >= 3 && i % 8 <= 4) ? 1.0f : 0.0f;
        pixels[2][i] = pixels[0][i];
        pixels[3][i] = pixels[1][i];
    }
    const size_t xs[4] = { 4, 1, 8, 8 };
    const size_t ys[2] = { 4, 2 };
    NC_Tensor *x = nc_tensor_create(&model, xs, 4, false);
    NC_Tensor *y = nc_tensor_create(&model, ys, 2, false);
    for (size_t i = 0; i < 4; ++i) {
        memcpy(x->data + i * 64, pixels[i], 64 * sizeof(float));
        y->data[i * 2] = (i % 2 == 0) ? 1.0f : 0.0f;
        y->data[i * 2 + 1] = (i % 2 == 0) ? 0.0f : 1.0f;
    }

    NC_OptimizerAdam opt = { 0.01f, 0.9f, 0.999f, 1e-8f };
    const size_t epochs = 800;
    float first = 0.0f, last = 0.0f;
    for (size_t epoch = 1; epoch <= epochs; ++epoch) {
        nc_arena_reset(&scratch);
        NC_Tensor *pred = nc_graph_forward(net, x, &scratch);
        last = nc_compute_mse_loss(pred, y);
        if (epoch == 1) first = last;
        nc_autograd_record_mse(net->tape, pred, y);
        nc_graph_backward(net);
        nc_train_backprop_step(net, &opt);
        if (epoch % 200 == 0 || epoch == 1) {
            printf("  época %4zu/%zu — loss %.6f\n", epoch, epochs, last);
        }
    }

    nc_arena_reset(&scratch);
    NC_Tensor *pred = nc_graph_forward(net, x, &scratch);
    size_t correct = 0;
    for (size_t i = 0; i < 4; ++i) {
        const size_t guess = pred->data[i * 2] > pred->data[i * 2 + 1] ? 0 : 1;
        if (guess == i % 2) ++correct;
    }
    printf("  acurácia: %zu/4 — loss %.6f -> %.6f\n", correct, first,
           nc_compute_mse_loss(pred, y));

    nc_graph_destroy(net);
    nc_arena_free(&scratch);
    nc_arena_free(&model);
    printf("=== Concluído sem vazamentos. ===\n");
    return EXIT_SUCCESS;
}
