/*
 * dae_example.c — Denoising Autoencoder com a libnn (Marco 2).
 *
 * Versão do exemplo do plano sem nc_render_svg() (Marco 6).
 * Build: make examples    Execução: ./examples/dae_example
 */
#include <stdio.h>
#include <stdlib.h>

#include "neurochart.h"

int main(void)
{
    printf("=== libnn Marco 2: Denoising Autoencoder ===\n");

    NC_Arena model = nc_arena_init(8u * 1024 * 1024);
    NC_Arena scratch = nc_arena_init(8u * 1024 * 1024);
    if (model.buffer == NULL || scratch.buffer == NULL) {
        fprintf(stderr, "Falha ao inicializar arenas.\n");
        nc_arena_free(&model);
        nc_arena_free(&scratch);
        return EXIT_FAILURE;
    }

    nc_rng_seed(123);
    const size_t input_dim = 8;
    NC_Graph *dae = nc_zoo_create_dae(&model, input_dim, 4, 0.2f);
    if (dae == NULL) {
        fprintf(stderr, "Falha ao criar o grafo DAE.\n");
        nc_arena_free(&scratch);
        nc_arena_free(&model);
        return EXIT_FAILURE;
    }

    /* Quatro padrões esparsos (one-hot) para reconstruir. */
    float values[4][8] = { { 0 } };
    for (size_t i = 0; i < 4; ++i) {
        values[i][i] = 1.0f;
        values[i][4 + i] = 0.5f;
    }
    const size_t x_shape[2] = { 4, input_dim };
    NC_Tensor *x = nc_tensor_create(&model, x_shape, 2, false);
    for (size_t i = 0; i < 32; ++i) {
        x->data[i] = values[i / 8][i % 8];
    }

    NC_OptimizerAdam opt = { 0.02f, 0.9f, 0.999f, 1e-8f };
    const size_t epochs = 2000;
    float first = 0.0f, last = 0.0f;

    for (size_t epoch = 1; epoch <= epochs; ++epoch) {
        nc_arena_reset(&scratch);
        NC_Tensor *recon = nc_graph_forward(dae, x, &scratch);
        if (recon == NULL) {
            fprintf(stderr, "Falha no forward.\n");
            return EXIT_FAILURE;
        }
        last = nc_compute_mse_loss(x, recon);
        if (epoch == 1) first = last;
        nc_autograd_record_mse(dae->tape, recon, x);
        nc_graph_backward(dae);
        nc_train_backprop_step(dae, &opt);

        if (epoch % 500 == 0 || epoch == 1) {
            printf("  época %4zu/%zu — loss %.6f\n", epoch, epochs, last);
        }
    }

    nc_arena_reset(&scratch);
    NC_Tensor *recon = nc_graph_forward(dae, x, &scratch);
    printf("  entrada:  ");
    for (size_t j = 0; j < 8; ++j) printf("%5.2f ", x->data[j]);
    printf("\n  reconstrução: ");
    for (size_t j = 0; j < 8; ++j) printf("%5.2f ", recon->data[j]);
    printf("\n  loss: %.6f -> %.6f\n", first, nc_compute_mse_loss(x, recon));

    nc_graph_destroy(dae);
    nc_arena_free(&scratch);
    nc_arena_free(&model);
    printf("=== Concluído sem vazamentos. ===\n");
    return EXIT_SUCCESS;
}
