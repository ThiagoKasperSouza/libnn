/*
 * xor_perceptron.c — classificador XOR com a libnn (Marco 2).
 *
 * XOR não é linearmente separável: um Perceptron (P) puro não resolve.
 * Usamos uma rede Feed Forward (FF) com uma camada oculta ReLU.
 *
 * Build: make examples    Execução: ./examples/xor_perceptron
 */
#include <stdio.h>
#include <stdlib.h>

#include "neurochart.h"

int main(void)
{
    printf("=== libnn Marco 2: XOR com FF ===\n");

    NC_Arena model = nc_arena_init(4u * 1024 * 1024);
    NC_Arena scratch = nc_arena_init(4u * 1024 * 1024);
    if (model.buffer == NULL || scratch.buffer == NULL) {
        fprintf(stderr, "Falha ao inicializar arenas.\n");
        nc_arena_free(&model);
        nc_arena_free(&scratch);
        return EXIT_FAILURE;
    }

    nc_rng_seed(42);
    NC_Graph *net = nc_zoo_create_ff(&model, 2, 8, 1);
    if (net == NULL) {
        fprintf(stderr, "Falha ao criar a rede.\n");
        nc_arena_free(&scratch);
        nc_arena_free(&model);
        return EXIT_FAILURE;
    }

    const size_t x_shape[2] = { 4, 2 };
    const size_t y_shape[2] = { 4, 1 };
    NC_Tensor *x = nc_tensor_create(&model, x_shape, 2, false);
    NC_Tensor *y = nc_tensor_create(&model, y_shape, 2, false);
    const float xv[] = { 0, 0, 0, 1, 1, 0, 1, 1 };
    const float yv[] = { 0, 1, 1, 0 };
    for (size_t i = 0; i < 8; ++i) x->data[i] = xv[i];
    for (size_t i = 0; i < 4; ++i) y->data[i] = yv[i];

    NC_OptimizerAdam opt = { 0.05f, 0.9f, 0.999f, 1e-8f };
    const size_t epochs = 3000;
    float loss = 0.0f;
    for (size_t epoch = 1; epoch <= epochs; ++epoch) {
        nc_arena_reset(&scratch);
        NC_Tensor *pred = nc_graph_forward(net, x, &scratch);
        if (pred == NULL) {
            fprintf(stderr, "Falha no forward.\n");
            return EXIT_FAILURE;
        }
        loss = nc_compute_mse_loss(pred, y);
        nc_autograd_record_mse(net->tape, pred, y);
        nc_graph_backward(net);
        nc_train_backprop_step(net, &opt);
        if (epoch % 500 == 0 || epoch == 1) {
            printf("  época %4zu/%zu — loss %.6f\n", epoch, epochs, loss);
        }
    }
    nc_arena_reset(&scratch);
    NC_Tensor *pred = nc_graph_forward(net, x, &scratch);
    printf("  previsões finais: %.3f %.3f %.3f %.3f\n",
           pred->data[0], pred->data[1], pred->data[2], pred->data[3]);
    printf("  loss final: %.6f\n", nc_compute_mse_loss(pred, y));

    nc_graph_destroy(net);
    nc_arena_free(&scratch);
    nc_arena_free(&model);
    printf("=== Concluído. ===\n");
    return EXIT_SUCCESS;
}
