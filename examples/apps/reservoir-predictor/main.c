/*
 * reservoir-predictor.c — ELM com ridge/pseudoinversa (Marco 5).
 * Build: make run_apps APP=reservoir-predictor
 */
#include <stdio.h>
#include <stdlib.h>

#include "neurochart.h"

int main(void)
{
    printf("=== libnn Marco 5: ELM + ridge ===\n");

    NC_Arena model = nc_arena_init(8u << 20);
    NC_Arena scratch = nc_arena_init(8u << 20);
    nc_rng_seed(9);

    NC_Graph *elm = nc_zoo_create_elm(&model, 2, 12, 1);
    if (elm == NULL) return EXIT_FAILURE;

    const size_t shape[2] = { 4, 2 };
    NC_Tensor *x = nc_tensor_create(&model, shape, 2, false);
    NC_Tensor *y = nc_tensor_create(&model, (size_t[2]){ 4, 1 }, 2, false);
    const float xv[] = { 0,0, 0,1, 1,0, 1,1 };
    const float yv[] = { 0, 1, 1, 2 }; /* y = x0 + x1 */
    for (size_t i = 0; i < 8; ++i) x->data[i] = xv[i];
    for (size_t i = 0; i < 4; ++i) y->data[i] = yv[i];

    nc_arena_reset(&scratch);
    if (nc_graph_forward(elm, x, &scratch) == NULL) return EXIT_FAILURE;
    NC_Tensor *features = elm->cells[1].activity;
    if (features == NULL ||
        !nc_train_reservoir_ridge_regression(elm, features, y, 1e-3f)) {
        fprintf(stderr, "Falha na regressão ridge.\n");
        return EXIT_FAILURE;
    }

    nc_arena_reset(&scratch);
    NC_Tensor *pred = nc_graph_forward(elm, x, &scratch);
    printf("  alvos:     %.2f %.2f %.2f %.2f\n",
           y->data[0], y->data[1], y->data[2], y->data[3]);
    printf("  previstos: %.2f %.2f %.2f %.2f\n",
           pred->data[0], pred->data[1], pred->data[2], pred->data[3]);
    printf("  MSE: %.6f\n", nc_compute_mse_loss(pred, y));

    nc_graph_destroy(elm);
    nc_arena_free(&scratch);
    nc_arena_free(&model);
    printf("=== Concluído sem vazamentos. ===\n");
    return EXIT_SUCCESS;
}
