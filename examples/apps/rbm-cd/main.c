/*
 * rbm-cd.c — RBM treinada com divergência contrastiva CD-k (Marco 5).
 * Build: make run_apps APP=rbm-cd
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "neurochart.h"

int main(void)
{
    printf("=== libnn Marco 5: RBM com CD-k ===\n");

    NC_Arena model = nc_arena_init(8u << 20);
    NC_Arena scratch = nc_arena_init(8u << 20);
    nc_rng_seed(7);

    NC_Graph *rbm = nc_zoo_create_rbm(&model, 6, 3);
    if (rbm == NULL) return EXIT_FAILURE;

    const size_t shape[2] = { 4, 6 };
    NC_Tensor *data = nc_tensor_create(&model, shape, 2, false);
    const float v[] = {
        1,1,0,0,0,0,  0,0,1,1,0,0,  0,0,0,0,1,1,  1,0,1,0,1,0
    };
    for (size_t i = 0; i < 24; ++i) data->data[i] = v[i];

    const size_t steps = 200;
    for (size_t s = 0; s < steps; ++s) {
        nc_train_contrastive_divergence(rbm, data, 1, 0.05f);
        if (s % 50 == 0 || s + 1 == steps) {
            float norm = 0.0f;
            NC_Tensor *w = rbm->cells[1].weights;
            for (size_t i = 0; i < w->size; ++i) norm += w->data[i] * w->data[i];
            printf("  passo %3zu — ||W|| = %.4f\n", s + 1, sqrtf(norm));
        }
    }

    nc_graph_destroy(rbm);
    nc_arena_free(&scratch);
    nc_arena_free(&model);
    printf("=== Concluído sem vazamentos. ===\n");
    return EXIT_SUCCESS;
}
