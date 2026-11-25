/*
 * kohonen-map.c — mapa de Kohonen com BMU (Marco 5).
 * Build: make run_apps APP=kohonen-map
 */
#include <stdio.h>
#include <stdlib.h>

#include "neurochart.h"

int main(void)
{
    printf("=== libnn Marco 5: Kohonen / SOM ===\n");

    NC_Arena model = nc_arena_init(4u << 20);
    NC_Arena scratch = nc_arena_init(4u << 20);
    nc_rng_seed(42);

    NC_Graph *som = nc_zoo_create_kohonen(&model, 2, 6);
    if (som == NULL) return EXIT_FAILURE;
    NC_Tensor *protos = som->cells[1].weights;

    const float samples[] = { 0, 0, 0, 1, 1, 0, 1, 1 };
    const size_t shape[2] = { 1, 2 };
    for (size_t epoch = 0; epoch < 200; ++epoch) {
        for (size_t s = 0; s < 4; ++s) {
            NC_Tensor *sample = nc_tensor_create(&scratch, shape, 2, false);
            sample->data[0] = samples[s * 2];
            sample->data[1] = samples[s * 2 + 1];
            nc_train_kohonen_bmu_step(som, sample, 0.3f, 2.0f);
        }
    }

    printf("  protótipos após o treino:\n");
    for (size_t i = 0; i < protos->shape[0]; ++i) {
        printf("    %zu: [%.3f %.3f]\n", i,
               protos->data[i * 2], protos->data[i * 2 + 1]);
    }

    nc_graph_destroy(som);
    nc_arena_free(&scratch);
    nc_arena_free(&model);
    printf("=== Concluído sem vazamentos. ===\n");
    return EXIT_SUCCESS;
}
