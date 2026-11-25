/*
 * hopfield-memory.c — memória associativa com aprendizado Hebbiano (Marco 5).
 * Build: make run_apps APP=hopfield-memory
 */
#include <stdio.h>
#include <stdlib.h>

#include "neurochart.h"

int main(void)
{
    printf("=== libnn Marco 5: Hopfield / Hebb ===\n");

    NC_Arena model = nc_arena_init(4u << 20);
    NC_Arena scratch = nc_arena_init(4u << 20);
    if (model.buffer == NULL || scratch.buffer == NULL) {
        nc_arena_free(&model);
        nc_arena_free(&scratch);
        return EXIT_FAILURE;
    }
    nc_rng_seed(11);

    NC_Graph *net = nc_zoo_create_hopfield(&model, 6);
    if (net == NULL) return EXIT_FAILURE;

    const size_t shape[2] = { 1, 6 };
    NC_Tensor *pattern = nc_tensor_create(&model, shape, 2, false);
    const float values[] = { 1, 1, -1, -1, 1, -1 };
    for (size_t i = 0; i < 6; ++i) pattern->data[i] = values[i];

    /* Fase de armazenamento: várias passadas Hebbianas. */
    for (size_t rep = 0; rep < 20; ++rep) {
        nc_arena_reset(&scratch);
        nc_graph_forward(net, pattern, &scratch);
        nc_train_hebbian_step(net, 0.5f);
    }

    /* Recuperação a partir de uma versão ruidosa. */
    NC_Tensor *noisy = nc_tensor_create(&model, shape, 2, false);
    for (size_t i = 0; i < 6; ++i) {
        noisy->data[i] = pattern->data[i] * 0.7f;
    }
    nc_arena_reset(&scratch);
    NC_Tensor *recalled = nc_graph_forward(net, noisy, &scratch);

    printf("  padrão:  ");
    for (size_t i = 0; i < 6; ++i) printf("%+5.1f ", values[i]);
    printf("\n  recuperado: ");
    size_t correct = 0;
    for (size_t i = 0; i < 6; ++i) {
        printf("%+5.1f ", recalled->data[i]);
        if (recalled->data[i] * values[i] > 0.0f) ++correct;
    }
    printf("\n  acerto: %zu/6 sinais\n", correct);

    nc_graph_destroy(net);
    nc_arena_free(&scratch);
    nc_arena_free(&model);
    printf("=== Concluído sem vazamentos. ===\n");
    return EXIT_SUCCESS;
}
