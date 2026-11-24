/*
 * ntm_demo.c — leitura/escrita por conteúdo numa NTM (Marco 4).
 * Build: make examples    Execução: ./examples/ntm_demo
 */
#include <stdio.h>
#include <stdlib.h>

#include "neurochart.h"

int main(void)
{
    printf("=== libnn Marco 4: Neural Turing Machine ===\n");

    NC_Arena model = nc_arena_init(32u * 1024 * 1024);
    NC_Arena scratch = nc_arena_init(32u * 1024 * 1024);
    if (model.buffer == NULL || scratch.buffer == NULL) {
        fprintf(stderr, "Falha ao inicializar arenas.\n");
        nc_arena_free(&model);
        nc_arena_free(&scratch);
        return EXIT_FAILURE;
    }
    nc_rng_seed(42);

    NC_Graph *ntm = nc_zoo_create_ntm(&model, 3, 4, 5, 2);
    if (ntm == NULL) {
        fprintf(stderr, "Falha ao criar a NTM.\n");
        nc_arena_free(&scratch);
        nc_arena_free(&model);
        return EXIT_FAILURE;
    }

    const size_t T = 5;
    const size_t xs[3] = { T, 1, 3 };
    NC_Tensor *x = nc_tensor_create(&model, xs, 3, false);
    const float xv[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1 };
    for (size_t i = 0; i < T * 3; ++i) x->data[i] = xv[i];

    nc_sequence_reset_state(ntm);
    NC_Tensor **outs = nc_sequence_forward(ntm, x, T, &scratch);
    if (outs == NULL) {
        fprintf(stderr, "Falha no forward da sequência.\n");
        return EXIT_FAILURE;
    }

    NC_Tensor *read = ntm->cells[1].state.ntm.read_head;
    NC_Tensor *memory = ntm->cells[1].state.ntm.memory_bank;
    printf("  leitura final (%zu dimensões): [", read->size);
    for (size_t j = 0; j < read->size; ++j) printf("%6.3f ", read->data[j]);
    printf("]\n  banco de memória (%zu posições):\n", memory->shape[0]);
    for (size_t i = 0; i < memory->shape[0]; ++i) {
        printf("    [");
        for (size_t j = 0; j < memory->shape[1]; ++j) {
            printf("%6.3f ", memory->data[i * memory->strides[0] + j]);
        }
        printf("]\n");
    }
    printf("  saídas do decodificador: ");
    for (size_t t = 0; t < T; ++t) printf("%.3f ", outs[t]->data[0]);
    printf("\n");

    nc_graph_destroy(ntm);
    nc_arena_free(&scratch);
    nc_arena_free(&model);
    printf("=== Concluído sem vazamentos. ===\n");
    return EXIT_SUCCESS;
}
