/*
 * gan_toy.c — GAN mínima com a libnn (Marco 3).
 *
 * O gerador aprende a produzir vetores 3D parecidos com três centros de dados.
 * Build: make examples    Execução: ./examples/gan_toy
 */
#include <stdio.h>
#include <stdlib.h>

#include "neurochart.h"

int main(void)
{
    printf("=== libnn Marco 3: GAN toy ===\n");

    NC_Arena model = nc_arena_init(32u * 1024 * 1024);
    NC_Arena scratch = nc_arena_init(32u * 1024 * 1024);
    if (model.buffer == NULL || scratch.buffer == NULL) {
        fprintf(stderr, "Falha ao inicializar arenas.\n");
        nc_arena_free(&model);
        nc_arena_free(&scratch);
        return EXIT_FAILURE;
    }
    nc_rng_seed(7);

    NC_GAN *gan = nc_zoo_create_gan(&model, 4, 3);
    if (gan == NULL) {
        fprintf(stderr, "Falha ao criar a GAN.\n");
        nc_arena_free(&scratch);
        nc_arena_free(&model);
        return EXIT_FAILURE;
    }

    const size_t rs[2] = { 8, 3 };
    NC_Tensor *real = nc_tensor_create(&model, rs, 2, false);
    const float centers[3][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            real->data[i * 3 + j] = centers[i % 3][j] +
                                    (float)nc_rng_next_normal() * 0.02f;
        }
    }

    NC_OptimizerAdam opt = { 0.002f, 0.9f, 0.999f, 1e-8f };
    const size_t steps = 600;
    for (size_t step = 1; step <= steps; ++step) {
        nc_arena_reset(&scratch);
        nc_train_gan_step(gan->generator, gan->discriminator, real, &scratch, &opt);
        if (step % 250 == 0 || step == 1) {
            printf("  passo %4zu/%zu concluído\n", step, steps);
        }
    }

    nc_arena_reset(&scratch);
    const size_t ns[2] = { 3, 4 };
    NC_Tensor *noise = nc_tensor_create(&scratch, ns, 2, false);
    nc_tensor_fill_random_normal(noise, 0.0f, 1.0f);
    NC_Tensor *fake = nc_graph_forward(gan->generator, noise, &scratch);
    printf("  amostras geradas:\n");
    for (size_t i = 0; i < 3; ++i) {
        printf("    [%.3f %.3f %.3f]\n", fake->data[i * 3],
               fake->data[i * 3 + 1], fake->data[i * 3 + 2]);
    }

    nc_graph_destroy(gan->generator);
    nc_graph_destroy(gan->discriminator);
    nc_arena_free(&scratch);
    nc_arena_free(&model);
    printf("=== Concluído sem vazamentos. ===\n");
    return EXIT_SUCCESS;
}
