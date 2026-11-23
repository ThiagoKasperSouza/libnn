/*
 * marco1_basics.c — exemplo do Marco 1 da libnn.
 *
 * Demonstra: arenas model/scratch, criação e preenchimento de tensores,
 * RNG determinístico, matmul (incluindo aliasing), convolução, pooling e
 * reset O(1) da scratch arena.
 *
 * Build: make examples    Execução: ./examples/marco1_basics
 */
#include <stdio.h>
#include <stdlib.h>

#include "neurochart.h"

static void print_matrix(const NC_Tensor *t, const char *label)
{
    printf("%s [%zux%zu]:\n", label, t->shape[0], t->shape[1]);
    for (size_t i = 0; i < t->shape[0]; ++i) {
        printf("  ");
        for (size_t j = 0; j < t->shape[1]; ++j) {
            printf("%9.4f ", t->data[i * t->strides[0] + j * t->strides[1]]);
        }
        printf("\n");
    }
}

int main(void)
{
    printf("=== libnn Marco 1: infraestrutura base ===\n");

    /* Pesos vivem na model arena; ativações temporárias na scratch. */
    NC_Arena model = nc_arena_init(1u << 20);
    NC_Arena scratch = nc_arena_init(1u << 20);
    if (model.buffer == NULL || scratch.buffer == NULL) {
        fprintf(stderr, "Falha ao inicializar as arenas.\n");
        nc_arena_free(&model);
        nc_arena_free(&scratch);
        return EXIT_FAILURE;
    }

    nc_rng_seed(20261123); /* qualquer seed é determinística */

    const size_t w_shape[2] = { 2, 3 };
    const size_t x_shape[2] = { 3, 2 };
    const size_t y_shape[2] = { 2, 2 };
    NC_Tensor *W = nc_tensor_create(&model, w_shape, 2, true);
    NC_Tensor *x = nc_tensor_create(&scratch, x_shape, 2, false);
    NC_Tensor *y = nc_tensor_create(&scratch, y_shape, 2, false);
    if (W == NULL || x == NULL || y == NULL) {
        fprintf(stderr, "Falha ao criar tensores.\n");
        nc_arena_free(&scratch);
        nc_arena_free(&model);
        return EXIT_FAILURE;
    }

    printf("[RNG] W ~ N(0, 0.1), x ~ U(0,1)\n");
    nc_tensor_fill_random_normal(W, 0.0f, 0.1f);
    nc_tensor_fill_random_uniform(x);

    nc_matmul(W, x, y);
    print_matrix(y, "W @ x");

    /* Aliasing: C reaproveita o buffer de A; o resultado ainda é correto. */
    const size_t sq[2] = { 2, 2 };
    NC_Tensor *a = nc_tensor_create(&scratch, sq, 2, false);
    NC_Tensor *b = nc_tensor_create(&scratch, sq, 2, false);
    const float av[] = { 1, 2, 3, 4 };
    const float bv[] = { 5, 6, 7, 8 };
    for (size_t i = 0; i < 4; ++i) {
        a->data[i] = av[i];
        b->data[i] = bv[i];
    }
    nc_matmul(a, b, a);
    print_matrix(a, "a @ b (C == A)");

    /* Convolução 2D e max pooling. */
    const size_t in_shape[4] = { 1, 1, 4, 4 };
    const size_t k_shape[4] = { 1, 1, 2, 2 };
    const size_t conv_shape[4] = { 1, 1, 3, 3 };
    const size_t pool_shape[4] = { 1, 1, 2, 2 };
    NC_Tensor *img = nc_tensor_create(&scratch, in_shape, 4, false);
    NC_Tensor *kernel = nc_tensor_create(&scratch, k_shape, 4, false);
    NC_Tensor *conv = nc_tensor_create(&scratch, conv_shape, 4, false);
    NC_Tensor *pool = nc_tensor_create(&scratch, pool_shape, 4, false);
    for (size_t i = 0; i < 16; ++i) img->data[i] = (float)(i + 1);
    for (size_t i = 0; i < 4; ++i) kernel->data[i] = 1.0f;

    nc_conv2d(img, kernel, conv, 1, 0);
    nc_max_pool2d(conv, pool, 2, 1);
    printf("[Conv] 2x2, stride 1:\n");
    for (size_t i = 0; i < 3; ++i) {
        printf("  ");
        for (size_t j = 0; j < 3; ++j) {
            printf("%6.1f ", conv->data[i * 3 + j]);
        }
        printf("\n");
    }
    printf("[Pool] 2x2, stride 1: %.1f %.1f %.1f %.1f\n",
           pool->data[0], pool->data[1], pool->data[2], pool->data[3]);

    const size_t used = nc_arena_bytes_used(&scratch);
    nc_arena_reset(&scratch); /* O(1): reaproveita toda a memória */
    printf("[Arena] scratch: %zu bytes usados -> %zu após reset\n",
           used, nc_arena_bytes_used(&scratch));
    printf("[Arena] model: %zu bytes usados (pesos persistentes)\n",
           nc_arena_bytes_used(&model));

    nc_arena_free(&scratch);
    nc_arena_free(&model);
    printf("=== Concluído sem vazamentos. ===\n");
    return EXIT_SUCCESS;
}
