#ifndef NC_TENSOR_H
#define NC_TENSOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "nc_arena.h"

/* Alinhamento dos buffers de dados (32 bytes = AVX2). */
#ifndef NC_TENSOR_ALIGNMENT
#define NC_TENSOR_ALIGNMENT 32u
#endif

/*
 * Tensor multidimensional contíguo (row-major), com até 4 dimensões.
 *
 * shape[0..ndim-1] contém as dimensões ativas, interpretadas como
 * [Batch, Channels, Height, Width]. Exemplos:
 *   ndim=2 -> shape = [M, N]        (matriz 2D)
 *   ndim=3 -> shape = [C, H, W]     (mapa de canais)
 *   ndim=4 -> shape = [N, C, H, W]  (batch de imagens)
 */
typedef struct NC_Tensor {
    float *data;          /* Dados alinhados (32 bytes por padrão). */
    float *grad;          /* Gradiente acumulado (NULL se não treinável). */
    size_t shape[4];      /* [Batch, Channels, Height, Width]. */
    size_t strides[4];    /* Passos em elementos (row-major contíguo). */
    size_t ndim;          /* Número de dimensões ativas (1 a 4). */
    size_t size;          /* Total de elementos escalares. */
    bool requires_grad;   /* Indica se o tensor é rastreado pelo Autograd. */
} NC_Tensor;

/* Cria o tensor e seus buffers na arena fornecida. NULL em caso de falha. */
NC_Tensor *nc_tensor_create(NC_Arena *arena, const size_t shape[], size_t ndim,
                            bool requires_grad);

/* Preenche data (e grad, se existir) com zeros. */
void nc_tensor_fill_zeros(NC_Tensor *t);

/* Preenche data com amostras de N(mean, stddev^2) via Box-Muller. */
void nc_tensor_fill_random_normal(NC_Tensor *t, float mean, float stddev);

/* Preenche data com amostras uniformes em [0, 1). */
void nc_tensor_fill_random_uniform(NC_Tensor *t);

/* ------------------------------------------------------------------ */
/* RNG determinístico: Xoshiro256** + splitmix64 para semeadura.       */
/* ------------------------------------------------------------------ */

/*
 * Semeadura global: qualquer valor produz uma sequência determinística
 * (inclusive 0). Se não houver chamada a esta função, usa-se uma semente
 * padrão fixa na primeira amostragem.
 */
void nc_rng_seed(uint64_t seed);
uint64_t nc_rng_next_u64(void);
double nc_rng_next_double(void);   /* uniforme em [0, 1) */
double nc_rng_next_normal(void);   /* N(0, 1) via Box-Muller */

/* ------------------------------------------------------------------ */
/* Álgebra linear                                                      */
/* ------------------------------------------------------------------ */

/*
 * C = A x B. As duas últimas dimensões são as dimensões de matriz
 * (M x K) * (K x N) -> (M x N); dimensões iniciais são batches e devem
 * casar entre A, B e C. Suporta ndim = 2, 3 ou 4.
 * Implementado com blocking (tiling) para melhor uso de cache L1/L2.
 * Se C compartilhar memória com A ou B (aliasing total ou parcial), o
 * resultado é calculado em buffer temporário e copiado ao final.
 */
void nc_matmul(const NC_Tensor *A, const NC_Tensor *B, NC_Tensor *C);

/*
 * Convolução 2D direta (sem im2col — otimização prevista para o Marco 3).
 * input  = [N, C_in,  H,  W]
 * kernel = [C_out, C_in, kH, kW]
 * output = [N, C_out, oH, oW], oH = (H + 2*pad - kH)/stride + 1
 */
void nc_conv2d(const NC_Tensor *input, const NC_Tensor *kernel,
               NC_Tensor *output, size_t stride, size_t padding);

/*
 * Convolução transposta 2D (deconvolução).
 * input  = [N, C_in,  H,  W]
 * kernel = [C_in, C_out, kH, kW]
 * output = [N, C_out, (H-1)*stride - 2*pad + kH, ...]
 */
void nc_conv2d_transpose(const NC_Tensor *input, const NC_Tensor *kernel,
                         NC_Tensor *output, size_t stride, size_t padding);

/*
 * Max pooling 2D sem padding.
 * input  = [N, C, H, W]
 * output = [N, C, (H-kH)/stride + 1, (W-kW)/stride + 1]
 */
void nc_max_pool2d(const NC_Tensor *input, NC_Tensor *output,
                   size_t kernel_size, size_t stride);

#endif /* NC_TENSOR_H */
