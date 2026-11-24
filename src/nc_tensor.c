#include "nc_tensor.h"

#include <math.h>
#include <stdalign.h>
#include <stdlib.h>
#include <string.h>

#define NC_TWO_PI 6.283185307179586476925286766559

/* ------------------------------- RNG -------------------------------- */

static uint64_t rng_state[4];
static bool rng_ready = false;
static bool rng_spare_ready = false;
static double rng_spare = 0.0;

static uint64_t splitmix64(uint64_t *state)
{
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static void rng_ensure_ready(void)
{
    if (!rng_ready) {
        nc_rng_seed(UINT64_C(0x9e3779b97f4a7c15));
    }
}

void nc_rng_seed(uint64_t seed)
{
    uint64_t state = seed;
    rng_state[0] = splitmix64(&state);
    rng_state[1] = splitmix64(&state);
    rng_state[2] = splitmix64(&state);
    rng_state[3] = splitmix64(&state);
    rng_ready = true;
    rng_spare_ready = false;
    rng_spare = 0.0;
}

static uint64_t rotl64(uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

/* Xoshiro256** */
uint64_t nc_rng_next_u64(void)
{
    rng_ensure_ready();

    const uint64_t result = rotl64(rng_state[1] * 5, 7) * 9;
    const uint64_t t = rng_state[1] << 17;

    rng_state[2] ^= rng_state[0];
    rng_state[3] ^= rng_state[1];
    rng_state[1] ^= rng_state[2];
    rng_state[0] ^= rng_state[3];
    rng_state[2] ^= t;
    rng_state[3] = rotl64(rng_state[3], 45);

    return result;
}

double nc_rng_next_double(void)
{
    /* 53 bits de mantissa -> [0, 1). */
    return (double)(nc_rng_next_u64() >> 11) * 0x1.0p-53;
}

double nc_rng_next_normal(void)
{
    if (rng_spare_ready) {
        rng_spare_ready = false;
        return rng_spare;
    }

    double u1 = nc_rng_next_double();
    if (u1 <= 0.0) {
        u1 = 0x1.0p-53; /* evita log(0) */
    }
    const double u2 = nc_rng_next_double();
    const double radius = sqrt(-2.0 * log(u1));
    const double angle = NC_TWO_PI * u2;

    rng_spare = radius * sin(angle);
    rng_spare_ready = true;
    return radius * cos(angle);
}

/* ----------------------------- Tensores ----------------------------- */

NC_Tensor *nc_tensor_create(NC_Arena *arena, const size_t shape[], size_t ndim,
                            bool requires_grad)
{
    if (arena == NULL || shape == NULL || ndim == 0 || ndim > 4) {
        return NULL;
    }

    size_t size = 1;
    for (size_t i = 0; i < ndim; ++i) {
        if (shape[i] == 0 || size > SIZE_MAX / shape[i]) {
            return NULL;
        }
        size *= shape[i];
    }
    if (size > SIZE_MAX / sizeof(float)) {
        return NULL;
    }

    const size_t saved_offset = arena->offset;

    NC_Tensor *t = (NC_Tensor *)nc_arena_alloc(arena, sizeof(*t), alignof(NC_Tensor));
    if (t == NULL) {
        arena->offset = saved_offset;
        return NULL;
    }

    t->data = (float *)nc_arena_alloc(arena, size * sizeof(float), NC_TENSOR_ALIGNMENT);
    if (t->data == NULL) {
        arena->offset = saved_offset;
        return NULL;
    }

    t->grad = NULL;
    if (requires_grad) {
        t->grad = (float *)nc_arena_alloc(arena, size * sizeof(float),
                                          NC_TENSOR_ALIGNMENT);
        if (t->grad == NULL) {
            arena->offset = saved_offset;
            return NULL;
        }
    }

    for (size_t i = 0; i < 4; ++i) {
        t->shape[i] = 0;
        t->strides[i] = 0;
    }
    for (size_t i = 0; i < ndim; ++i) {
        t->shape[i] = shape[i];
    }

    size_t stride = 1;
    for (size_t i = ndim; i-- > 0;) {
        t->strides[i] = stride;
        stride *= shape[i];
    }

    t->ndim = ndim;
    t->size = size;
    t->requires_grad = requires_grad;

    memset(t->data, 0, size * sizeof(float));
    if (t->grad != NULL) {
        memset(t->grad, 0, size * sizeof(float));
    }
    return t;
}

void nc_tensor_fill_zeros(NC_Tensor *t)
{
    if (t == NULL) {
        return;
    }
    memset(t->data, 0, t->size * sizeof(float));
    if (t->grad != NULL) {
        memset(t->grad, 0, t->size * sizeof(float));
    }
}

void nc_tensor_fill_random_normal(NC_Tensor *t, float mean, float stddev)
{
    if (t == NULL) {
        return;
    }
    for (size_t i = 0; i < t->size; ++i) {
        t->data[i] = (float)(mean + stddev * nc_rng_next_normal());
    }
}

void nc_tensor_fill_random_uniform(NC_Tensor *t)
{
    if (t == NULL) {
        return;
    }
    for (size_t i = 0; i < t->size; ++i) {
        t->data[i] = (float)nc_rng_next_double();
    }
}

/* ---------------------- Validação e aritmética segura ---------------- */

static bool checked_add(size_t a, size_t b, size_t *out)
{
    if (a > SIZE_MAX - b) {
        return false;
    }
    *out = a + b;
    return true;
}

static bool checked_mul(size_t a, size_t b, size_t *out)
{
    if (a != 0 && b > SIZE_MAX / a) {
        return false;
    }
    *out = a * b;
    return true;
}

/*
 * Confere que o tensor tem layout contíguo coerente com shape/size. Com isso,
 * todo offset calculado a partir de índices válidos é menor que size e não
 * pode transbordar — mesmo para tensores construídos fora de nc_tensor_create.
 */
bool nc_tensor_layout_valid(const NC_Tensor *t)
{
    if (t == NULL || t->data == NULL || t->ndim == 0 || t->ndim > 4) {
        return false;
    }

    size_t expected_stride = 1;
    for (size_t i = t->ndim; i-- > 0;) {
        if (t->shape[i] == 0 || t->strides[i] != expected_stride) {
            return false;
        }
        if (!checked_mul(expected_stride, t->shape[i], &expected_stride)) {
            return false;
        }
    }
    return t->size == expected_stride;
}

/* --------------------------- Álgebra linear -------------------------- */

/*
 * Tiles pequenos o bastante para caberem em ~2 KiB de stack (ambientes
 * embarcados/threads com stack limitada). O produto M*N por bloco é zeroado
 * apenas na região efetivamente usada.
 */
#define NC_TILE_M 16
#define NC_TILE_N 32
#define NC_TILE_K 16

static size_t batch_count(const NC_Tensor *t)
{
    size_t count = 1;
    for (size_t i = 0; i + 2 < t->ndim; ++i) {
        count *= t->shape[i];
    }
    return count;
}

/* Offset das dimensões de batch para o índice `b`. */
static size_t leading_offset(const NC_Tensor *t, size_t b)
{
    size_t offset = 0;
    size_t remaining = b;
    for (size_t i = t->ndim - 2; i-- > 0;) {
        const size_t dim = t->shape[i];
        if (dim == 0) {
            return 0; /* tensor malformado: nunca dividir por zero */
        }
        const size_t index = remaining % dim;
        remaining /= dim;
        offset += index * t->strides[i];
    }
    return offset;
}

/* Detecta sobreposição entre dois intervalos de floats (aliasing total ou parcial). */
static bool ranges_overlap(const float *a, size_t count_a,
                           const float *b, size_t count_b)
{
    const uintptr_t a_begin = (uintptr_t)a;
    const uintptr_t a_end = a_begin + count_a * sizeof(float);
    const uintptr_t b_begin = (uintptr_t)b;
    const uintptr_t b_end = b_begin + count_b * sizeof(float);
    return a_begin < b_end && b_begin < a_end;
}

/* Núcleo do GEMM: escreve em C, que não pode compartilhar memória com A/B. */
static void matmul_core(const NC_Tensor *A, const NC_Tensor *B, NC_Tensor *C)
{
    const size_t batch = batch_count(A);
    const size_t M = A->shape[A->ndim - 2];
    const size_t K = A->shape[A->ndim - 1];
    const size_t N = B->shape[B->ndim - 1];

    const size_t as0 = A->strides[A->ndim - 2];
    const size_t as1 = A->strides[A->ndim - 1];
    const size_t bs0 = B->strides[B->ndim - 2];
    const size_t bs1 = B->strides[B->ndim - 1];
    const size_t cs0 = C->strides[C->ndim - 2];
    const size_t cs1 = C->strides[C->ndim - 1];
    const bool unit_strides = (as1 == 1 && bs1 == 1);

    float tile[NC_TILE_M][NC_TILE_N];

    for (size_t b = 0; b < batch; ++b) {
        const float *abase = A->data + leading_offset(A, b);
        const float *bbase = B->data + leading_offset(B, b);
        float *cbase = C->data + leading_offset(C, b);

        for (size_t i0 = 0; i0 < M; i0 += NC_TILE_M) {
            const size_t imax = (M - i0 < NC_TILE_M) ? M - i0 : NC_TILE_M;
            for (size_t j0 = 0; j0 < N; j0 += NC_TILE_N) {
                const size_t jmax = (N - j0 < NC_TILE_N) ? N - j0 : NC_TILE_N;
                for (size_t i = 0; i < imax; ++i) {
                    memset(tile[i], 0, jmax * sizeof(float));
                }

                for (size_t k0 = 0; k0 < K; k0 += NC_TILE_K) {
                    const size_t kmax = (K - k0 < NC_TILE_K) ? K - k0 : NC_TILE_K;

                    if (unit_strides) {
                        for (size_t i = 0; i < imax; ++i) {
                            const float *arow = abase + (i0 + i) * as0 + k0;
                            for (size_t kk = 0; kk < kmax; ++kk) {
                                const float av = arow[kk];
                                const float *brow =
                                    bbase + (k0 + kk) * bs0 + j0;
                                for (size_t j = 0; j < jmax; ++j) {
                                    tile[i][j] += av * brow[j];
                                }
                            }
                        }
                    } else {
                        for (size_t i = 0; i < imax; ++i) {
                            for (size_t kk = 0; kk < kmax; ++kk) {
                                const float av =
                                    abase[(i0 + i) * as0 + (k0 + kk) * as1];
                                for (size_t j = 0; j < jmax; ++j) {
                                    tile[i][j] += av *
                                        bbase[(k0 + kk) * bs0 + (j0 + j) * bs1];
                                }
                            }
                        }
                    }
                }

                for (size_t i = 0; i < imax; ++i) {
                    for (size_t j = 0; j < jmax; ++j) {
                        cbase[(i0 + i) * cs0 + (j0 + j) * cs1] = tile[i][j];
                    }
                }
            }
        }
    }
}

void nc_matmul(const NC_Tensor *A, const NC_Tensor *B, NC_Tensor *C)
{
    if (A == NULL || B == NULL || C == NULL ||
        A->data == NULL || B->data == NULL || C->data == NULL) {
        return;
    }
    if (A->ndim < 2 || A->ndim != B->ndim || A->ndim != C->ndim) {
        return;
    }

    const size_t batch = batch_count(A);
    const size_t M = A->shape[A->ndim - 2];
    const size_t K = A->shape[A->ndim - 1];
    const size_t N = B->shape[B->ndim - 1];

    if (batch == 0 || batch_count(B) == 0 || batch_count(C) == 0 ||
        M == 0 || K == 0 || N == 0 ||
        K != B->shape[B->ndim - 2] ||
        M != C->shape[C->ndim - 2] ||
        N != C->shape[C->ndim - 1]) {
        return;
    }

    const bool overlaps = ranges_overlap(C->data, C->size, A->data, A->size) ||
                          ranges_overlap(C->data, C->size, B->data, B->size);

    if (!overlaps) {
        matmul_core(A, B, C);
        return;
    }

    /*
     * C compartilha memória com uma entrada: calcular diretamente corromperia
     * os dados ainda não lidos. Usamos um buffer temporário (caminho excepcional,
     * fora do hot path) e copiamos o resultado no layout real de C ao final.
     */
    if (M != 0 && batch > SIZE_MAX / M) {
        return;
    }
    const size_t plane = batch * M;
    if (N != 0 && plane > SIZE_MAX / N) {
        return;
    }
    const size_t total = plane * N;
    if (total > SIZE_MAX / sizeof(float)) {
        return;
    }

    float *temp = (float *)malloc(total * sizeof(float));
    if (temp == NULL) {
        return;
    }

    NC_Tensor tmp = *C;
    tmp.data = temp;
    tmp.grad = NULL;
    size_t stride = 1;
    for (size_t i = tmp.ndim; i-- > 0;) {
        tmp.strides[i] = stride;
        stride *= tmp.shape[i];
    }

    matmul_core(A, B, &tmp);

    const size_t cs0 = C->strides[C->ndim - 2];
    const size_t cs1 = C->strides[C->ndim - 1];
    for (size_t b = 0; b < batch; ++b) {
        const float *src = temp + b * M * N;
        float *dst = C->data + leading_offset(C, b);
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                dst[i * cs0 + j * cs1] = src[i * N + j];
            }
        }
    }
    free(temp);
}

static bool conv_input_shapes(const NC_Tensor *input, const NC_Tensor *kernel,
                              const NC_Tensor *output, size_t stride,
                              size_t padding, size_t *out_h, size_t *out_w)
{
    if (input == NULL || kernel == NULL || output == NULL) {
        return false;
    }
    if (input->ndim != 4 || kernel->ndim != 4 || output->ndim != 4 || stride == 0) {
        return false;
    }
    if (!nc_tensor_layout_valid(input) || !nc_tensor_layout_valid(kernel) ||
        !nc_tensor_layout_valid(output)) {
        return false;
    }

    const size_t c_in = input->shape[1];
    const size_t h = input->shape[2];
    const size_t w = input->shape[3];
    const size_t k_h = kernel->shape[2];
    const size_t k_w = kernel->shape[3];

    if (k_h == 0 || k_w == 0 || kernel->shape[1] != c_in) {
        return false;
    }

    size_t two_pad, padded_h, padded_w;
    if (!checked_mul(padding, 2, &two_pad) ||
        !checked_add(h, two_pad, &padded_h) ||
        !checked_add(w, two_pad, &padded_w)) {
        return false;
    }
    if (padded_h < k_h || padded_w < k_w) {
        return false;
    }

    *out_h = (padded_h - k_h) / stride + 1;
    *out_w = (padded_w - k_w) / stride + 1;

    return output->shape[0] == input->shape[0] &&
           output->shape[2] == *out_h &&
           output->shape[3] == *out_w;
}

void nc_conv2d(const NC_Tensor *input, const NC_Tensor *kernel,
               NC_Tensor *output, size_t stride, size_t padding)
{
    size_t o_h, o_w;
    if (!conv_input_shapes(input, kernel, output, stride, padding, &o_h, &o_w) ||
        output->shape[1] != kernel->shape[0]) {
        return;
    }

    nc_tensor_fill_zeros(output);

    const size_t n = input->shape[0];
    const size_t c_in = input->shape[1];
    const size_t c_out = kernel->shape[0];
    const size_t k_h = kernel->shape[2];
    const size_t k_w = kernel->shape[3];

    for (size_t ni = 0; ni < n; ++ni) {
        for (size_t oc = 0; oc < c_out; ++oc) {
            for (size_t oh = 0; oh < o_h; ++oh) {
                for (size_t ow = 0; ow < o_w; ++ow) {
                    float sum = 0.0f;
                    for (size_t ic = 0; ic < c_in; ++ic) {
                        for (size_t kh = 0; kh < k_h; ++kh) {
                            const size_t ih = oh * stride + kh;
                            if (ih < padding || ih >= input->shape[2] + padding) {
                                continue;
                            }
                            for (size_t kw = 0; kw < k_w; ++kw) {
                                const size_t iw = ow * stride + kw;
                                if (iw < padding || iw >= input->shape[3] + padding) {
                                    continue;
                                }
                                const size_t in_off =
                                    ni * input->strides[0] + ic * input->strides[1] +
                                    (ih - padding) * input->strides[2] +
                                    (iw - padding) * input->strides[3];
                                const size_t k_off =
                                    oc * kernel->strides[0] + ic * kernel->strides[1] +
                                    kh * kernel->strides[2] + kw * kernel->strides[3];
                                sum += input->data[in_off] * kernel->data[k_off];
                            }
                        }
                    }
                    const size_t out_off =
                        ni * output->strides[0] + oc * output->strides[1] +
                        oh * output->strides[2] + ow * output->strides[3];
                    output->data[out_off] = sum;
                }
            }
        }
    }
}

void nc_conv2d_transpose(const NC_Tensor *input, const NC_Tensor *kernel,
                         NC_Tensor *output, size_t stride, size_t padding)
{
    if (input == NULL || kernel == NULL || output == NULL) {
        return;
    }
    if (input->ndim != 4 || kernel->ndim != 4 || output->ndim != 4 || stride == 0) {
        return;
    }
    if (!nc_tensor_layout_valid(input) || !nc_tensor_layout_valid(kernel) ||
        !nc_tensor_layout_valid(output)) {
        return;
    }

    const size_t n = input->shape[0];
    const size_t c_in = input->shape[1];
    const size_t h = input->shape[2];
    const size_t w = input->shape[3];
    const size_t k_h = kernel->shape[2];
    const size_t k_w = kernel->shape[3];

    if (kernel->shape[0] != c_in || kernel->shape[1] == 0 ||
        k_h == 0 || k_w == 0) {
        return;
    }
    const size_t c_out = kernel->shape[1];
    size_t two_pad, base_h, span_h, base_w, span_w;
    if (!checked_mul(padding, 2, &two_pad) ||
        !checked_mul(h - 1, stride, &base_h) ||
        !checked_add(base_h, k_h, &span_h) ||
        !checked_mul(w - 1, stride, &base_w) ||
        !checked_add(base_w, k_w, &span_w)) {
        return;
    }
    if (span_h < two_pad || span_w < two_pad) {
        return;
    }
    const size_t out_h = span_h - two_pad;
    const size_t out_w = span_w - two_pad;

    if (out_h == 0 || out_w == 0) {
        return;
    }
    if (output->shape[0] != n || output->shape[1] != c_out ||
        output->shape[2] != out_h || output->shape[3] != out_w) {
        return;
    }

    nc_tensor_fill_zeros(output);

    for (size_t ni = 0; ni < n; ++ni) {
        for (size_t ic = 0; ic < c_in; ++ic) {
            for (size_t ih = 0; ih < h; ++ih) {
                for (size_t iw = 0; iw < w; ++iw) {
                    const size_t in_off =
                        ni * input->strides[0] + ic * input->strides[1] +
                        ih * input->strides[2] + iw * input->strides[3];
                    const float value = input->data[in_off];

                    for (size_t kh = 0; kh < k_h; ++kh) {
                        const size_t oh = ih * stride + kh;
                        if (oh < padding || oh >= span_h - padding) {
                            continue;
                        }
                        for (size_t kw = 0; kw < k_w; ++kw) {
                            const size_t ow = iw * stride + kw;
                            if (ow < padding || ow >= span_w - padding) {
                                continue;
                            }
                            for (size_t oc = 0; oc < c_out; ++oc) {
                                const size_t k_off =
                                    ic * kernel->strides[0] + oc * kernel->strides[1] +
                                    kh * kernel->strides[2] + kw * kernel->strides[3];
                                const size_t out_off =
                                    ni * output->strides[0] + oc * output->strides[1] +
                                    (oh - padding) * output->strides[2] +
                                    (ow - padding) * output->strides[3];
                                output->data[out_off] += value * kernel->data[k_off];
                            }
                        }
                    }
                }
            }
        }
    }
}

void nc_max_pool2d(const NC_Tensor *input, NC_Tensor *output,
                   size_t kernel_size, size_t stride)
{
    if (input == NULL || output == NULL || input->ndim != 4 ||
        output->ndim != 4 || kernel_size == 0 || stride == 0) {
        return;
    }
    if (!nc_tensor_layout_valid(input) || !nc_tensor_layout_valid(output)) {
        return;
    }

    const size_t h = input->shape[2];
    const size_t w = input->shape[3];
    if (h < kernel_size || w < kernel_size) {
        return;
    }

    const size_t o_h = (h - kernel_size) / stride + 1;
    const size_t o_w = (w - kernel_size) / stride + 1;
    /* (o_h-1)*stride + kernel_size - 1 <= h-1: os índices cabem em h. */

    if (output->shape[0] != input->shape[0] ||
        output->shape[1] != input->shape[1] ||
        output->shape[2] != o_h || output->shape[3] != o_w) {
        return;
    }

    for (size_t ni = 0; ni < input->shape[0]; ++ni) {
        for (size_t ic = 0; ic < input->shape[1]; ++ic) {
            for (size_t oh = 0; oh < o_h; ++oh) {
                for (size_t ow = 0; ow < o_w; ++ow) {
                    float best = -INFINITY;
                    for (size_t kh = 0; kh < kernel_size; ++kh) {
                        for (size_t kw = 0; kw < kernel_size; ++kw) {
                            const size_t in_off =
                                ni * input->strides[0] + ic * input->strides[1] +
                                (oh * stride + kh) * input->strides[2] +
                                (ow * stride + kw) * input->strides[3];
                            const float value = input->data[in_off];
                            if (value > best) {
                                best = value;
                            }
                        }
                    }
                    const size_t out_off =
                        ni * output->strides[0] + ic * output->strides[1] +
                        oh * output->strides[2] + ow * output->strides[3];
                    output->data[out_off] = best;
                }
            }
        }
    }
}

/* ------------------- Operações espaciais do Marco 3 ------------------- */

bool nc_conv2d_output_shape(const NC_Tensor *input, const NC_Tensor *kernel,
                            size_t stride, size_t padding, size_t out_shape[4])
{
    if (input == NULL || kernel == NULL || out_shape == NULL ||
        input->ndim != 4 || kernel->ndim != 4 || stride == 0 ||
        !nc_tensor_layout_valid(input) || !nc_tensor_layout_valid(kernel) ||
        kernel->shape[1] != input->shape[1] ||
        kernel->shape[2] == 0 || kernel->shape[3] == 0) {
        return false;
    }
    size_t two_pad, padded_h, padded_w;
    if (!checked_mul(padding, 2, &two_pad) ||
        !checked_add(input->shape[2], two_pad, &padded_h) ||
        !checked_add(input->shape[3], two_pad, &padded_w) ||
        padded_h < kernel->shape[2] || padded_w < kernel->shape[3]) {
        return false;
    }
    out_shape[0] = input->shape[0];
    out_shape[1] = kernel->shape[0];
    out_shape[2] = (padded_h - kernel->shape[2]) / stride + 1;
    out_shape[3] = (padded_w - kernel->shape[3]) / stride + 1;
    return out_shape[1] != 0 && out_shape[2] != 0 && out_shape[3] != 0;
}

bool nc_conv2d_transpose_output_shape(const NC_Tensor *input,
                                      const NC_Tensor *kernel, size_t stride,
                                      size_t padding, size_t out_shape[4])
{
    if (input == NULL || kernel == NULL || out_shape == NULL ||
        input->ndim != 4 || kernel->ndim != 4 || stride == 0 ||
        !nc_tensor_layout_valid(input) || !nc_tensor_layout_valid(kernel) ||
        kernel->shape[0] != input->shape[1] || kernel->shape[1] == 0 ||
        kernel->shape[2] == 0 || kernel->shape[3] == 0) {
        return false;
    }
    size_t two_pad, base_h, span_h, base_w, span_w;
    if (!checked_mul(padding, 2, &two_pad) ||
        !checked_mul(input->shape[2] - 1, stride, &base_h) ||
        !checked_add(base_h, kernel->shape[2], &span_h) ||
        !checked_mul(input->shape[3] - 1, stride, &base_w) ||
        !checked_add(base_w, kernel->shape[3], &span_w) ||
        span_h < two_pad || span_w < two_pad) {
        return false;
    }
    out_shape[0] = input->shape[0];
    out_shape[1] = kernel->shape[1];
    out_shape[2] = span_h - two_pad;
    out_shape[3] = span_w - two_pad;
    return out_shape[2] != 0 && out_shape[3] != 0;
}

bool nc_pool_output_shape(const NC_Tensor *input, size_t kernel_size,
                          size_t stride, size_t out_shape[4])
{
    if (input == NULL || out_shape == NULL || input->ndim != 4 ||
        kernel_size == 0 || stride == 0 ||
        !nc_tensor_layout_valid(input) ||
        input->shape[2] < kernel_size || input->shape[3] < kernel_size) {
        return false;
    }
    out_shape[0] = input->shape[0];
    out_shape[1] = input->shape[1];
    out_shape[2] = (input->shape[2] - kernel_size) / stride + 1;
    out_shape[3] = (input->shape[3] - kernel_size) / stride + 1;
    return true;
}

NC_Tensor *nc_tensor_flatten(NC_Arena *scratch, const NC_Tensor *t,
                             bool requires_grad)
{
    if (scratch == NULL || !nc_tensor_layout_valid(t) ||
        (requires_grad && t->grad == NULL)) {
        return NULL; /* sem buffer de gradiente não há como compartilhá-lo */
    }
    /* Para [N,C,H,W], o resultado é [N, C*H*W] (não [N*C, H*W]). */
    const size_t batch = t->shape[0];
    size_t features = 1;
    for (size_t i = 1; i < t->ndim; ++i) {
        if (features > SIZE_MAX / t->shape[i]) {
            return NULL;
        }
        features *= t->shape[i];
    }

    NC_Tensor *view = (NC_Tensor *)nc_arena_alloc(scratch, sizeof(*view),
                                                  alignof(NC_Tensor));
    if (view == NULL) {
        return NULL;
    }
    *view = *t;
    view->ndim = 2;
    view->shape[0] = batch;
    view->shape[1] = features;
    view->shape[2] = 0;
    view->shape[3] = 0;
    view->strides[0] = features;
    view->strides[1] = 1;
    view->strides[2] = 0;
    view->strides[3] = 0;
    view->size = batch * features;
    view->requires_grad = requires_grad;
    /* Compartilha o buffer de gradiente com o tensor original: assim o
     * backward de uma camada densa alcança a ativação convolucional. */
    view->grad = requires_grad ? t->grad : NULL;
    return view;
}

void nc_avg_pool2d(const NC_Tensor *input, NC_Tensor *output,
                   size_t kernel_size, size_t stride)
{
    if (input == NULL || output == NULL || input->ndim != 4 ||
        output->ndim != 4 || kernel_size == 0 || stride == 0 ||
        !nc_tensor_layout_valid(input) || !nc_tensor_layout_valid(output)) {
        return;
    }
    const size_t h = input->shape[2];
    const size_t w = input->shape[3];
    if (h < kernel_size || w < kernel_size) {
        return;
    }
    const size_t o_h = (h - kernel_size) / stride + 1;
    const size_t o_w = (w - kernel_size) / stride + 1;
    if (output->shape[0] != input->shape[0] ||
        output->shape[1] != input->shape[1] ||
        output->shape[2] != o_h || output->shape[3] != o_w) {
        return;
    }
    const float inv = 1.0f / (float)(kernel_size * kernel_size);

    for (size_t n = 0; n < input->shape[0]; ++n) {
        for (size_t c = 0; c < input->shape[1]; ++c) {
            for (size_t oh = 0; oh < o_h; ++oh) {
                for (size_t ow = 0; ow < o_w; ++ow) {
                    float sum = 0.0f;
                    for (size_t kh = 0; kh < kernel_size; ++kh) {
                        for (size_t kw = 0; kw < kernel_size; ++kw) {
                            const size_t off =
                                n * input->strides[0] + c * input->strides[1] +
                                (oh * stride + kh) * input->strides[2] +
                                (ow * stride + kw) * input->strides[3];
                            sum += input->data[off];
                        }
                    }
                    const size_t off =
                        n * output->strides[0] + c * output->strides[1] +
                        oh * output->strides[2] + ow * output->strides[3];
                    output->data[off] = sum * inv;
                }
            }
        }
    }
}

void nc_conv2d_im2col(NC_Arena *scratch, const NC_Tensor *input,
                      const NC_Tensor *kernel, NC_Tensor *output,
                      size_t stride, size_t padding)
{
    size_t o_h, o_w;
    if (scratch == NULL ||
        !conv_input_shapes(input, kernel, output, stride, padding, &o_h, &o_w) ||
        output->shape[1] != kernel->shape[0]) {
        return;
    }

    const size_t n = input->shape[0];
    const size_t c_in = input->shape[1];
    const size_t h = input->shape[2];
    const size_t w = input->shape[3];
    const size_t c_out = kernel->shape[0];
    const size_t k_h = kernel->shape[2];
    const size_t k_w = kernel->shape[3];

    size_t k_depth, cols, kernel_elems, col_elems, out_elems;
    if (!checked_mul(c_in, k_h, &k_depth) ||
        !checked_mul(k_depth, k_w, &k_depth) ||
        !checked_mul(n, o_h, &cols) ||
        !checked_mul(cols, o_w, &cols) ||
        !checked_mul(c_out, k_depth, &kernel_elems) ||
        !checked_mul(k_depth, cols, &col_elems) ||
        !checked_mul(c_out, cols, &out_elems)) {
        return;
    }

    float *col_data = (float *)nc_arena_alloc(scratch, col_elems * sizeof(float),
                                              NC_TENSOR_ALIGNMENT);
    float *out_data = (float *)nc_arena_alloc(scratch, out_elems * sizeof(float),
                                              NC_TENSOR_ALIGNMENT);
    if (col_data == NULL || out_data == NULL) {
        return;
    }
    memset(col_data, 0, col_elems * sizeof(float)); /* posições de padding */

    /* col[(c,kh,kw), (n,oh,ow)] */
    for (size_t ni = 0; ni < n; ++ni) {
        for (size_t oh = 0; oh < o_h; ++oh) {
            for (size_t ow = 0; ow < o_w; ++ow) {
                const size_t col_index = (ni * o_h + oh) * o_w + ow;
                for (size_t ci = 0; ci < c_in; ++ci) {
                    for (size_t kh = 0; kh < k_h; ++kh) {
                        const size_t ih = oh * stride + kh;
                        if (ih < padding || ih >= h + padding) {
                            continue;
                        }
                        for (size_t kw = 0; kw < k_w; ++kw) {
                            const size_t iw = ow * stride + kw;
                            if (iw < padding || iw >= w + padding) {
                                continue;
                            }
                            const size_t in_off =
                                ni * input->strides[0] + ci * input->strides[1] +
                                (ih - padding) * input->strides[2] +
                                (iw - padding) * input->strides[3];
                            col_data[((ci * k_h + kh) * k_w + kw) * cols + col_index] =
                                input->data[in_off];
                        }
                    }
                }
            }
        }
    }

    NC_Tensor col = { 0 }, kernel_view = { 0 }, out_matrix = { 0 };
    col.data = col_data;
    col.ndim = 2; col.shape[0] = k_depth; col.shape[1] = cols;
    col.strides[0] = cols; col.strides[1] = 1;
    col.size = col_elems;

    kernel_view.data = kernel->data;
    kernel_view.ndim = 2; kernel_view.shape[0] = c_out; kernel_view.shape[1] = k_depth;
    kernel_view.strides[0] = k_depth; kernel_view.strides[1] = 1;
    kernel_view.size = kernel_elems;

    out_matrix.data = out_data;
    out_matrix.ndim = 2; out_matrix.shape[0] = c_out; out_matrix.shape[1] = cols;
    out_matrix.strides[0] = cols; out_matrix.strides[1] = 1;
    out_matrix.size = out_elems;

    nc_matmul(&kernel_view, &col, &out_matrix);

    for (size_t ni = 0; ni < n; ++ni) {
        for (size_t oc = 0; oc < c_out; ++oc) {
            for (size_t oh = 0; oh < o_h; ++oh) {
                for (size_t ow = 0; ow < o_w; ++ow) {
                    const size_t out_off =
                        ni * output->strides[0] + oc * output->strides[1] +
                        oh * output->strides[2] + ow * output->strides[3];
                    output->data[out_off] =
                        out_data[oc * cols + (ni * o_h + oh) * o_w + ow];
                }
            }
        }
    }
}

void nc_conv2d_transpose_im2col(NC_Arena *scratch, const NC_Tensor *input,
                                const NC_Tensor *kernel, NC_Tensor *output,
                                size_t stride, size_t padding)
{
    if (scratch == NULL || input == NULL || kernel == NULL || output == NULL ||
        input->ndim != 4 || kernel->ndim != 4 || output->ndim != 4 ||
        stride == 0 || !nc_tensor_layout_valid(input) ||
        !nc_tensor_layout_valid(kernel) || !nc_tensor_layout_valid(output)) {
        return;
    }

    const size_t n = input->shape[0];
    const size_t c_in = input->shape[1];
    const size_t h = input->shape[2];
    const size_t w = input->shape[3];
    const size_t k_h = kernel->shape[2];
    const size_t k_w = kernel->shape[3];
    if (kernel->shape[0] != c_in || kernel->shape[1] == 0 || k_h == 0 || k_w == 0) {
        return;
    }
    const size_t c_out = kernel->shape[1];

    size_t two_pad, base_h, span_h, base_w, span_w;
    if (!checked_mul(padding, 2, &two_pad) ||
        !checked_mul(h - 1, stride, &base_h) ||
        !checked_add(base_h, k_h, &span_h) ||
        !checked_mul(w - 1, stride, &base_w) ||
        !checked_add(base_w, k_w, &span_w) ||
        span_h < two_pad || span_w < two_pad) {
        return;
    }
    const size_t out_h = span_h - two_pad;
    const size_t out_w = span_w - two_pad;
    if (out_h == 0 || out_w == 0 ||
        output->shape[0] != n || output->shape[1] != c_out ||
        output->shape[2] != out_h || output->shape[3] != out_w) {
        return;
    }

    size_t k_plane, kout, kmat_elems, spatial, in_elems, result_elems;
    if (!checked_mul(k_h, k_w, &k_plane) ||
        !checked_mul(c_out, k_plane, &kout) ||          /* colunas por canal de entrada */
        !checked_mul(c_in, kout, &kmat_elems) ||
        !checked_mul(n, h, &spatial) ||
        !checked_mul(spatial, w, &spatial) ||
        !checked_mul(c_in, spatial, &in_elems) ||
        !checked_mul(kout, spatial, &result_elems)) {
        return;
    }

    float *kmat = (float *)nc_arena_alloc(scratch, kmat_elems * sizeof(float),
                                          NC_TENSOR_ALIGNMENT);
    float *imat = (float *)nc_arena_alloc(scratch, in_elems * sizeof(float),
                                          NC_TENSOR_ALIGNMENT);
    float *result = (float *)nc_arena_alloc(scratch, result_elems * sizeof(float),
                                            NC_TENSOR_ALIGNMENT);
    if (kmat == NULL || imat == NULL || result == NULL) {
        return;
    }

    /* kmat[ic, (oc,kh,kw)] = kernel[ic,oc,kh,kw] */
    for (size_t ic = 0; ic < c_in; ++ic) {
        for (size_t oc = 0; oc < c_out; ++oc) {
            for (size_t kh = 0; kh < k_h; ++kh) {
                for (size_t kw = 0; kw < k_w; ++kw) {
                    const size_t koff =
                        ic * kernel->strides[0] + oc * kernel->strides[1] +
                        kh * kernel->strides[2] + kw * kernel->strides[3];
                    kmat[ic * kout + (oc * k_h + kh) * k_w + kw] =
                        kernel->data[koff];
                }
            }
        }
    }
    /* imat[ic, (n,ih,iw)] */
    for (size_t ni = 0; ni < n; ++ni) {
        for (size_t ic = 0; ic < c_in; ++ic) {
            for (size_t ih = 0; ih < h; ++ih) {
                for (size_t iw = 0; iw < w; ++iw) {
                    const size_t in_off =
                        ni * input->strides[0] + ic * input->strides[1] +
                        ih * input->strides[2] + iw * input->strides[3];
                    imat[ic * spatial + (ni * h + ih) * w + iw] =
                        input->data[in_off];
                }
            }
        }
    }

    NC_Tensor kt = { 0 }, it = { 0 }, rt = { 0 };
    kt.data = kmat;
    kt.ndim = 2; kt.shape[0] = kout; kt.shape[1] = c_in;
    kt.strides[0] = 1; kt.strides[1] = kout; /* transposta de [c_in, kout] */
    kt.size = kmat_elems;
    it.data = imat;
    it.ndim = 2; it.shape[0] = c_in; it.shape[1] = spatial;
    it.strides[0] = spatial; it.strides[1] = 1;
    it.size = in_elems;
    rt.data = result;
    rt.ndim = 2; rt.shape[0] = kout; rt.shape[1] = spatial;
    rt.strides[0] = spatial; rt.strides[1] = 1;
    rt.size = result_elems;

    nc_matmul(&kt, &it, &rt);

    nc_tensor_fill_zeros(output);
    for (size_t ni = 0; ni < n; ++ni) {
        for (size_t oc = 0; oc < c_out; ++oc) {
            for (size_t kh = 0; kh < k_h; ++kh) {
                for (size_t kw = 0; kw < k_w; ++kw) {
                    const size_t row = (oc * k_h + kh) * k_w + kw;
                    for (size_t ih = 0; ih < h; ++ih) {
                        const size_t oh = ih * stride + kh;
                        if (oh < padding || oh >= out_h + padding) {
                            continue;
                        }
                        for (size_t iw = 0; iw < w; ++iw) {
                            const size_t ow = iw * stride + kw;
                            if (ow < padding || ow >= out_w + padding) {
                                continue;
                            }
                            const size_t out_off =
                                ni * output->strides[0] + oc * output->strides[1] +
                                (oh - padding) * output->strides[2] +
                                (ow - padding) * output->strides[3];
                            output->data[out_off] +=
                                result[row * spatial + (ni * h + ih) * w + iw];
                        }
                    }
                }
            }
        }
    }
}
