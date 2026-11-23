#include "test_common.h"
#include "neurochart.h"

#include <stdint.h>
#include <string.h>

/* `values` deve apontar para pelo menos t->size floats. */
static NC_Tensor *make_tensor(NC_Arena *arena, const size_t shape[], size_t ndim,
                              const float *values)
{
    NC_Tensor *t = nc_tensor_create(arena, shape, ndim, false);
    if (t != NULL && values != NULL) {
        memcpy(t->data, values, t->size * sizeof(float));
    }
    return t;
}

static void test_shape_strides_and_allocation(void)
{
    NC_Arena arena = nc_arena_init(64 * 1024);

    const size_t s1[1] = { 7 };
    NC_Tensor *t1 = nc_tensor_create(&arena, s1, 1, false);
    NC_CHECK(t1 != NULL);
    NC_CHECK(t1->size == 7);
    NC_CHECK(t1->shape[0] == 7 && t1->strides[0] == 1);
    NC_CHECK(t1->grad == NULL);
    NC_CHECK(((uintptr_t)t1->data % NC_TENSOR_ALIGNMENT) == 0);

    const size_t s2[2] = { 3, 5 };
    NC_Tensor *t2 = nc_tensor_create(&arena, s2, 2, true);
    NC_CHECK(t2 != NULL);
    NC_CHECK(t2->size == 15);
    NC_CHECK(t2->strides[0] == 5 && t2->strides[1] == 1);
    NC_CHECK(t2->requires_grad && t2->grad != NULL);

    const size_t s4[4] = { 2, 3, 4, 5 };
    NC_Tensor *t4 = nc_tensor_create(&arena, s4, 4, false);
    NC_CHECK(t4 != NULL);
    NC_CHECK(t4->size == 120);
    NC_CHECK(t4->strides[0] == 60 && t4->strides[1] == 20 &&
             t4->strides[2] == 5 && t4->strides[3] == 1);
    for (size_t i = 0; i < t4->size; ++i) {
        NC_CHECK(t4->data[i] == 0.0f); /* zeros na criação */
    }

    nc_tensor_fill_zeros(t2);
    for (size_t i = 0; i < t2->size; ++i) {
        NC_CHECK(t2->grad[i] == 0.0f);
    }

    NC_CHECK(nc_tensor_create(NULL, s2, 2, false) == NULL);
    NC_CHECK(nc_tensor_create(&arena, NULL, 2, false) == NULL);
    NC_CHECK(nc_tensor_create(&arena, s2, 0, false) == NULL);
    NC_CHECK(nc_tensor_create(&arena, s2, 5, false) == NULL);
    const size_t bad[2] = { 0, 4 };
    NC_CHECK(nc_tensor_create(&arena, bad, 2, false) == NULL);

    NC_Arena tiny = nc_arena_init(16);
    const size_t saved = tiny.offset;
    NC_CHECK(nc_tensor_create(&tiny, s4, 4, false) == NULL);
    NC_CHECK(tiny.offset == saved); /* rollback em caso de OOM */

    nc_arena_free(&tiny);
    nc_arena_free(&arena);
}

static void test_rng(void)
{
    nc_rng_seed(42);
    const uint64_t a = nc_rng_next_u64();
    const double ua = nc_rng_next_double();
    const double na = nc_rng_next_normal();

    nc_rng_seed(42);
    NC_CHECK(nc_rng_next_u64() == a);
    NC_CHECK_NEAR(nc_rng_next_double(), ua, 1e-15);
    NC_CHECK_NEAR(nc_rng_next_normal(), na, 1e-15);

    /* seed 0 também é determinístico e reinicia o par em cache do Box-Muller. */
    nc_rng_seed(0);
    const uint64_t zero_seed_u64 = nc_rng_next_u64();
    const double zero_seed_normal = nc_rng_next_normal();
    nc_rng_seed(0);
    NC_CHECK(nc_rng_next_u64() == zero_seed_u64);
    NC_CHECK_NEAR(nc_rng_next_normal(), zero_seed_normal, 1e-15);

    for (size_t i = 0; i < 1000; ++i) {
        const double u = nc_rng_next_double();
        NC_CHECK(u >= 0.0 && u < 1.0);
    }

    NC_Arena arena = nc_arena_init(1024 * 1024);
    const size_t shape[1] = { 100000 };
    NC_Tensor *t = nc_tensor_create(&arena, shape, 1, false);
    NC_CHECK(t != NULL);

    nc_rng_seed(7);
    nc_tensor_fill_random_normal(t, 0.0f, 1.0f);
    double sum = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < t->size; ++i) {
        NC_CHECK(isfinite(t->data[i]));
        sum += t->data[i];
        sum_sq += (double)t->data[i] * t->data[i];
    }
    const double mean = sum / (double)t->size;
    const double stddev = sqrt(sum_sq / (double)t->size - mean * mean);
    NC_CHECK(fabs(mean) < 0.02);
    NC_CHECK(fabs(stddev - 1.0) < 0.02);

    nc_rng_seed(7);
    NC_Tensor *copy = nc_tensor_create(&arena, shape, 1, false);
    nc_tensor_fill_random_normal(copy, 0.0f, 1.0f);
    NC_CHECK(memcmp(t->data, copy->data, t->size * sizeof(float)) == 0);

    nc_tensor_fill_random_uniform(t);
    for (size_t i = 0; i < t->size; ++i) {
        NC_CHECK(t->data[i] >= 0.0f && t->data[i] < 1.0f);
    }

    nc_arena_free(&arena);
}

static void test_matmul(void)
{
    NC_Arena arena = nc_arena_init(4 * 1024 * 1024);

    const size_t sa[2] = { 2, 3 };
    const size_t sb[2] = { 3, 2 };
    const size_t sc[2] = { 2, 2 };
    const float av[] = { 1, 2, 3, 4, 5, 6 };
    const float bv[] = { 7, 8, 9, 10, 11, 12 };
    NC_Tensor *A = make_tensor(&arena, sa, 2, av);
    NC_Tensor *B = make_tensor(&arena, sb, 2, bv);
    NC_Tensor *C = nc_tensor_create(&arena, sc, 2, false);

    nc_matmul(A, B, C);
    const float expected[] = { 58, 64, 139, 154 };
    for (size_t i = 0; i < 4; ++i) {
        NC_CHECK_NEAR(C->data[i], expected[i], 1e-5f);
    }

    /* Dimensões maiores que os tiles (32x64x16) exercitam o caminho com tiling. */
    const size_t m = 70, k = 45, n = 130;
    const size_t shape_a[2] = { m, k };
    const size_t shape_b[2] = { k, n };
    const size_t shape_c[2] = { m, n };
    NC_Tensor *X = nc_tensor_create(&arena, shape_a, 2, false);
    NC_Tensor *Y = nc_tensor_create(&arena, shape_b, 2, false);
    NC_Tensor *Z = nc_tensor_create(&arena, shape_c, 2, false);
    nc_rng_seed(123);
    nc_tensor_fill_random_normal(X, 0.0f, 1.0f);
    nc_tensor_fill_random_normal(Y, 0.0f, 1.0f);
    nc_matmul(X, Y, Z);

    for (size_t i = 0; i < m; i += 7) {
        for (size_t j = 0; j < n; j += 11) {
            double reference = 0.0;
            for (size_t p = 0; p < k; ++p) {
                reference += (double)X->data[i * k + p] * Y->data[p * n + j];
            }
            NC_CHECK_NEAR(Z->data[i * n + j], reference, 2e-3);
        }
    }

    /* Caso em batch (ndim = 3). */
    const size_t bat[3] = { 2, 2, 3 };
    const size_t bbt[3] = { 2, 3, 2 };
    const size_t bct[3] = { 2, 2, 2 };
    NC_Tensor *BA = nc_tensor_create(&arena, bat, 3, false);
    NC_Tensor *BB = nc_tensor_create(&arena, bbt, 3, false);
    NC_Tensor *BC = nc_tensor_create(&arena, bct, 3, false);
    memcpy(BA->data, av, sizeof(av));
    memcpy(BA->data + 6, av, sizeof(av));
    memcpy(BB->data, bv, sizeof(bv));
    memcpy(BB->data + 6, bv, sizeof(bv));
    nc_matmul(BA, BB, BC);
    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 4; ++i) {
            NC_CHECK_NEAR(BC->data[b * 4 + i], expected[i], 1e-5f);
        }
    }

    /* Dimensões incompatíveis são ignoradas sem corromper a saída. */
    nc_tensor_fill_zeros(C);
    nc_matmul(B, A, C);
    NC_CHECK(C->data[0] == 0.0f);

    nc_arena_free(&arena);
}

static void test_conv2d(void)
{
    NC_Arena arena = nc_arena_init(64 * 1024);

    const size_t in_shape[4] = { 1, 1, 3, 3 };
    const size_t k_shape[4] = { 1, 1, 2, 2 };
    const size_t out_shape[4] = { 1, 1, 2, 2 };
    const float in_values[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const float k_values[] = { 1, 1, 1, 1 };
    NC_Tensor *in = make_tensor(&arena, in_shape, 4, in_values);
    NC_Tensor *kernel = make_tensor(&arena, k_shape, 4, k_values);
    NC_Tensor *out = nc_tensor_create(&arena, out_shape, 4, false);

    nc_conv2d(in, kernel, out, 1, 0);
    const float expected[] = { 12, 16, 24, 28 };
    for (size_t i = 0; i < 4; ++i) {
        NC_CHECK_NEAR(out->data[i], expected[i], 1e-6f);
    }

    /* padding = 1 => saída 4x4; cantos veem apenas um elemento de entrada. */
    const size_t padded_shape[4] = { 1, 1, 4, 4 };
    NC_Tensor *padded = nc_tensor_create(&arena, padded_shape, 4, false);
    nc_conv2d(in, kernel, padded, 1, 1);
    NC_CHECK_NEAR(padded->data[0], 1.0f, 1e-6f);
    NC_CHECK_NEAR(padded->data[3], 3.0f, 1e-6f);
    NC_CHECK_NEAR(padded->data[15], 9.0f, 1e-6f);
    NC_CHECK_NEAR(padded->data[5], 12.0f, 1e-6f); /* center: 1+2+4+5 */

    nc_arena_free(&arena);
}

static void test_conv2d_transpose_and_pool(void)
{
    NC_Arena arena = nc_arena_init(64 * 1024);

    const size_t in_shape[4] = { 1, 1, 2, 2 };
    const size_t k_shape[4] = { 1, 1, 2, 2 };
    const size_t out_shape[4] = { 1, 1, 3, 3 };
    const float in_values[] = { 1, 2, 3, 4 };
    const float k_values[] = { 1, 1, 1, 1 };
    NC_Tensor *in = make_tensor(&arena, in_shape, 4, in_values);
    NC_Tensor *kernel = make_tensor(&arena, k_shape, 4, k_values);
    NC_Tensor *out = nc_tensor_create(&arena, out_shape, 4, false);

    nc_conv2d_transpose(in, kernel, out, 1, 0);
    const float expected[] = { 1, 3, 2, 4, 10, 6, 3, 7, 4 };
    for (size_t i = 0; i < 9; ++i) {
        NC_CHECK_NEAR(out->data[i], expected[i], 1e-6f);
    }

    const size_t pool_in[4] = { 1, 1, 4, 4 };
    const size_t pool_out[4] = { 1, 1, 2, 2 };
    const float pool_values[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    NC_Tensor *p_in = make_tensor(&arena, pool_in, 4, pool_values);
    NC_Tensor *p_out = nc_tensor_create(&arena, pool_out, 4, false);
    nc_max_pool2d(p_in, p_out, 2, 2);
    const float pool_expected[] = { 6, 8, 14, 16 };
    for (size_t i = 0; i < 4; ++i) {
        NC_CHECK_NEAR(p_out->data[i], pool_expected[i], 1e-6f);
    }

    nc_arena_free(&arena);
}

static void test_matmul_aliasing(void)
{
    NC_Arena arena = nc_arena_init(2 * 1024 * 1024);

    /* Caso multi-tile (N=128 > tile de 64): sem cópia temporária, C==A
     * corrompe as colunas de A que ainda serão lidas pelos tiles seguintes. */
    const size_t shape[2] = { 128, 128 };
    const size_t total = 128 * 128;
    NC_Tensor *A = nc_tensor_create(&arena, shape, 2, false);
    NC_Tensor *B = nc_tensor_create(&arena, shape, 2, false);
    NC_Tensor *ref = nc_tensor_create(&arena, shape, 2, false);
    NC_Tensor *saved = nc_tensor_create(&arena, shape, 2, false);

    for (size_t i = 0; i < total; ++i) {
        A->data[i] = (float)(i + 1);
        B->data[i] = 1.0f; /* soma das linhas de A */
    }
    memcpy(saved->data, A->data, total * sizeof(float));
    nc_matmul(A, B, ref);

    nc_matmul(A, B, A); /* C == A */
    for (size_t i = 0; i < total; ++i) {
        NC_CHECK_NEAR(A->data[i], ref->data[i], 1e-3f);
    }

    memcpy(A->data, saved->data, total * sizeof(float));
    nc_matmul(A, B, ref);
    nc_matmul(A, B, B); /* C == B */
    for (size_t i = 0; i < total; ++i) {
        NC_CHECK_NEAR(B->data[i], ref->data[i], 1e-3f);
    }

    /* Aliasing em matriz pequena e em tensor em batch. */
    const size_t s2[2] = { 2, 2 };
    const float av[] = { 1, 2, 3, 4 };
    const float bv[] = { 1, 1, 1, 1 };
    NC_Tensor *a2 = make_tensor(&arena, s2, 2, av);
    NC_Tensor *b2 = make_tensor(&arena, s2, 2, bv);
    nc_matmul(a2, b2, a2);
    const float expected2[] = { 3, 3, 7, 7 };
    for (size_t i = 0; i < 4; ++i) {
        NC_CHECK_NEAR(a2->data[i], expected2[i], 1e-6f);
    }

    const size_t sb[3] = { 2, 2, 2 };
    const float abv[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    const float bbv[] = { 1, 1, 1, 1, 1, 1, 1, 1 };
    NC_Tensor *ab = make_tensor(&arena, sb, 3, abv);
    NC_Tensor *bb = make_tensor(&arena, sb, 3, bbv);
    nc_matmul(ab, bb, ab);
    const float expected_batch[] = { 3, 3, 7, 7, 11, 11, 15, 15 };
    for (size_t i = 0; i < 8; ++i) {
        NC_CHECK_NEAR(ab->data[i], expected_batch[i], 1e-6f);
    }

    nc_arena_free(&arena);
}

static void test_conv_stride_batch_and_pool_stride(void)
{
    NC_Arena arena = nc_arena_init(64 * 1024);

    const size_t in_shape[4] = { 2, 1, 4, 4 };
    const size_t k_shape[4] = { 1, 1, 2, 2 };
    const size_t out_shape[4] = { 2, 1, 2, 2 };
    float values[32];
    for (size_t i = 0; i < 32; ++i) {
        values[i] = (float)(i + 1);
    }
    const float k_values[] = { 1, 1, 1, 1 };
    NC_Tensor *in = make_tensor(&arena, in_shape, 4, values);
    NC_Tensor *kernel = make_tensor(&arena, k_shape, 4, k_values);
    NC_Tensor *out = nc_tensor_create(&arena, out_shape, 4, false);

    nc_conv2d(in, kernel, out, 2, 0);
    const float expected[] = { 14, 22, 46, 54, 78, 86, 110, 118 };
    for (size_t i = 0; i < 8; ++i) {
        NC_CHECK_NEAR(out->data[i], expected[i], 1e-6f);
    }

    const size_t pool_in[4] = { 1, 1, 3, 3 };
    const size_t pool_out[4] = { 1, 1, 2, 2 };
    const float pool_values[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    NC_Tensor *p_in = make_tensor(&arena, pool_in, 4, pool_values);
    NC_Tensor *p_out = nc_tensor_create(&arena, pool_out, 4, false);
    nc_max_pool2d(p_in, p_out, 2, 1); /* janelas sobrepostas */
    const float pool_expected[] = { 5, 6, 8, 9 };
    for (size_t i = 0; i < 4; ++i) {
        NC_CHECK_NEAR(p_out->data[i], pool_expected[i], 1e-6f);
    }

    nc_arena_free(&arena);
}

static void test_hardening_and_strided_views(void)
{
    NC_Arena arena = nc_arena_init(128 * 1024);

    /* Views com strides não-unitários exercitam o caminho genérico do GEMM. */
    const size_t sa[2] = { 4, 6 };
    const size_t sb[2] = { 3, 4 };
    NC_Tensor *base_a = nc_tensor_create(&arena, sa, 2, false);
    NC_Tensor *base_b = nc_tensor_create(&arena, sb, 2, false);
    for (size_t i = 0; i < 24; ++i) {
        base_a->data[i] = (float)(i + 1);
    }
    for (size_t i = 0; i < 12; ++i) {
        base_b->data[i] = (float)(12 - i);
    }

    NC_Tensor av = *base_a;
    av.shape[1] = 3; av.strides[0] = 6; av.strides[1] = 2; av.size = 12;
    NC_Tensor bv = *base_b;
    bv.shape[1] = 2; bv.strides[0] = 4; bv.strides[1] = 2; bv.size = 6;

    const size_t sc[2] = { 4, 2 };
    NC_Tensor *C = nc_tensor_create(&arena, sc, 2, false);
    nc_matmul(&av, &bv, C);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            float reference = 0.0f;
            for (size_t k = 0; k < 3; ++k) {
                reference += base_a->data[i * 6 + k * 2] *
                             base_b->data[k * 4 + j * 2];
            }
            NC_CHECK_NEAR(C->data[i * 2 + j], reference, 1e-3f);
        }
    }

    /* Dimensão zero em tensor malformado é rejeitada sem tocar na saída. */
    const float sentinel = C->data[0];
    NC_Tensor bad = *C;
    bad.shape[1] = 0;
    bad.size = 0;
    nc_matmul(&av, &bv, &bad);
    NC_CHECK(C->data[0] == sentinel);

    /* Padding grande é válido; padding/stride extremos são rejeitados. */
    const size_t in_s[4] = { 1, 1, 3, 3 };
    const size_t k_s[4] = { 1, 1, 2, 2 };
    const size_t big_out_s[4] = { 1, 1, 22, 22 };
    const size_t small_out_s[4] = { 1, 1, 1, 1 };
    float ones9[9], ones4[4];
    for (size_t i = 0; i < 9; ++i) ones9[i] = 1.0f;
    for (size_t i = 0; i < 4; ++i) ones4[i] = 1.0f;
    NC_Tensor *in = make_tensor(&arena, in_s, 4, ones9);
    NC_Tensor *kernel = make_tensor(&arena, k_s, 4, ones4);
    NC_Tensor *big_out = nc_tensor_create(&arena, big_out_s, 4, false);
    NC_Tensor *small_out = nc_tensor_create(&arena, small_out_s, 4, false);

    nc_conv2d(in, kernel, big_out, 1, 10);
    NC_CHECK_NEAR(big_out->data[10 * 22 + 10], 4.0f, 1e-6f);
    NC_CHECK_NEAR(big_out->data[0], 0.0f, 1e-6f);

    nc_conv2d(in, kernel, big_out, 1, SIZE_MAX / 2);
    NC_CHECK_NEAR(big_out->data[10 * 22 + 10], 4.0f, 1e-6f); /* inalterado */

    nc_conv2d_transpose(in, kernel, small_out, SIZE_MAX / 2, 0);
    NC_CHECK(small_out->data[0] == 0.0f);

    nc_max_pool2d(in, small_out, SIZE_MAX, 1);
    NC_CHECK(small_out->data[0] == 0.0f);

    /* Tensor com strides inconsistentes é rejeitado pela validação de layout. */
    const size_t valid_out_s[4] = { 1, 1, 2, 2 };
    NC_Tensor *valid_out = nc_tensor_create(&arena, valid_out_s, 4, false);
    NC_Tensor bad_layout = *in;
    bad_layout.strides[3] = 2;
    nc_conv2d(&bad_layout, kernel, valid_out, 1, 0);
    NC_CHECK(valid_out->data[0] == 0.0f);

    nc_arena_free(&arena);
}

int main(void)
{
    printf("=== test_tensor (Marco 1) ===\n");
    test_shape_strides_and_allocation();
    test_rng();
    test_matmul();
    test_matmul_aliasing();
    test_conv2d();
    test_conv2d_transpose_and_pool();
    test_conv_stride_batch_and_pool_stride();
    test_hardening_and_strided_views();
    NC_TEST_SUMMARY("test_tensor");
}
