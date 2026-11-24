#include "test_common.h"
#include "neurochart.h"

#include <string.h>

static NC_Tensor *make4(NC_Arena *arena, const size_t shape[4],
                        const float *values, bool requires_grad)
{
    NC_Tensor *t = nc_tensor_create(arena, shape, 4, requires_grad);
    if (t != NULL && values != NULL) {
        memcpy(t->data, values, t->size * sizeof(float));
    }
    return t;
}

static float sum_values(const NC_Tensor *t)
{
    float sum = 0.0f;
    for (size_t i = 0; i < t->size; ++i) sum += t->data[i];
    return sum;
}

static void zeros(NC_Tensor *t)
{
    if (t != NULL && t->grad != NULL) {
        memset(t->grad, 0, t->size * sizeof(float));
    }
}

static void test_im2col_matches_direct(void)
{
    NC_Arena model = nc_arena_init(1024 * 1024);
    NC_Arena scratch = nc_arena_init(1024 * 1024);

    const size_t is[4] = { 1, 2, 4, 4 };
    const size_t ks[4] = { 3, 2, 2, 2 };
    const size_t os[4] = { 1, 3, 3, 3 };
    NC_Tensor *in = nc_tensor_create(&model, is, 4, false);
    NC_Tensor *k = nc_tensor_create(&model, ks, 4, false);
    NC_Tensor *a = nc_tensor_create(&model, os, 4, false);
    NC_Tensor *b = nc_tensor_create(&model, os, 4, false);
    for (size_t i = 0; i < in->size; ++i) in->data[i] = (float)(i % 7) - 2.0f;
    for (size_t i = 0; i < k->size; ++i) k->data[i] = (float)(i % 5) - 1.0f;

    nc_conv2d(in, k, a, 1, 1);
    nc_conv2d_im2col(&scratch, in, k, b, 1, 1);
    for (size_t i = 0; i < a->size; ++i) NC_CHECK_NEAR(b->data[i], a->data[i], 1e-5f);

    const size_t kts[4] = { 2, 3, 2, 2 };
    const size_t ts[4] = { 1, 2, 6, 6 };
    NC_Tensor *kt = nc_tensor_create(&model, kts, 4, false);
    NC_Tensor *ta = nc_tensor_create(&model, ts, 4, false);
    NC_Tensor *tb = nc_tensor_create(&model, ts, 4, false);
    for (size_t i = 0; i < kt->size; ++i) kt->data[i] = (float)(i % 3) - 1.0f;
    nc_conv2d_transpose(in, kt, ta, 2, 0);
    nc_conv2d_transpose_im2col(&scratch, in, kt, tb, 2, 0);
    for (size_t i = 0; i < ta->size; ++i) NC_CHECK_NEAR(tb->data[i], ta->data[i], 1e-5f);

    /* Flatten 4D: [N,C,H,W] -> [N, C*H*W]. */
    NC_Tensor *flat = nc_tensor_flatten(&scratch, in, false);
    NC_CHECK(flat != NULL);
    NC_CHECK(flat->shape[0] == 1 && flat->shape[1] == 32);
    for (size_t i = 0; i < flat->size; ++i) NC_CHECK(flat->data[i] == in->data[i]);

    /* avg pooling: valores conhecidos */
    const size_t ps[4] = { 1, 1, 4, 4 };
    const size_t pos[4] = { 1, 1, 2, 2 };
    const float pv[] = { 1, 1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7 };
    NC_Tensor *pool_in = make4(&model, ps, pv, false);
    NC_Tensor *pool_out = nc_tensor_create(&model, pos, 4, false);
    nc_avg_pool2d(pool_in, pool_out, 2, 2);
    const float expected[] = { 2, 2, 6, 6 };
    for (size_t i = 0; i < 4; ++i) NC_CHECK_NEAR(pool_out->data[i], expected[i], 1e-6f);

    nc_arena_free(&scratch);
    nc_arena_free(&model);
}

static NC_Tensor *spatial_forward(NC_Tape *tape, NC_Arena *scratch, int op,
                                   NC_Tensor *in, NC_Tensor *k)
{
    nc_autograd_tape_reset(tape);
    switch (op) {
    case 0: return nc_nn_conv2d(tape, scratch, in, k, 1, 1);
    case 1: return nc_nn_conv2d_transpose(tape, scratch, in, k, 2, 1);
    case 2: return nc_nn_max_pool2d(tape, scratch, in, 2, 1);
    default: return nc_nn_avg_pool2d(tape, scratch, in, 2, 1);
    }
}

static float spatial_sum(NC_Tape *tape, NC_Arena *scratch, int op,
                         NC_Tensor *in, NC_Tensor *k)
{
    NC_Tensor *out = spatial_forward(tape, scratch, op, in, k);
    return out != NULL ? sum_values(out) : 0.0f;
}

static void test_spatial_gradients(void)
{
    NC_Arena model = nc_arena_init(2 * 1024 * 1024);
    NC_Arena scratch = nc_arena_init(2 * 1024 * 1024);
    NC_Tape *tape = nc_autograd_tape_init(&model, 64);
    const float step = 1e-3f;

    const size_t is[4] = { 1, 1, 4, 4 };
    const size_t ks[4] = { 1, 1, 2, 2 };
    NC_Tensor *in = nc_tensor_create(&model, is, 4, true);
    NC_Tensor *k = nc_tensor_create(&model, ks, 4, true);
    for (size_t i = 0; i < 16; ++i) in->data[i] = (float)(i % 5) - 2.0f;
    for (size_t i = 0; i < 4; ++i) k->data[i] = (float)(i % 3) - 1.0f;

    for (int op = 0; op < 4; ++op) {
        nc_arena_reset(&scratch);
        NC_Tensor *out = spatial_forward(tape, &scratch, op, in, k);
        NC_CHECK(out != NULL);
        for (size_t i = 0; i < out->size; ++i) out->grad[i] = 1.0f;
        zeros(in);
        zeros(k);
        nc_autograd_backward(tape);

        NC_Tensor *weights = op < 2 ? k : in;
        const size_t count = op < 2 ? k->size : in->size;
        float analytic[16];
        memcpy(analytic, weights->grad, count * sizeof(float));

        for (size_t i = 0; i < count; ++i) {
            if (op == 2 && i > 1) continue; /* amostra representativa do maxpool */
            if (op == 3 && i > 1) continue;
            if (op == 0 && i >= 4) break;   /* analisamos dK na conv */
            if (op == 1 && i >= 16) break;  /* analisamos dIn na transposta */
            const float original = weights->data[i];
            weights->data[i] = original + step;
            nc_arena_reset(&scratch);
            const float plus = spatial_sum(tape, &scratch, op, in, k);
            weights->data[i] = original - step;
            nc_arena_reset(&scratch);
            const float minus = spatial_sum(tape, &scratch, op, in, k);
            weights->data[i] = original;
            NC_CHECK_NEAR(analytic[i], (plus - minus) / (2.0f * step), 2e-2f);
        }
    }

    nc_arena_free(&scratch);
    nc_arena_free(&model);
}

static float spatial_train_step(NC_Graph *graph, NC_Tensor *x, NC_Tensor *target,
                                NC_Arena *scratch, NC_OptimizerAdam *opt)
{
    nc_arena_reset(scratch);
    NC_Tensor *pred = nc_graph_forward(graph, x, scratch);
    if (pred == NULL) return NAN;
    const float loss = nc_compute_mse_loss(pred, target);
    nc_autograd_record_mse(graph->tape, pred, target);
    nc_graph_backward(graph);
    nc_train_backprop_step(graph, opt);
    return loss;
}

static void test_conv_zoo(void)
{
    NC_Arena model = nc_arena_init(16 * 1024 * 1024);
    NC_Arena scratch = nc_arena_init(16 * 1024 * 1024);
    nc_rng_seed(31);
    NC_OptimizerAdam opt = { 0.01f, 0.9f, 0.999f, 1e-8f };

    NC_Graph *dcn = nc_zoo_create_dcn(&model, 1, 8, 8, 2);
    NC_Graph *dn = nc_zoo_create_dn(&model, 1, 4, 4);
    NC_Graph *dcign = nc_zoo_create_dcign(&model, 1, 8, 8, 4);
    NC_CHECK(dcn != NULL && dn != NULL && dcign != NULL);
    NC_CHECK(nc_graph_cell_count(dcn) == 4);
    NC_CHECK(nc_graph_cell_count(dn) == 3);
    NC_CHECK(nc_graph_cell_count(dcign) == 5);

    const size_t s8[4] = { 4, 1, 8, 8 };
    const size_t s4[4] = { 4, 1, 4, 4 };
    NC_Tensor *x8 = make4(&model, s8, NULL, false);
    NC_Tensor *x4 = make4(&model, s4, NULL, false);
    for (size_t i = 0; i < x8->size; ++i) x8->data[i] = (i % 4) < 2 ? 1.0f : 0.0f;
    for (size_t i = 0; i < x4->size; ++i) x4->data[i] = (float)(i % 3) / 3.0f;

    nc_arena_reset(&scratch);
    NC_Tensor *o = nc_graph_forward(dcn, x8, &scratch);
    NC_CHECK(o != NULL && o->ndim == 2 && o->shape[0] == 4 && o->shape[1] == 2);
    nc_arena_reset(&scratch);
    o = nc_graph_forward(dn, x4, &scratch);
    NC_CHECK(o != NULL && o->ndim == 4 && o->shape[2] == 8 && o->shape[3] == 8);
    nc_arena_reset(&scratch);
    o = nc_graph_forward(dcign, x8, &scratch);
    NC_CHECK(o != NULL && o->ndim == 4 && o->shape[2] == 8 && o->shape[3] == 8);

    /* Treino de reconstrução do DCIGN. */
    const size_t ty[2] = { 4, 2 };
    NC_Tensor *target = nc_tensor_create(&model, ty, 2, false);
    for (size_t i = 0; i < 4; ++i) {
        target->data[i * 2] = (i < 2) ? 1.0f : 0.0f;
        target->data[i * 2 + 1] = (i < 2) ? 0.0f : 1.0f;
    }
    float first = 0.0f, last = 0.0f;
    for (size_t epoch = 0; epoch < 400; ++epoch) {
        last = spatial_train_step(dcn, x8, target, &scratch, &opt);
        if (epoch == 0) first = last;
    }
    printf("  DCN classificação: loss %.5f -> %.5f\n", first, last);
    NC_CHECK(isfinite(last));
    NC_CHECK(last < first * 0.9f);

    NC_OptimizerAdam small_opt = { 0.002f, 0.9f, 0.999f, 1e-8f };

    /* Alvo do DN precisa ter o shape da saída (upsampling 2x). */
    for (size_t epoch = 0; epoch < 400; ++epoch) {
        last = spatial_train_step(dn, x4, x8, &scratch, &small_opt);
        if (epoch == 0) first = last;
    }
    printf("  DN reconstrução: loss %.5f -> %.5f\n", first, last);
    NC_CHECK(last < first * 0.95f);

    for (size_t epoch = 0; epoch < 400; ++epoch) {
        last = spatial_train_step(dcign, x8, x8, &scratch, &small_opt);
        if (epoch == 0) first = last;
    }
    printf("  DCIGN reconstrução: loss %.5f -> %.5f\n", first, last);
    NC_CHECK(last < first * 0.95f);

    nc_arena_free(&scratch);
    nc_arena_free(&model);
}

static void test_gan(void)
{
    NC_Arena model = nc_arena_init(16 * 1024 * 1024);
    NC_Arena scratch = nc_arena_init(16 * 1024 * 1024);
    nc_rng_seed(99);
    NC_GAN *gan = nc_zoo_create_gan(&model, 4, 3);
    NC_CHECK(gan != NULL);
    if (gan == NULL) {
        nc_arena_free(&scratch);
        nc_arena_free(&model);
        return;
    }

    const size_t rs[2] = { 8, 3 };
    NC_Tensor *real = nc_tensor_create(&model, rs, 2, false);
    for (size_t i = 0; i < real->size; ++i) real->data[i] = (float)(i % 3) / 3.0f;

    NC_Tensor *g_weight = NULL;
    for (size_t i = 0; i < gan->generator->cell_count; ++i) {
        if (gan->generator->cells[i].weights != NULL) {
            g_weight = gan->generator->cells[i].weights;
            break;
        }
    }
    NC_CHECK(g_weight != NULL);
    float before[8];
    memcpy(before, g_weight->data, 8 * sizeof(float));

    NC_OptimizerAdam opt = { 0.01f, 0.9f, 0.999f, 1e-8f };
    for (size_t step = 0; step < 200; ++step) {
        nc_arena_reset(&scratch);
        nc_train_gan_step(gan->generator, gan->discriminator, real, &scratch, &opt);
    }

    bool changed = false;
    for (size_t i = 0; i < 8; ++i) {
        if (g_weight->data[i] != before[i]) changed = true;
        NC_CHECK(isfinite(g_weight->data[i]));
    }
    NC_CHECK(changed);

    nc_arena_reset(&scratch);
    const size_t ns[2] = { 8, 4 };
    NC_Tensor *noise = nc_tensor_create(&scratch, ns, 2, false);
    nc_tensor_fill_random_normal(noise, 0.0f, 1.0f);
    NC_Tensor *fake = nc_graph_forward(gan->generator, noise, &scratch);
    NC_CHECK(fake != NULL);
    for (size_t i = 0; i < fake->size; ++i) NC_CHECK(isfinite(fake->data[i]));

    nc_arena_free(&scratch);
    nc_arena_free(&model);
}

static void test_multichannel_regressions(void)
{
    NC_Arena model = nc_arena_init(8 * 1024 * 1024);
    NC_Arena scratch = nc_arena_init(8 * 1024 * 1024);
    NC_Tape *tape = nc_autograd_tape_init(&model, 64);
    const float step = 1e-3f;

    /* A) im2col da transposta com kernel grande: kmat era subdimensionado. */
    const size_t is[4] = { 1, 2, 4, 4 };
    const size_t ks[4] = { 2, 4, 3, 3 };
    NC_Tensor *in = nc_tensor_create(&model, is, 4, false);
    NC_Tensor *k = nc_tensor_create(&model, ks, 4, false);
    for (size_t i = 0; i < in->size; ++i) in->data[i] = (float)(i % 5) - 2.0f;
    for (size_t i = 0; i < k->size; ++i) k->data[i] = (float)(i % 7) - 3.0f;
    size_t out_shape[4];
    NC_CHECK(nc_conv2d_transpose_output_shape(in, k, 3, 0, out_shape));
    NC_Tensor *direct = nc_tensor_create(&model, out_shape, 4, false);
    NC_Tensor *giemm = nc_tensor_create(&model, out_shape, 4, false);
    nc_conv2d_transpose(in, k, direct, 3, 0);
    nc_conv2d_transpose_im2col(&scratch, in, k, giemm, 3, 0);
    for (size_t i = 0; i < direct->size; ++i) {
        NC_CHECK_NEAR(giemm->data[i], direct->data[i], 1e-4f);
    }

    /* B) dInput com N>1 e C_in>1: todos os canais devem receber gradiente. */
    const size_t xs[4] = { 2, 2, 4, 4 };
    const size_t ws[4] = { 3, 2, 2, 2 };
    NC_Tensor *x = nc_tensor_create(&model, xs, 4, true);
    NC_Tensor *w = nc_tensor_create(&model, ws, 4, true);
    for (size_t i = 0; i < x->size; ++i) x->data[i] = (float)(i % 4) - 1.5f;
    for (size_t i = 0; i < w->size; ++i) w->data[i] = (float)(i % 5) - 2.0f;

    nc_arena_reset(&scratch);
    NC_Tensor *out = spatial_forward(tape, &scratch, 0, x, w);
    NC_CHECK(out != NULL);
    for (size_t i = 0; i < out->size; ++i) out->grad[i] = 1.0f;
    zeros(x);
    zeros(w);
    nc_autograd_backward(tape);
    float analytic_x[64], analytic_w[24];
    memcpy(analytic_x, x->grad, sizeof(analytic_x));
    memcpy(analytic_w, w->grad, sizeof(analytic_w));

    const size_t samples[] = { 0, 17, 33, 63 }; /* canal 0 e 1, batches 0 e 1 */
    for (size_t s = 0; s < 4; ++s) {
        const size_t idx = samples[s];
        const float original = x->data[idx];
        x->data[idx] = original + step;
        nc_arena_reset(&scratch);
        const float plus = spatial_sum(tape, &scratch, 0, x, w);
        x->data[idx] = original - step;
        nc_arena_reset(&scratch);
        const float minus = spatial_sum(tape, &scratch, 0, x, w);
        x->data[idx] = original;
        NC_CHECK_NEAR(analytic_x[idx], (plus - minus) / (2.0f * step), 2e-2f);
    }
    for (size_t idx = 0; idx < 8; idx += 3) {
        const float original = w->data[idx];
        w->data[idx] = original + step;
        nc_arena_reset(&scratch);
        const float plus = spatial_sum(tape, &scratch, 0, x, w);
        w->data[idx] = original - step;
        nc_arena_reset(&scratch);
        const float minus = spatial_sum(tape, &scratch, 0, x, w);
        w->data[idx] = original;
        NC_CHECK_NEAR(analytic_w[idx], (plus - minus) / (2.0f * step), 2e-2f);
    }

    /* C) max pooling em múltiplos canais: máscara size_t corrige o backward. */
    const size_t ps[4] = { 1, 2, 4, 4 };
    NC_Tensor *px = nc_tensor_create(&model, ps, 4, true);
    for (size_t i = 0; i < px->size; ++i) px->data[i] = (float)(i % 6);
    nc_arena_reset(&scratch);
    NC_Tensor *pool = nc_nn_max_pool2d(tape, &scratch, px, 2, 2);
    NC_CHECK(pool != NULL);
    for (size_t i = 0; i < pool->size; ++i) pool->grad[i] = 1.0f;
    zeros(px);
    nc_autograd_backward(tape);
    bool channel_one_grad = false;
    for (size_t i = 16; i < 32; ++i) {
        if (px->grad[i] != 0.0f) channel_one_grad = true;
    }
    NC_CHECK(channel_one_grad);

    /* D) contrato do flatten. */
    const size_t fs[4] = { 1, 1, 2, 2 };
    NC_Tensor *no_grad = nc_tensor_create(&model, fs, 4, false);
    NC_Tensor *with_grad = nc_tensor_create(&model, fs, 4, true);
    NC_CHECK(nc_tensor_flatten(&scratch, no_grad, true) == NULL);
    NC_Tensor *flat = nc_tensor_flatten(&scratch, with_grad, true);
    NC_CHECK(flat != NULL && flat->grad == with_grad->grad);

    nc_arena_free(&scratch);
    nc_arena_free(&model);
}

int main(void)
{
    printf("=== test_spatial (Marco 3) ===\n");
    test_im2col_matches_direct();
    test_spatial_gradients();
    test_multichannel_regressions();
    test_conv_zoo();
    test_gan();
    NC_TEST_SUMMARY("test_spatial");
}
