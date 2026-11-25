#include "test_common.h"
#include "neurochart.h"

#include <string.h>

static NC_Tensor *matrix(NC_Arena *arena, size_t r, size_t c, const float *v)
{
    const size_t shape[2] = { r, c };
    NC_Tensor *t = nc_tensor_create(arena, shape, 2, false);
    if (t != NULL && v != NULL) memcpy(t->data, v, t->size * sizeof(float));
    return t;
}

static float l1(const NC_Tensor *t)
{
    float sum = 0.0f;
    if (t == NULL) return 0.0f;
    for (size_t i = 0; i < t->size; ++i) sum += fabsf(t->data[i]);
    return sum;
}

static void test_constructors(void)
{
    NC_Arena model = nc_arena_init(32 * 1024 * 1024);
    const size_t dims[2] = { 4, 4 };
    NC_Graph *nets[12] = {
        nc_zoo_create_markov_chain(&model, 4, 3),
        nc_zoo_create_hopfield(&model, 4),
        nc_zoo_create_boltzmann(&model, 4, 3),
        nc_zoo_create_rbm(&model, 4, 3),
        nc_zoo_create_dbn(&model, 4, dims, 2),
        nc_zoo_create_vae(&model, 4, 2),
        nc_zoo_create_lsm(&model, 4, 6, 1),
        nc_zoo_create_elm(&model, 4, 6, 1),
        nc_zoo_create_esn(&model, 4, 6, 1),
        nc_zoo_create_kohonen(&model, 4, 5),
        nc_zoo_create_rbf(&model, 4, 3, 1),
        nc_zoo_create_svm(&model, 4)
    };
    for (size_t i = 0; i < 12; ++i) NC_CHECK(nets[i] != NULL);
    nc_arena_free(&model);
}

static void test_cd_and_hebbian(void)
{
    NC_Arena model = nc_arena_init(4 * 1024 * 1024);
    NC_Arena scratch = nc_arena_init(4 * 1024 * 1024);
    nc_rng_seed(17);

    /* RBM: CD-k deve alterar os pesos. */
    NC_Graph *rbm = nc_zoo_create_rbm(&model, 4, 3);
    NC_CHECK(rbm != NULL);
    float input[] = { 1, 0, 1, 0, 0, 1, 0, 1 };
    NC_Tensor *x = matrix(&model, 2, 4, input);
    NC_Tensor *w = rbm->cells[1].weights;
    float before[12];
    memcpy(before, w->data, sizeof(before));
    for (size_t step = 0; step < 50; ++step) {
        nc_train_contrastive_divergence(rbm, x, 1, 0.1f);
    }
    float changed = 0.0f;
    for (size_t i = 0; i < 12; ++i) changed += fabsf(w->data[i] - before[i]);
    printf("  RBM CD-k: ΔW L1 = %.4f\n", changed);
    NC_CHECK(changed > 1e-3f);

    /* Hopfield: Hebbiano grava o padrão; forward com padrão ruidoso recupera. */
    NC_Graph *hop = nc_zoo_create_hopfield(&model, 4);
    NC_CHECK(hop != NULL);
    const float pattern[] = { 1, 1, -1, -1 };
    NC_Tensor *p = matrix(&model, 1, 4, pattern);
    nc_arena_reset(&scratch);
    NC_Tensor *act = nc_graph_forward(hop, p, &scratch);
    NC_CHECK(act != NULL);
    for (size_t rep = 0; rep < 10; ++rep) {
        nc_train_hebbian_step(hop, 0.5f);
    }
    NC_CHECK(l1(hop->cells[0].weights) > 0.0f);
    const float noisy[] = { 0.8f, 0.9f, -0.7f, -0.6f };
    NC_Tensor *pn = matrix(&model, 1, 4, noisy);
    nc_arena_reset(&scratch);
    NC_Tensor *recalled = nc_graph_forward(hop, pn, &scratch);
    printf("  Hopfield recall:");
    for (size_t i = 0; i < 4; ++i) printf(" %+.3f", recalled ? recalled->data[i] : 0.0f);
    printf("\n");
    NC_CHECK(recalled != NULL);
    for (size_t i = 0; i < 4; ++i) {
        NC_CHECK(recalled->data[i] * pattern[i] > 0.0f);
    }

    nc_arena_free(&scratch);
    nc_arena_free(&model);
}

static void test_spiking_stdp(void)
{
    NC_Arena model = nc_arena_init(4 * 1024 * 1024);
    NC_Arena scratch = nc_arena_init(4 * 1024 * 1024);
    nc_rng_seed(3);
    NC_Graph *lsm = nc_zoo_create_lsm(&model, 3, 4, 1);
    NC_CHECK(lsm != NULL);
    const float spikes[] = { 1, 1, 1, 1, 1, 1 };
    NC_Tensor *x = matrix(&model, 2, 3, spikes);
    nc_arena_reset(&scratch);
    NC_Tensor *out = nc_graph_forward(lsm, x, &scratch);
    NC_CHECK(out != NULL);
    NC_CHECK(l1(lsm->cells[1].activity) > 0.0f); /* spikes na camada LIF */
    NC_Tensor *w = lsm->cells[1].weights;
    float before[12];
    memcpy(before, w->data, w->size * sizeof(float));
    nc_train_stdp_step(lsm, 0.1f, 10.0f);
    float before_l1 = 0.0f;
    for (size_t i = 0; i < 12; ++i) before_l1 += fabsf(before[i]);
    printf("  STDP L1 antes=%.6f depois=%.6f\n", before_l1, l1(w));
    NC_CHECK(memcmp(before, w->data, w->size * sizeof(float)) != 0);

    nc_arena_free(&scratch);
    nc_arena_free(&model);
}

static void test_ridge_and_kohonen(void)
{
    NC_Arena model = nc_arena_init(8 * 1024 * 1024);
    NC_Arena scratch = nc_arena_init(8 * 1024 * 1024);
    nc_rng_seed(9);

    /* ELM: ridge resolve a saída para funções das entradas. */
    NC_Graph *elm = nc_zoo_create_elm(&model, 2, 10, 1);
    NC_CHECK(elm != NULL);
    const float xv[] = { 0,0, 0,1, 1,0, 1,1 };
    const float tv[] = { 0, 1, 1, 2 }; /* y = x0 + x1 */
    NC_Tensor *x = matrix(&model, 4, 2, xv);
    NC_Tensor *y = matrix(&model, 4, 1, tv);
    nc_arena_reset(&scratch);
    NC_Tensor *features = NULL;
    NC_CHECK(nc_graph_forward(elm, x, &scratch) != NULL);
    features = elm->cells[1].activity;
    NC_CHECK(features != NULL);
    NC_CHECK(nc_train_reservoir_ridge_regression(elm, features, y, 1e-3f));
    nc_arena_reset(&scratch);
    NC_Tensor *pred = nc_graph_forward(elm, x, &scratch);
    const float loss = nc_compute_mse_loss(pred, y);
    printf("  ELM ridge: loss = %.6f\n", loss);
    NC_CHECK(loss < 1e-3f);

    /* Kohonen: BMU move os protótipos para perto das amostras. */
    NC_Graph *som = nc_zoo_create_kohonen(&model, 2, 4);
    NC_CHECK(som != NULL);
    NC_Tensor *proto = som->cells[1].weights;
    const float before = l1(proto);
    for (size_t rep = 0; rep < 50; ++rep) {
        const float v[] = { 1.0f, 0.0f };
        NC_Tensor *sample = matrix(&scratch, 1, 2, v);
        nc_train_kohonen_bmu_step(som, sample, 0.2f, 1.5f);
    }
    NC_CHECK(l1(proto) != before);
    float min_dist = INFINITY;
    for (size_t i = 0; i < 4; ++i) {
        const float d0 = proto->data[i * 2] - 1.0f;
        const float d1 = proto->data[i * 2 + 1];
        const float d = d0 * d0 + d1 * d1;
        if (d < min_dist) min_dist = d;
    }
    NC_CHECK(min_dist < 0.1f);

    nc_arena_free(&scratch);
    nc_arena_free(&model);
}

static void test_kernel_and_svm(void)
{
    NC_Arena model = nc_arena_init(8 * 1024 * 1024);
    NC_Arena scratch = nc_arena_init(8 * 1024 * 1024);
    nc_rng_seed(21);

    /* RBF + ridge: mesma função linear. */
    NC_Graph *rbf = nc_zoo_create_rbf(&model, 2, 10, 1);
    NC_CHECK(rbf != NULL);
    const float xv[] = { 0,0, 0,1, 1,0, 1,1 };
    const float tv[] = { 0, 1, 1, 2 };
    NC_Tensor *x = matrix(&model, 4, 2, xv);
    NC_Tensor *y = matrix(&model, 4, 1, tv);
    nc_arena_reset(&scratch);
    NC_CHECK(nc_graph_forward(rbf, x, &scratch) != NULL);
    NC_CHECK(nc_train_reservoir_ridge_regression(rbf, rbf->cells[1].activity,
                                                 y, 1e-6f));
    nc_arena_reset(&scratch);
    printf("  RBF ridge: loss = %.6f\n",
           nc_compute_mse_loss(nc_graph_forward(rbf, x, &scratch), y));
    NC_CHECK(nc_compute_mse_loss(nc_graph_forward(rbf, x, &scratch), y) < 0.05f);

    /* SVM linear com hinge: separa duas classes. */
    NC_Graph *svm = nc_zoo_create_svm(&model, 2);
    NC_CHECK(svm != NULL);
    const float sx[] = { 0.2f, 0.3f, 0.8f, 0.9f };
    const float sy[] = { -1, 1 };
    NC_Tensor *xs = matrix(&model, 2, 2, sx);
    NC_Tensor *ys = matrix(&model, 2, 1, sy);
    for (size_t step = 0; step < 200; ++step) {
        nc_train_kernel_hinge_loss(svm, xs, ys, 1.0f);
    }
    const float *w = svm->cells[1].weights->data;
    printf("  SVM hinge: w = [%.3f %.3f]\n", w[0], w[1]);
    NC_CHECK(w[0] * 0.8f + w[1] * 0.9f > w[0] * 0.2f + w[1] * 0.3f);

    nc_arena_free(&scratch);
    nc_arena_free(&model);
}

int main(void)
{
    printf("=== test_energy (Marco 5) ===\n");
    test_constructors();
    test_cd_and_hebbian();
    test_spiking_stdp();
    test_ridge_and_kohonen();
    test_kernel_and_svm();
    NC_TEST_SUMMARY("test_energy");
}
