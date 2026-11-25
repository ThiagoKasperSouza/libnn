#include "nc_trainers.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

static float sigmoidf(float x)
{
    return 1.0f / (1.0f + (float)exp(-(double)x));
}

/* Gauss-Jordan com pivoteamento parcial: resolve A·X = B.
 * A é n×n, B é n×m, X é n×m (tudo em arrays contíguos row-major). */
static bool solve_linear(const float *A, const float *B, size_t n, size_t m,
                         float *out)
{
    if (n == 0 || m == 0 || n > 4096 || m > 4096) {
        return false;
    }
    const size_t cols = n + m;
    float *aug = (float *)malloc(n * cols * sizeof(float));
    if (aug == NULL) {
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            aug[i * cols + j] = A[i * n + j];
        }
        for (size_t j = 0; j < m; ++j) {
            aug[i * cols + n + j] = B[i * m + j];
        }
    }

    for (size_t col = 0; col < n; ++col) {
        size_t pivot = col;
        float best = fabsf(aug[col * cols + col]);
        for (size_t r = col + 1; r < n; ++r) {
            const float v = fabsf(aug[r * cols + col]);
            if (v > best) {
                best = v;
                pivot = r;
            }
        }
        if (best < 1e-10f) {
            free(aug);
            return false;
        }
        if (pivot != col) {
            for (size_t j = 0; j < cols; ++j) {
                const float tmp = aug[col * cols + j];
                aug[col * cols + j] = aug[pivot * cols + j];
                aug[pivot * cols + j] = tmp;
            }
        }
        const float inv = 1.0f / aug[col * cols + col];
        for (size_t j = 0; j < cols; ++j) aug[col * cols + j] *= inv;
        for (size_t r = 0; r < n; ++r) {
            if (r == col) continue;
            const float factor = aug[r * cols + col];
            if (factor == 0.0f) continue;
            for (size_t j = 0; j < cols; ++j) {
                aug[r * cols + j] -= factor * aug[col * cols + j];
            }
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            out[i * m + j] = aug[i * cols + n + j];
        }
    }
    free(aug);
    return true;
}

static NC_Cell *find_cell(NC_Graph *graph, NC_CellType type)
{
    if (graph == NULL) return NULL;
    for (size_t i = 0; i < graph->cell_count; ++i) {
        if (graph->cells[i].type == type) return &graph->cells[i];
    }
    return NULL;
}

static NC_Tensor *predecessor_activity(NC_Graph *graph, size_t cell_id)
{
    for (size_t i = 0; i < graph->edge_count; ++i) {
        if (graph->edges[i].dst_cell_id == cell_id) {
            return graph->cells[graph->edges[i].src_cell_id].activity;
        }
    }
    return NULL;
}

/* -------------------------- Backprop / Adam --------------------------- */

void nc_train_backprop_step(NC_Graph *graph, NC_OptimizerAdam *opt)
{
    if (graph == NULL || opt == NULL || opt->learning_rate <= 0.0f ||
        opt->beta1 <= 0.0f || opt->beta1 >= 1.0f ||
        opt->beta2 <= 0.0f || opt->beta2 >= 1.0f || opt->epsilon <= 0.0f) {
        return;
    }
    nc_graph_ensure_adam_states(graph);
    for (size_t s = 0; s < graph->adam_count; ++s) {
        NC_AdamState *state = &graph->adam[s];
        NC_Tensor *param = state->param;
        if (param == NULL || param->grad == NULL || state->m == NULL ||
            state->v == NULL) continue;
        ++state->step;
        const float b1 = powf(opt->beta1, (float)state->step);
        const float b2 = powf(opt->beta2, (float)state->step);
        const float corr1 = 1.0f - b1;
        const float corr2 = 1.0f - b2;
        if (corr1 <= 0.0f || corr2 <= 0.0f) continue;
        for (size_t i = 0; i < param->size; ++i) {
            const float g = param->grad[i];
            const float m = opt->beta1 * state->m->data[i] +
                            (1.0f - opt->beta1) * g;
            const float v = opt->beta2 * state->v->data[i] +
                            (1.0f - opt->beta2) * g * g;
            state->m->data[i] = m;
            state->v->data[i] = v;
            const float mh = m / corr1;
            const float vh = v / corr2;
            param->data[i] -= opt->learning_rate * mh /
                              (sqrtf(vh) + opt->epsilon);
        }
    }
}

static float discriminator_step(NC_Graph *discriminator, const NC_Tensor *input,
                                const NC_Tensor *target, NC_Arena *scratch,
                                NC_OptimizerAdam *opt)
{
    NC_Tensor *pred = nc_graph_forward(discriminator, input, scratch);
    if (pred == NULL) return NAN;
    const float loss = nc_compute_mse_loss(pred, target);
    nc_autograd_record_mse(discriminator->tape, pred, target);
    nc_graph_backward(discriminator);
    if (opt != NULL) nc_train_backprop_step(discriminator, opt);
    return loss;
}

static size_t first_param_dim(const NC_Graph *graph)
{
    for (size_t i = 0; i < graph->cell_count; ++i) {
        if (graph->cells[i].weights != NULL) {
            return graph->cells[i].weights->shape[0];
        }
    }
    return 0;
}

void nc_train_gan_step(NC_Graph *generator, NC_Graph *discriminator,
                       NC_Tensor *real_data, NC_Arena *scratch,
                       NC_OptimizerAdam *opt)
{
    if (generator == NULL || discriminator == NULL || real_data == NULL ||
        scratch == NULL || opt == NULL || real_data->ndim != 2) return;
    const size_t batch = real_data->shape[0];
    const size_t data_dim = real_data->shape[1];
    const size_t latent_dim = first_param_dim(generator);
    if (batch == 0 || data_dim == 0 || latent_dim == 0) return;

    const size_t noise_shape[2] = { batch, latent_dim };
    const size_t target_shape[2] = { batch, 1 };
    NC_Tensor *noise = nc_tensor_create(scratch, noise_shape, 2, false);
    NC_Tensor *ones = nc_tensor_create(scratch, target_shape, 2, false);
    NC_Tensor *zeros = nc_tensor_create(scratch, target_shape, 2, false);
    NC_Tensor *fake_copy = nc_tensor_create(scratch, real_data->shape, 2, false);
    if (noise == NULL || ones == NULL || zeros == NULL || fake_copy == NULL) return;
    nc_tensor_fill_random_normal(noise, 0.0f, 1.0f);
    for (size_t i = 0; i < ones->size; ++i) ones->data[i] = 1.0f;

    discriminator_step(discriminator, real_data, ones, scratch, opt);
    NC_Tensor *fake = nc_graph_forward(generator, noise, scratch);
    if (fake == NULL || fake->size != fake_copy->size) return;
    memcpy(fake_copy->data, fake->data, fake->size * sizeof(float));
    discriminator_step(discriminator, fake_copy, zeros, scratch, opt);

    NC_Tensor *fake_for_g = nc_graph_forward(generator, noise, scratch);
    if (fake_for_g == NULL) return;
    NC_Tensor *verdict = nc_graph_forward(discriminator, fake_for_g, scratch);
    if (verdict == NULL) return;
    nc_autograd_record_mse(discriminator->tape, verdict, ones);
    nc_graph_backward(discriminator);
    nc_graph_backward_no_zero(generator);
    nc_train_backprop_step(generator, opt);
}

/* ---------------------- Divergência contrastiva ----------------------- */

void nc_train_contrastive_divergence(NC_Graph *graph, NC_Tensor *input,
                                     size_t k_steps, float lr)
{
    if (graph == NULL || input == NULL || input->ndim != 2 || lr <= 0.0f) return;
    NC_Cell *cell = find_cell(graph, NC_CELL_PROBABILISTIC_HIDDEN);
    if (cell == NULL || cell->weights == NULL || cell->weights->ndim != 2) return;

    const size_t batch = input->shape[0];
    const size_t vis = input->shape[1];
    const size_t hid = cell->weights->shape[1];
    NC_Tensor *W = cell->weights;
    NC_Tensor *hb = cell->bias;             // [1, hid]
    NC_Tensor *vb = cell->recurrent_weights; // [1, vis], pode ser NULL

    const size_t gw_size = vis * hid;
    float *grad_w = (float *)calloc(gw_size, sizeof(float));
    float *grad_h = (float *)calloc(hid, sizeof(float));
    float *grad_v = (float *)calloc(vis, sizeof(float));
    float *v_pos = (float *)malloc(vis * sizeof(float));
    float *v_neg = (float *)malloc(vis * sizeof(float));
    float *h_prob = (float *)malloc(hid * sizeof(float));
    float *h_bin = (float *)malloc(hid * sizeof(float));
    if (!grad_w || !grad_h || !grad_v || !v_pos || !v_neg || !h_prob || !h_bin) {
        free(grad_w); free(grad_h); free(grad_v);
        free(v_pos); free(v_neg); free(h_prob); free(h_bin);
        return;
    }

    for (size_t n = 0; n < batch; ++n) {
        memcpy(v_pos, input->data + n * vis, vis * sizeof(float));
        for (size_t step = 0; step <= k_steps; ++step) {
            /* Fase positiva: p(h|v_pos) e amostra binária. */
            for (size_t h = 0; h < hid; ++h) {
                float sum = hb ? hb->data[h] : 0.0f;
                for (size_t v = 0; v < vis; ++v) {
                    sum += v_pos[v] * W->data[v * hid + h];
                }
                h_prob[h] = sigmoidf(sum);
                h_bin[h] = nc_rng_next_double() < h_prob[h] ? 1.0f : 0.0f;
            }
            /* Fase negativa: v' amostrado de p(v|h), depois p(h|v'). */
            for (size_t v = 0; v < vis; ++v) {
                float sum = vb ? vb->data[v] : 0.0f;
                for (size_t h = 0; h < hid; ++h) {
                    sum += h_bin[h] * W->data[v * hid + h];
                }
                v_neg[v] = nc_rng_next_double() < sigmoidf(sum) ? 1.0f : 0.0f;
            }
            for (size_t h = 0; h < hid; ++h) {
                float sum = hb ? hb->data[h] : 0.0f;
                for (size_t v = 0; v < vis; ++v) {
                    sum += v_neg[v] * W->data[v * hid + h];
                }
                h_prob[h] = sigmoidf(sum); // sem amostrar na última fase
            }
            for (size_t v = 0; v < vis; ++v) {
                for (size_t h = 0; h < hid; ++h) {
                    grad_w[v * hid + h] +=
                        v_pos[v] * h_bin[h] - v_neg[v] * h_prob[h];
                }
                grad_v[v] += v_pos[v] - v_neg[v];
            }
            for (size_t h = 0; h < hid; ++h) {
                grad_h[h] += h_bin[h] - h_prob[h];
            }
            memcpy(v_pos, v_neg, vis * sizeof(float));
        }
    }

    const float scale = lr / (float)batch;
    for (size_t i = 0; i < gw_size; ++i) W->data[i] += scale * grad_w[i];
    if (hb) for (size_t i = 0; i < hid; ++i) hb->data[i] += scale * grad_h[i];
    if (vb) for (size_t i = 0; i < vis; ++i) vb->data[i] += scale * grad_v[i];

    free(grad_w); free(grad_h); free(grad_v);
    free(v_pos); free(v_neg); free(h_prob); free(h_bin);
}

/* ------------------------- Hebbiano e STDP ---------------------------- */

void nc_train_hebbian_step(NC_Graph *graph, float lr)
{
    if (graph == NULL || lr <= 0.0f) return;
    for (size_t i = 0; i < graph->cell_count; ++i) {
        NC_Cell *cell = &graph->cells[i];
        if (cell->weights == NULL || !cell->weights->requires_grad ||
            cell->activity == NULL) continue;
        NC_Tensor *pre = predecessor_activity(graph, i);
        if (pre == NULL && cell->type == NC_CELL_BACKFED_INPUT) {
            pre = graph->last_input; /* padrão apresentado ao Hopfield */
        }
        if (pre == NULL) pre = cell->activity; /* Hebbiano autorecorrente */
        NC_Tensor *post = cell->activity;
        if (cell->type == NC_CELL_BACKFED_INPUT && graph->last_input != NULL) {
            post = graph->last_input; /* Hopfield: pós também é o padrão */
        }
        if (pre == NULL || pre->ndim != 2 || post->ndim != 2) continue;
        const size_t in_dim = pre->shape[1];
        const size_t out_dim = post->shape[1];
        if (cell->weights->shape[0] != in_dim ||
            cell->weights->shape[1] != out_dim) continue;
        for (size_t n = 0; n < pre->shape[0]; ++n) {
            for (size_t a = 0; a < in_dim; ++a) {
                for (size_t b = 0; b < out_dim; ++b) {
                    cell->weights->data[a * out_dim + b] +=
                        lr * pre->data[n * pre->strides[0] + a] *
                        post->data[n * post->strides[0] + b];
                }
            }
        }
    }
}

void nc_train_stdp_step(NC_Graph *graph, float lr, float tau)
{
    if (graph == NULL || lr <= 0.0f || tau <= 0.0f) return;
    for (size_t i = 0; i < graph->cell_count; ++i) {
        NC_Cell *cell = &graph->cells[i];
        if (cell->type != NC_CELL_SPIKING_HIDDEN || cell->weights == NULL ||
            cell->activity == NULL) continue;
        NC_Tensor *pre = predecessor_activity(graph, i);
        if (pre == NULL || pre->ndim != 2 || cell->activity->ndim != 2) continue;
        const size_t in_dim = pre->shape[1];
        const size_t out_dim = cell->activity->shape[1];
        const float dt = cell->state.spiking.last_post_time -
                         cell->state.spiking.last_pre_time;
        const float sign = dt >= 0.0f ? 1.0f : -1.0f;
        const float magnitude = expf(-fabsf(dt) / tau);
        for (size_t n = 0; n < pre->shape[0]; ++n) {
            for (size_t a = 0; a < in_dim && a < cell->weights->shape[0]; ++a) {
                for (size_t b = 0; b < out_dim && b < cell->weights->shape[1]; ++b) {
                    const float pre_spike =
                        pre->data[n * pre->strides[0] + a];
                    const float post_spike =
                        cell->activity->data[n * cell->activity->strides[0] + b];
                    if (pre_spike <= 0.0f || post_spike <= 0.0f) continue;
                    cell->weights->data[a * cell->weights->strides[0] +
                                        b * cell->weights->strides[1]] +=
                        lr * sign * magnitude;
                }
            }
        }
    }
}

/* ---------------------- Ridge / pseudoinversa ------------------------- */

bool nc_train_reservoir_ridge_regression(NC_Graph *graph, NC_Tensor *states,
                                         NC_Tensor *targets, float alpha)
{
    if (graph == NULL || states == NULL || targets == NULL || states->ndim != 2 ||
        targets->ndim != 2 || states->shape[0] != targets->shape[0] || alpha < 0.0f) {
        return false;
    }
    NC_Cell *out_cell = find_cell(graph, NC_CELL_OUTPUT);
    if (out_cell == NULL || out_cell->weights == NULL ||
        out_cell->weights->shape[0] != states->shape[1] ||
        out_cell->weights->shape[1] != targets->shape[1]) {
        return false;
    }
    const size_t n = states->shape[1];
    const size_t m = targets->shape[1];
    float *A = (float *)calloc(n * n, sizeof(float));
    float *B = (float *)calloc(n * m, sizeof(float));
    float *X = (float *)calloc(n * m, sizeof(float));
    if (A == NULL || B == NULL || X == NULL) {
        free(A); free(B); free(X);
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t s = 0; s < states->shape[0]; ++s) {
                sum += (double)states->data[s * states->strides[0] + i] *
                       states->data[s * states->strides[0] + j];
            }
            A[i * n + j] = (float)(sum + (i == j ? alpha : 0.0));
        }
        for (size_t j = 0; j < m; ++j) {
            double sum = 0.0;
            for (size_t s = 0; s < states->shape[0]; ++s) {
                sum += (double)states->data[s * states->strides[0] + i] *
                       targets->data[s * targets->strides[0] + j];
            }
            B[i * m + j] = (float)sum;
        }
    }
    const bool ok = solve_linear(A, B, n, m, X);
    if (ok) {
        memcpy(out_cell->weights->data, X, n * m * sizeof(float));
    }
    free(A); free(B); free(X);
    return ok;
}

/* --------------------------- Kohonen BMU ------------------------------ */

void nc_train_kohonen_bmu_step(NC_Graph *graph, NC_Tensor *input_sample,
                               float lr, float radius)
{
    if (graph == NULL || input_sample == NULL || input_sample->ndim != 2 ||
        input_sample->shape[0] != 1 || lr <= 0.0f || radius <= 0.0f) return;
    NC_Cell *cell = find_cell(graph, NC_CELL_HIDDEN);
    if (cell == NULL || cell->weights == NULL ||
        cell->weights->ndim != 2) return;
    const size_t neurons = cell->weights->shape[0];
    const size_t dim = cell->weights->shape[1];
    if (dim != input_sample->shape[1]) return;

    size_t bmu = 0;
    float best = INFINITY;
    for (size_t i = 0; i < neurons; ++i) {
        float dist = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            const float d = cell->weights->data[i * dim + j] -
                            input_sample->data[j];
            dist += d * d;
        }
        if (dist < best) { best = dist; bmu = i; }
    }
    for (size_t i = 0; i < neurons; ++i) {
        const float d = (float)i - (float)bmu;
        const float neighborhood = expf(-(d * d) / (2.0f * radius * radius));
        const float factor = lr * neighborhood;
        for (size_t j = 0; j < dim; ++j) {
            cell->weights->data[i * dim + j] +=
                factor * (input_sample->data[j] - cell->weights->data[i * dim + j]);
        }
    }
}

/* ---------------------------- SVM / hinge ------------------------------ */

void nc_train_kernel_hinge_loss(NC_Graph *graph, NC_Tensor *X,
                                NC_Tensor *y, float C)
{
    if (graph == NULL || X == NULL || y == NULL || X->ndim != 2 ||
        y->ndim != 2 || X->shape[0] != y->shape[0] || C <= 0.0f) return;
    NC_Cell *cell = find_cell(graph, NC_CELL_OUTPUT);
    if (cell == NULL || cell->weights == NULL ||
        cell->weights->shape[0] != X->shape[1]) return;
    NC_Tensor *w = cell->weights;
    const float lr = 0.01f;
    const size_t features = X->shape[1];
    const size_t outputs = cell->weights->shape[1];
    if (outputs != 1) return;

    for (size_t s = 0; s < X->shape[0]; ++s) {
        const float label = y->data[s] >= 0.0f ? 1.0f : -1.0f;
        float margin = 0.0f;
        for (size_t j = 0; j < features; ++j) {
            margin += w->data[j] * X->data[s * X->strides[0] + j];
        }
        if (label * margin >= 1.0f) continue;
        for (size_t j = 0; j < features; ++j) {
            w->data[j] += lr * C * label * X->data[s * X->strides[0] + j];
        }
    }
}

