#include "nc_cells.h"

#include <math.h>
#include <stdalign.h>
#include <string.h>

NC_Cell *nc_cell_create(NC_Arena *arena, size_t id, NC_CellType type,
                        NC_Activation activation)
{
    if (arena == NULL) {
        return NULL;
    }
    NC_Cell *cell = (NC_Cell *)nc_arena_alloc(arena, sizeof(*cell),
                                              alignof(NC_Cell));
    if (cell == NULL) {
        return NULL;
    }
    memset(cell, 0, sizeof(*cell));
    cell->id = id;
    cell->type = type;
    cell->activation = activation;
    return cell;
}

bool nc_cell_init_params(NC_Cell *cell, NC_Arena *arena,
                         size_t in_dim, size_t out_dim, bool with_bias)
{
    if (cell == NULL || arena == NULL || in_dim == 0 || out_dim == 0) {
        return false;
    }

    const size_t w_shape[2] = { in_dim, out_dim };
    NC_Tensor *w = nc_tensor_create(arena, w_shape, 2, true);
    if (w == NULL) {
        return false;
    }
    const float stddev = sqrtf(2.0f / (float)(in_dim + out_dim));
    nc_tensor_fill_random_normal(w, 0.0f, stddev);
    cell->weights = w;

    if (with_bias) {
        const size_t b_shape[2] = { 1, out_dim };
        cell->bias = nc_tensor_create(arena, b_shape, 2, true);
        if (cell->bias == NULL) {
            cell->weights = NULL;
            return false;
        }
        nc_tensor_fill_zeros(cell->bias);
    }
    return true;
}

bool nc_cell_init_conv_params(NC_Cell *cell, NC_Arena *arena,
                              size_t c_out, size_t c_in,
                              size_t k_h, size_t k_w, bool with_bias)
{
    if (cell == NULL || arena == NULL || c_out == 0 || c_in == 0 ||
        k_h == 0 || k_w == 0) {
        return false;
    }
    const size_t shape[4] = { c_out, c_in, k_h, k_w };
    NC_Tensor *w = nc_tensor_create(arena, shape, 4, true);
    if (w == NULL) {
        return false;
    }
    const float fan_in = (float)(c_in * k_h * k_w);
    const float fan_out = (float)(c_out * k_h * k_w);
    nc_tensor_fill_random_normal(w, 0.0f, sqrtf(2.0f / (fan_in + fan_out)));
    cell->weights = w;

    if (with_bias) {
        const size_t b_shape[2] = { 1, c_out };
        cell->bias = nc_tensor_create(arena, b_shape, 2, true);
        if (cell->bias == NULL) {
            cell->weights = NULL;
            return false;
        }
        nc_tensor_fill_zeros(cell->bias);
    }
    return true;
}

static NC_OpType op_from_activation(NC_Activation activation)
{
    switch (activation) {
    case NC_ACT_RELU:    return NC_OP_RELU;
    case NC_ACT_TANH:    return NC_OP_TANH;
    case NC_ACT_SIGMOID: return NC_OP_SIGMOID;
    case NC_ACT_GELU:    return NC_OP_GELU;
    default:             return NC_OP_LINEAR;
    }
}

static NC_Tensor *affine_forward(NC_Cell *cell, const NC_Tensor *input,
                                 NC_Tape *tape, NC_Arena *scratch)
{
    NC_Tensor *z = nc_nn_matmul(tape, scratch, input, cell->weights);
    if (z == NULL) {
        return NULL;
    }
    if (cell->bias != NULL) {
        z = nc_nn_add(tape, scratch, z, cell->bias);
        if (z == NULL) {
            return NULL;
        }
    }
    if (cell->activation == NC_ACT_LINEAR) {
        return z;
    }
    return nc_nn_activation(tape, scratch, z, op_from_activation(cell->activation));
}

static NC_Tensor *noisy_input_forward(NC_Cell *cell, const NC_Tensor *input,
                                      NC_Tape *tape, NC_Arena *scratch)
{
    if (cell->state.noisy.dropout_rate > 0.0f) {
        return nc_nn_dropout(tape, scratch, input,
                             1.0f - cell->state.noisy.dropout_rate);
    }
    if (cell->state.noisy.noise_std <= 0.0f) {
        return (NC_Tensor *)input;
    }

    const size_t shape[2] = { input->shape[0], input->shape[1] };
    NC_Tensor *noisy = nc_tensor_create(scratch, shape, 2, false);
    if (noisy == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < input->size; ++i) {
        noisy->data[i] = (float)(input->data[i] +
                         nc_rng_next_normal() * cell->state.noisy.noise_std);
    }
    /* Gradiente identidade: o ruído não é treinável. */
    return nc_nn_activation(tape, scratch, noisy, NC_OP_LINEAR);
}

bool nc_cell_init_gated_params(NC_Cell *cell, NC_Arena *arena,
                               size_t in_dim, size_t hidden_dim, size_t gates)
{
    if (cell == NULL || arena == NULL || in_dim == 0 || hidden_dim == 0 ||
        gates == 0 || gates > 4) {
        return false;
    }
    const float stddev = sqrtf(2.0f / (float)(in_dim + hidden_dim));
    for (size_t g = 0; g < gates; ++g) {
        const size_t ws[2] = { in_dim, hidden_dim };
        const size_t rs[2] = { hidden_dim, hidden_dim };
        const size_t bs[2] = { 1, hidden_dim };
        cell->gate_weights[g] = nc_tensor_create(arena, ws, 2, true);
        cell->gate_recurrent[g] = nc_tensor_create(arena, rs, 2, true);
        cell->gate_bias[g] = nc_tensor_create(arena, bs, 2, true);
        if (cell->gate_weights[g] == NULL || cell->gate_recurrent[g] == NULL ||
            cell->gate_bias[g] == NULL) {
            return false;
        }
        nc_tensor_fill_random_normal(cell->gate_weights[g], 0.0f, stddev);
        nc_tensor_fill_random_normal(cell->gate_recurrent[g], 0.0f, stddev);
        nc_tensor_fill_zeros(cell->gate_bias[g]);
    }
    return true;
}

void nc_cell_reset_state(NC_Cell *cell)
{
    if (cell == NULL) {
        return;
    }
    memset(&cell->state, 0, sizeof(cell->state));
}

static void zero_state_tensor_grad(NC_Tensor *t)
{
    if (t != NULL && t->grad != NULL) {
        memset(t->grad, 0, t->size * sizeof(float));
    }
}

void nc_cell_zero_state_grads(NC_Cell *cell)
{
    if (cell == NULL) {
        return;
    }
    /* A union só contém ponteiros válidos para o tipo da célula. */
    switch (cell->type) {
    case NC_CELL_RECURRENT:
        zero_state_tensor_grad(cell->state.recurrent.hidden);
        break;
    case NC_CELL_MEMORY:
        zero_state_tensor_grad(cell->state.lstm.hidden);
        zero_state_tensor_grad(cell->state.lstm.cell_state);
        zero_state_tensor_grad(cell->state.lstm.gate_input);
        zero_state_tensor_grad(cell->state.lstm.gate_forget);
        zero_state_tensor_grad(cell->state.lstm.gate_output);
        break;
    case NC_CELL_DIFFERENT_MEMORY:
        if (cell->gate_weights[2] != NULL) { /* GRU */
            zero_state_tensor_grad(cell->state.gru.hidden);
            zero_state_tensor_grad(cell->state.gru.gate_z);
            zero_state_tensor_grad(cell->state.gru.gate_r);
            zero_state_tensor_grad(cell->state.gru.candidate);
        } else { /* NTM */
            zero_state_tensor_grad(cell->state.ntm.memory_bank);
            zero_state_tensor_grad(cell->state.ntm.read_head);
            zero_state_tensor_grad(cell->state.ntm.write_head);
        }
        break;
    default:
        break;
    }
}

static NC_Tensor *ensure_state(NC_Arena *scratch, NC_Tensor **slot,
                               size_t rows, size_t cols)
{
    if (*slot == NULL) {
        const size_t shape[2] = { rows, cols };
        *slot = nc_tensor_create(scratch, shape, 2, true);
        if (*slot != NULL) {
            nc_tensor_fill_zeros(*slot);
        }
    }
    return *slot;
}

static NC_Tensor *linear_gate(NC_Tape *tape, NC_Arena *scratch,
                              const NC_Tensor *x, const NC_Tensor *h,
                              NC_Tensor *wx, NC_Tensor *wh, NC_Tensor *bias,
                              NC_OpType activation)
{
    NC_Tensor *z = nc_nn_matmul(tape, scratch, x, wx);
    NC_Tensor *zh = nc_nn_matmul(tape, scratch, h, wh);
    z = nc_nn_add(tape, scratch, z, zh);
    z = nc_nn_add(tape, scratch, z, bias);
    return nc_nn_activation(tape, scratch, z, activation);
}

static NC_Tensor *rnn_forward(NC_Cell *cell, const NC_Tensor *input,
                              NC_Tape *tape, NC_Arena *scratch)
{
    if (cell->weights == NULL || cell->recurrent_weights == NULL ||
        cell->bias == NULL) {
        return NULL;
    }
    const size_t hidden = cell->weights->shape[1];
    NC_Tensor *h_prev = ensure_state(scratch, &cell->state.recurrent.hidden,
                                     input->shape[0], hidden);
    if (h_prev == NULL) {
        return NULL;
    }
    NC_Tensor *h = linear_gate(tape, scratch, input, h_prev,
                               cell->weights, cell->recurrent_weights,
                               cell->bias, NC_OP_TANH);
    cell->state.recurrent.hidden = h;
    return h;
}

static NC_Tensor *lstm_forward(NC_Cell *cell, const NC_Tensor *input,
                               NC_Tape *tape, NC_Arena *scratch)
{
    for (size_t g = 0; g < 4; ++g) {
        if (cell->gate_weights[g] == NULL ||
            cell->gate_recurrent[g] == NULL || cell->gate_bias[g] == NULL) {
            return NULL;
        }
    }
    const size_t hidden = cell->gate_weights[0]->shape[1];
    NC_Tensor *h_prev = ensure_state(scratch, &cell->state.lstm.hidden,
                                     input->shape[0], hidden);
    NC_Tensor *c_prev = ensure_state(scratch, &cell->state.lstm.cell_state,
                                     input->shape[0], hidden);
    if (h_prev == NULL || c_prev == NULL) {
        return NULL;
    }

    NC_Tensor *i = linear_gate(tape, scratch, input, h_prev,
                               cell->gate_weights[0], cell->gate_recurrent[0],
                               cell->gate_bias[0], NC_OP_SIGMOID);
    NC_Tensor *f = linear_gate(tape, scratch, input, h_prev,
                               cell->gate_weights[1], cell->gate_recurrent[1],
                               cell->gate_bias[1], NC_OP_SIGMOID);
    NC_Tensor *o = linear_gate(tape, scratch, input, h_prev,
                               cell->gate_weights[2], cell->gate_recurrent[2],
                               cell->gate_bias[2], NC_OP_SIGMOID);
    NC_Tensor *cand = linear_gate(tape, scratch, input, h_prev,
                                  cell->gate_weights[3], cell->gate_recurrent[3],
                                  cell->gate_bias[3], NC_OP_TANH);

    NC_Tensor *fc = nc_nn_mul(tape, scratch, f, c_prev);
    NC_Tensor *ic = nc_nn_mul(tape, scratch, i, cand);
    NC_Tensor *c = nc_nn_add(tape, scratch, fc, ic);
    NC_Tensor *tc = nc_nn_activation(tape, scratch, c, NC_OP_TANH);
    NC_Tensor *h = nc_nn_mul(tape, scratch, o, tc);

    cell->state.lstm.hidden = h;
    cell->state.lstm.cell_state = c;
    cell->state.lstm.gate_input = i;
    cell->state.lstm.gate_forget = f;
    cell->state.lstm.gate_output = o;
    return h;
}

static NC_Tensor *gru_forward(NC_Cell *cell, const NC_Tensor *input,
                              NC_Tape *tape, NC_Arena *scratch)
{
    for (size_t g = 0; g < 3; ++g) {
        if (cell->gate_weights[g] == NULL ||
            cell->gate_recurrent[g] == NULL || cell->gate_bias[g] == NULL) {
            return NULL;
        }
    }
    const size_t hidden = cell->gate_weights[0]->shape[1];
    NC_Tensor *h_prev = ensure_state(scratch, &cell->state.gru.hidden,
                                     input->shape[0], hidden);
    if (h_prev == NULL) {
        return NULL;
    }
    NC_Tensor *z = linear_gate(tape, scratch, input, h_prev,
                               cell->gate_weights[0], cell->gate_recurrent[0],
                               cell->gate_bias[0], NC_OP_SIGMOID);
    NC_Tensor *r = linear_gate(tape, scratch, input, h_prev,
                               cell->gate_weights[1], cell->gate_recurrent[1],
                               cell->gate_bias[1], NC_OP_SIGMOID);
    NC_Tensor *hh = nc_nn_matmul(tape, scratch, h_prev, cell->gate_recurrent[2]);
    NC_Tensor *rh = nc_nn_mul(tape, scratch, r, hh);
    NC_Tensor *xc = nc_nn_matmul(tape, scratch, input, cell->gate_weights[2]);
    NC_Tensor *cpre = nc_nn_add(tape, scratch, xc, rh);
    cpre = nc_nn_add(tape, scratch, cpre, cell->gate_bias[2]);
    NC_Tensor *cand = nc_nn_activation(tape, scratch, cpre, NC_OP_TANH);

    NC_Tensor *neg_h = nc_nn_scale(tape, scratch, h_prev, -1.0f);
    NC_Tensor *diff = nc_nn_add(tape, scratch, cand, neg_h);
    NC_Tensor *update = nc_nn_mul(tape, scratch, z, diff);
    NC_Tensor *h = nc_nn_add(tape, scratch, h_prev, update);

    cell->state.gru.hidden = h;
    cell->state.gru.gate_z = z;
    cell->state.gru.gate_r = r;
    cell->state.gru.candidate = cand;
    return h;
}

static NC_Tensor *ntm_forward(NC_Cell *cell, const NC_Tensor *input,
                              NC_Tape *tape, NC_Arena *scratch)
{
    /* NTM com endereçamento por conteúdo e um slot de batch por sequência. */
    if (input->shape[0] != 1 || cell->gate_weights[0] == NULL ||
        cell->gate_weights[1] == NULL) {
        return NULL;
    }
    const size_t slots = cell->memory_slots;
    const size_t mem_dim = cell->gate_weights[0]->shape[1];
    if (slots == 0) {
        return NULL;
    }
    NC_Tensor *mem = cell->state.ntm.memory_bank;
    if (mem == NULL) {
        const size_t ms[2] = { slots, mem_dim };
        mem = nc_tensor_create(scratch, ms, 2, true);
        if (mem == NULL) {
            return NULL;
        }
        nc_tensor_fill_random_normal(mem, 0.0f, 0.1f);
        cell->state.ntm.memory_bank = mem;
    }

    NC_Tensor *key_pre = nc_nn_matmul(tape, scratch, input, cell->gate_weights[0]);
    NC_Tensor *key = nc_nn_activation(tape, scratch, key_pre, NC_OP_TANH);
    NC_Tensor *key_t = nc_nn_transpose(tape, scratch, key);
    NC_Tensor *scores = nc_nn_matmul(tape, scratch, mem, key_t);
    NC_Tensor *weights = nc_nn_softmax(tape, scratch, scores);
    NC_Tensor *weights_t = nc_nn_transpose(tape, scratch, weights);
    NC_Tensor *read = nc_nn_matmul(tape, scratch, weights_t, mem);

    NC_Tensor *write_pre = nc_nn_matmul(tape, scratch, input, cell->gate_weights[1]);
    NC_Tensor *write_vec = nc_nn_activation(tape, scratch, write_pre, NC_OP_TANH);
    /* weights [slots,1] @ write_vec [1,mem_dim] -> atualização [slots,mem_dim]. */
    NC_Tensor *update = nc_nn_matmul(tape, scratch, weights, write_vec);
    NC_Tensor *new_mem = nc_nn_add(tape, scratch, mem, update);
    if (new_mem == NULL) {
        return NULL;
    }

    cell->state.ntm.memory_bank = new_mem;
    cell->state.ntm.read_head = read;
    cell->state.ntm.write_head = write_vec;
    return read;
}

static NC_Tensor *probabilistic_forward(NC_Cell *cell, const NC_Tensor *input,
                                        NC_Tape *tape, NC_Arena *scratch)
{
    /* Sem pesos de reconstrução, mantém o comportamento determinístico
     * usado pelo DCIGN (mu = x). */
    if (cell->gate_weights[0] == NULL || cell->gate_weights[1] == NULL) {
        return nc_nn_identity(tape, scratch, input);
    }
    NC_Tensor *mu = nc_nn_matmul(tape, scratch, input, cell->gate_weights[0]);
    if (cell->gate_bias[0] != NULL) {
        mu = nc_nn_add(tape, scratch, mu, cell->gate_bias[0]);
    }
    NC_Tensor *logvar = nc_nn_matmul(tape, scratch, input, cell->gate_weights[1]);
    if (cell->gate_bias[1] != NULL) {
        logvar = nc_nn_add(tape, scratch, logvar, cell->gate_bias[1]);
    }
    NC_Tensor *half = nc_nn_scale(tape, scratch, logvar, 0.5f);
    NC_Tensor *sigma = nc_nn_exp(tape, scratch, half);
    NC_Tensor *eps = nc_tensor_create(scratch, input->shape, input->ndim, false);
    if (mu == NULL || sigma == NULL || eps == NULL) {
        return NULL;
    }
    nc_tensor_fill_random_normal(eps, 0.0f, 1.0f);
    NC_Tensor *noise = nc_nn_mul(tape, scratch, sigma, eps);
    NC_Tensor *z = nc_nn_add(tape, scratch, mu, noise);
    cell->state.probabilistic.mu = mu;
    cell->state.probabilistic.log_var = logvar;
    cell->state.probabilistic.eps = eps;
    return z;
}

static NC_Tensor *spiking_forward(NC_Cell *cell, const NC_Tensor *input,
                                  NC_Tape *tape, NC_Arena *scratch)
{
    (void)tape; /* LIF não é diferenciável: usa STDP/Hebbian */
    if (cell->weights == NULL || input->ndim != 2) return NULL;
    const size_t hidden = cell->weights->shape[1];
    const size_t n = input->shape[0];
    if (cell->state.spiking.membrane == NULL) {
        cell->state.spiking.membrane =
            (float *)nc_arena_alloc(scratch, hidden * sizeof(float),
                                    NC_TENSOR_ALIGNMENT);
        if (cell->state.spiking.membrane == NULL) return NULL;
        memset(cell->state.spiking.membrane, 0, hidden * sizeof(float));
    }
    const size_t shape[2] = { n, hidden };
    NC_Tensor *out = nc_tensor_create(scratch, shape, 2, false);
    if (out == NULL) return NULL;

    for (size_t sample = 0; sample < n; ++sample) {
        for (size_t h = 0; h < hidden; ++h) {
            float sum = 0.0f;
            for (size_t a = 0; a < input->shape[1]; ++a) {
                sum += input->data[sample * input->strides[0] + a] *
                       cell->weights->data[a * hidden + h];
            }
            float v = cell->state.spiking.decay_rate *
                      cell->state.spiking.membrane[h] + sum;
            float spike = 0.0f;
            if (cell->state.spiking.current_refractory > 0) {
                --cell->state.spiking.current_refractory;
                v = cell->state.spiking.v_reset;
            } else if (v >= cell->state.spiking.v_threshold) {
                spike = 1.0f;
                v = cell->state.spiking.v_reset;
                cell->state.spiking.current_refractory =
                    cell->state.spiking.refractory_steps;
                cell->state.spiking.last_post_time += 1.0f;
            }
            cell->state.spiking.membrane[h] = v;
            out->data[sample * hidden + h] = spike;
        }
    }
    return out;
}

static NC_Tensor *backfed_forward(NC_Cell *cell, const NC_Tensor *input,
                                  NC_Tape *tape, NC_Arena *scratch)
{
    if (cell->weights == NULL) return NULL;
    NC_Tensor *z = (NC_Tensor *)input;
    if (cell->recurrent_weights != NULL) {
        const size_t hidden = cell->weights->shape[1];
        NC_Tensor *h_prev = ensure_state(scratch, &cell->state.recurrent.hidden,
                                         input->shape[0], hidden);
        if (h_prev == NULL) return NULL;
        NC_Tensor *feedback =
            nc_nn_matmul(tape, scratch, h_prev, cell->recurrent_weights);
        z = nc_nn_add(tape, scratch, z, feedback);
    }
    NC_Tensor *out = nc_nn_matmul(tape, scratch, z, cell->weights);
    if (cell->bias != NULL) out = nc_nn_add(tape, scratch, out, cell->bias);
    cell->activity = nc_nn_activation(tape, scratch, out, NC_OP_TANH);
    return cell->activity;
}

NC_Tensor *nc_cell_forward(NC_Cell *cell, const NC_Tensor *input,
                           NC_Tape *tape, NC_Arena *scratch)
{
    if (cell == NULL || input == NULL || scratch == NULL) {
        return NULL;
    }

    switch (cell->type) {
    case NC_CELL_INPUT:
        cell->activity = (NC_Tensor *)input;
        return cell->activity;
    case NC_CELL_NOISY_INPUT:
        cell->activity = noisy_input_forward(cell, input, tape, scratch);
        return cell->activity;
    case NC_CELL_MATCH_INPUT_OUTPUT:
        /* Sem pesos, atua como pass-through (saída de redes de reconstrução). */
        if (cell->weights == NULL) {
            cell->activity = nc_nn_identity(tape, scratch, input);
            return cell->activity;
        }
        /* Com pesos, comporta-se como uma camada densa (AE/DAE). */
        /* fallthrough */
    case NC_CELL_HIDDEN:
        if (cell->competitive) {
            /* Kohonen: os pesos são protótipos, atualizados pelo BMU. */
            cell->activity = nc_nn_identity(tape, scratch, input);
            return cell->activity;
        }
        /* fallthrough */
    case NC_CELL_OUTPUT: {
        const NC_Tensor *dense_input = input;
        if (input->ndim == 4) {
            NC_Tensor *flat = nc_tensor_flatten(scratch, input,
                                                input->grad != NULL);
            if (flat == NULL) {
                return NULL;
            }
            dense_input = nc_nn_activation(tape, scratch, flat, NC_OP_LINEAR);
            if (dense_input == NULL) {
                return NULL;
            }
        }
        if (cell->weights == NULL) {
            return NULL;
        }
        cell->activity = affine_forward(cell, dense_input, tape, scratch);
        return cell->activity;
    }
    case NC_CELL_KERNEL:
        if (cell->weights == NULL) return NULL;
        if (input->ndim == 2 && cell->weights->ndim == 2) {
            /* RBF: centros nas linhas de weights, features = exp(-||x-c||²). */
            const size_t centers = cell->weights->shape[0];
            const size_t dim = cell->weights->shape[1];
            const size_t shape[2] = { input->shape[0], centers };
            NC_Tensor *features = nc_tensor_create(scratch, shape, 2, false);
            if (features == NULL) return NULL;
            for (size_t n = 0; n < shape[0]; ++n) {
                for (size_t c = 0; c < centers; ++c) {
                    double dist2 = 0.0;
                    for (size_t j = 0; j < dim; ++j) {
                        const double d = input->data[n * input->strides[0] + j] -
                                         cell->weights->data[c * dim + j];
                        dist2 += d * d;
                    }
                    features->data[n * centers + c] = (float)exp(-dist2);
                }
            }
            cell->activity = features;
            return cell->activity;
        }
        if (input->ndim != 4) return NULL;
        cell->activity = nc_nn_conv2d(tape, scratch, input, cell->weights,
                                      cell->state.conv.stride,
                                      cell->state.conv.padding);
        return cell->activity;
    case NC_CELL_CONV_OR_POOL:
        if (input->ndim != 4) {
            return NULL;
        }
        if (cell->state.conv.is_pool) {
            cell->activity = cell->state.conv.avg_pool
                ? nc_nn_avg_pool2d(tape, scratch, input,
                                   cell->state.conv.pool_kernel,
                                   cell->state.conv.pool_stride)
                : nc_nn_max_pool2d(tape, scratch, input,
                                   cell->state.conv.pool_kernel,
                                   cell->state.conv.pool_stride);
        } else if (cell->weights != NULL) {
            cell->activity = cell->state.conv.transpose
                ? nc_nn_conv2d_transpose(tape, scratch, input, cell->weights,
                                         cell->state.conv.stride,
                                         cell->state.conv.padding)
                : nc_nn_conv2d(tape, scratch, input, cell->weights,
                               cell->state.conv.stride,
                               cell->state.conv.padding);
        } else {
            return NULL;
        }
        return cell->activity;
    case NC_CELL_PROBABILISTIC_HIDDEN:
        cell->activity = probabilistic_forward(cell, input, tape, scratch);
        return cell->activity;
    case NC_CELL_SPIKING_HIDDEN:
        cell->activity = spiking_forward(cell, input, tape, scratch);
        return cell->activity;
    case NC_CELL_BACKFED_INPUT:
        cell->activity = backfed_forward(cell, input, tape, scratch);
        return cell->activity;
    case NC_CELL_RECURRENT:
        cell->activity = rnn_forward(cell, input, tape, scratch);
        return cell->activity;
    case NC_CELL_MEMORY:
        cell->activity = lstm_forward(cell, input, tape, scratch);
        return cell->activity;
    case NC_CELL_DIFFERENT_MEMORY:
        /* 3 portes => GRU; 2 portes (chave/escrita) => NTM. */
        cell->activity = cell->gate_weights[2] != NULL
            ? gru_forward(cell, input, tape, scratch)
            : ntm_forward(cell, input, tape, scratch);
        return cell->activity;
    default:
        /* Células probabilísticas/spiking/kernel avançadas: marcos futuros. */
        return NULL;
    }
}

const char *nc_cell_type_name(NC_CellType type)
{
    switch (type) {
    case NC_CELL_BACKFED_INPUT:          return "BackfedInput";
    case NC_CELL_INPUT:                  return "Input";
    case NC_CELL_NOISY_INPUT:            return "NoisyInput";
    case NC_CELL_HIDDEN:                 return "Hidden";
    case NC_CELL_PROBABILISTIC_HIDDEN:   return "ProbabilisticHidden";
    case NC_CELL_SPIKING_HIDDEN:         return "SpikingHidden";
    case NC_CELL_OUTPUT:                 return "Output";
    case NC_CELL_MATCH_INPUT_OUTPUT:     return "MatchInputOutput";
    case NC_CELL_RECURRENT:              return "Recurrent";
    case NC_CELL_MEMORY:                 return "Memory";
    case NC_CELL_DIFFERENT_MEMORY:       return "DifferentMemory";
    case NC_CELL_KERNEL:                 return "Kernel";
    case NC_CELL_CONV_OR_POOL:           return "ConvOrPool";
    default:                             return "Unknown";
    }
}

const char *nc_activation_name(NC_Activation activation)
{
    switch (activation) {
    case NC_ACT_LINEAR:  return "Linear";
    case NC_ACT_RELU:    return "ReLU";
    case NC_ACT_TANH:    return "Tanh";
    case NC_ACT_SIGMOID: return "Sigmoid";
    case NC_ACT_GELU:    return "GELU";
    default:             return "Unknown";
    }
}
