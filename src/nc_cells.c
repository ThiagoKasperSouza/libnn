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
        if (cell->weights == NULL || input->ndim != 4) {
            return NULL;
        }
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
    case NC_CELL_PROBABILISTIC_HIDDEN: {
        /* Latente determinístico neste marco: mu = x. Amostragem e KL (VAE)
         * pertencem ao Marco 5; os tensores de estado já são reservados. */
        if (cell->state.probabilistic.mu == NULL) {
            const size_t shape[4] = { input->shape[0], input->shape[1],
                                      input->shape[2] ? input->shape[2] : 1,
                                      input->shape[3] ? input->shape[3] : 1 };
            const size_t ndim = input->ndim;
            cell->state.probabilistic.mu =
                nc_tensor_create(scratch, shape, ndim, false);
            cell->state.probabilistic.log_var =
                nc_tensor_create(scratch, shape, ndim, false);
            cell->state.probabilistic.eps =
                nc_tensor_create(scratch, shape, ndim, false);
        }
        cell->activity = nc_nn_identity(tape, scratch, input);
        return cell->activity;
    }
    default:
        /* Células recorrentes/energéticas/spiking pertencem a marcos futuros. */
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
