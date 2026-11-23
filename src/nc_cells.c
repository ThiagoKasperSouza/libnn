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
    case NC_CELL_HIDDEN:
    case NC_CELL_OUTPUT:
    case NC_CELL_MATCH_INPUT_OUTPUT:
        if (cell->weights == NULL) {
            return NULL;
        }
        cell->activity = affine_forward(cell, input, tape, scratch);
        return cell->activity;
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
