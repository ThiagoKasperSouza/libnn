#ifndef NC_CELLS_H
#define NC_CELLS_H

#include <stdbool.h>
#include <stddef.h>

#include "nc_autograd.h"
#include "nc_tensor.h"

/* As 13 primitivas do infográfico (as demais serão comportamentais nos marcos seguintes). */
typedef enum {
    NC_CELL_BACKFED_INPUT,
    NC_CELL_INPUT,
    NC_CELL_NOISY_INPUT,
    NC_CELL_HIDDEN,
    NC_CELL_PROBABILISTIC_HIDDEN,
    NC_CELL_SPIKING_HIDDEN,
    NC_CELL_OUTPUT,
    NC_CELL_MATCH_INPUT_OUTPUT,
    NC_CELL_RECURRENT,
    NC_CELL_MEMORY,
    NC_CELL_DIFFERENT_MEMORY,
    NC_CELL_KERNEL,
    NC_CELL_CONV_OR_POOL
} NC_CellType;

typedef enum {
    NC_ACT_LINEAR,
    NC_ACT_RELU,
    NC_ACT_TANH,
    NC_ACT_SIGMOID,
    NC_ACT_GELU
} NC_Activation;

typedef struct {
    NC_Tensor *mu;
    NC_Tensor *log_var;
    NC_Tensor *eps;
} NC_CellProbabilisticState;

typedef struct {
    float v_membrane;
    float v_threshold;
    float v_reset;
    float decay_rate;
    size_t refractory_steps;
    size_t current_refractory;
} NC_CellSpikingState;

typedef struct {
    NC_Tensor *cell_state;
    NC_Tensor *gate_input;
    NC_Tensor *gate_forget;
    NC_Tensor *gate_output;
} NC_CellLSTMState;

typedef struct {
    NC_Tensor *memory_bank;
    NC_Tensor *read_head;
    NC_Tensor *write_head;
} NC_CellNTMState;

typedef struct NC_Cell {
    size_t id;
    NC_CellType type;
    NC_Tensor *weights;
    NC_Tensor *bias;
    NC_Tensor *activity;
    NC_Activation activation;

    union {
        NC_CellProbabilisticState probabilistic;
        NC_CellSpikingState spiking;
        NC_CellLSTMState lstm;
        NC_CellNTMState ntm;
        struct {
            float noise_std;
            float dropout_rate;
        } noisy;
    } state;
} NC_Cell;

NC_Cell *nc_cell_create(NC_Arena *arena, size_t id, NC_CellType type,
                        NC_Activation activation);

/* Cria weights [in_dim, out_dim] e (opcionalmente) bias [1, out_dim]. */
bool nc_cell_init_params(NC_Cell *cell, NC_Arena *arena,
                         size_t in_dim, size_t out_dim, bool with_bias);

/* Executa a célula; retorna NULL para tipos ainda não implementados. */
NC_Tensor *nc_cell_forward(NC_Cell *cell, const NC_Tensor *input,
                           NC_Tape *tape, NC_Arena *scratch);

const char *nc_cell_type_name(NC_CellType type);
const char *nc_activation_name(NC_Activation activation);

#endif /* NC_CELLS_H */
