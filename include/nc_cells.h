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
    float *last_pre_times;   /* timestamps do último spike pré por neurônio */
    float *last_post_times;  /* timestamps do último spike pós por neurônio */
    size_t neuron_count;     /* dimensão dos arrays de timestamps */
    float *membrane; /* potentials por neurônio, alocados na scratch */
} NC_CellSpikingState;

typedef struct {
    NC_Tensor *cell_state;
    NC_Tensor *gate_input;
    NC_Tensor *gate_forget;
    NC_Tensor *gate_output;
    NC_Tensor *hidden;
} NC_CellLSTMState;

typedef struct {
    NC_Tensor *hidden;
} NC_CellRecurrentState;

typedef struct {
    NC_Tensor *hidden;
    NC_Tensor *gate_z;
    NC_Tensor *gate_r;
    NC_Tensor *candidate;
} NC_CellGRUState;

typedef struct {
    NC_Tensor *memory_bank;
    NC_Tensor *read_head;
    NC_Tensor *write_head;
} NC_CellNTMState;

/* Parâmetros de células convolucionais / pooling (Marco 3). */
typedef struct {
    size_t stride;
    size_t padding;
    size_t pool_kernel;
    size_t pool_stride;
    bool transpose;
    bool is_pool;
    bool avg_pool;
} NC_CellConvState;

typedef struct NC_Cell {
    size_t id;
    NC_CellType type;
    NC_Tensor *weights;
    NC_Tensor *recurrent_weights;
    NC_Tensor *bias;
    NC_Tensor *activity;
    NC_Activation activation;
    size_t memory_slots; /* NTM: linhas do banco de memória */
    bool competitive;    /* Kohonen: pesos são protótipos, não matriz de projeção */

    /* Porte separados: [0..gates-1] = entrada, recorrrente e viés. */
    NC_Tensor *gate_weights[4];
    NC_Tensor *gate_recurrent[4];
    NC_Tensor *gate_bias[4];

    union {
        NC_CellProbabilisticState probabilistic;
        NC_CellSpikingState spiking;
        NC_CellLSTMState lstm;
        NC_CellNTMState ntm;
        NC_CellConvState conv;
        NC_CellRecurrentState recurrent;
        NC_CellGRUState gru;
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

/* Cria pesos de entrada/recorrentes/viés para `gates` portes (1..4). */
bool nc_cell_init_gated_params(NC_Cell *cell, NC_Arena *arena,
                               size_t in_dim, size_t hidden_dim, size_t gates);

/* Cria weights convolucionais [c_out, c_in, k_h, k_w] e bias opcional. */
bool nc_cell_init_conv_params(NC_Cell *cell, NC_Arena *arena,
                              size_t c_out, size_t c_in,
                              size_t k_h, size_t k_w, bool with_bias);

/* Zera os ponteiros de estado recorrente (chamado no início de cada sequência). */
void nc_cell_reset_state(NC_Cell *cell);

/* Zera os gradientes dos tensores de estado ainda referenciados pela union. */
void nc_cell_zero_state_grads(NC_Cell *cell);

/* Executa a célula; retorna NULL para tipos ainda não implementados. */
NC_Tensor *nc_cell_forward(NC_Cell *cell, const NC_Tensor *input,
                           NC_Tape *tape, NC_Arena *scratch);

const char *nc_cell_type_name(NC_CellType type);
const char *nc_activation_name(NC_Activation activation);

#endif /* NC_CELLS_H */
