#ifndef NC_AUTOGRAD_H
#define NC_AUTOGRAD_H

#include <stdbool.h>
#include <stddef.h>

#include "nc_tensor.h"

/* Operações rastreáveis pelo Marco 2. */
typedef enum {
    NC_OP_MATMUL,
    NC_OP_ADD,
    NC_OP_RELU,
    NC_OP_TANH,
    NC_OP_SIGMOID,
    NC_OP_GELU,
    NC_OP_DROPOUT,
    NC_OP_MSE,
    NC_OP_LINEAR,
    NC_OP_CONV2D,
    NC_OP_CONV_TRANSPOSE,
    NC_OP_MAXPOOL,
    NC_OP_AVGPOOL
} NC_OpType;

struct NC_TapeNode;

typedef void (*NC_BackwardFn)(struct NC_TapeNode *node);

/* Parâmetros específicos de cada operação, apontados por custom_params. */
typedef struct NC_OpParams {
    NC_OpType op;
    NC_Tensor *target; /* NC_OP_MSE */
    NC_Tensor *mask;   /* NC_OP_DROPOUT */
    size_t *pool_indices; /* NC_OP_MAXPOOL: offsets exatos do máximo */
    float scale;       /* fator de escala do gradiente (ex.: 2/size) */
    size_t stride;     /* convolução/pooling */
    size_t padding;    /* convolução */
    size_t kernel_size;/* pooling */
} NC_OpParams;

typedef struct NC_TapeNode {
    NC_Tensor *inputs[3];
    NC_Tensor *output;
    NC_BackwardFn backward_fn;
    void *custom_params;
} NC_TapeNode;

typedef struct {
    NC_TapeNode *nodes;
    size_t count;
    size_t capacity;
    NC_Arena *arena; /* arena persistente, usada para parâmetros dos nós */
} NC_Tape;

/*
 * A fita é alocada na arena persistente (model_arena); os nós são reutilizados
 * a cada iteração via nc_autograd_tape_reset(). Retorna NULL em caso de falha.
 */
NC_Tape *nc_autograd_tape_init(NC_Arena *arena, size_t capacity);
void nc_autograd_tape_reset(NC_Tape *tape);

/*
 * Registra uma operação. `params` deve apontar para dados com vida útil até o
 * backward (aloque na scratch arena). Esta assinatura difere do esboço do
 * plano: o ponteiro da fita e os parâmetros da operação são necessários para
 * suportar múltiplas fitas e operações com estado.
 */
void nc_autograd_record(NC_Tape *tape, NC_Tensor *in1, NC_Tensor *in2,
                        NC_Tensor *out, NC_BackwardFn fn, void *params);

/* Executa a fita em ordem reversa, acumulando gradientes em inputs[].grad. */
void nc_autograd_backward(NC_Tape *tape);

/* ------------------------ Operações diferenciáveis ------------------------ */

NC_Tensor *nc_nn_matmul(NC_Tape *tape, NC_Arena *scratch,
                        const NC_Tensor *A, const NC_Tensor *B);
NC_Tensor *nc_nn_add(NC_Tape *tape, NC_Arena *scratch,
                     const NC_Tensor *A, const NC_Tensor *B);
NC_Tensor *nc_nn_activation(NC_Tape *tape, NC_Arena *scratch,
                            const NC_Tensor *x, NC_OpType op);
NC_Tensor *nc_nn_dropout(NC_Tape *tape, NC_Arena *scratch,
                         const NC_Tensor *x, float keep_prob);

/* Identidade diferenciável para qualquer ndim (1–4). */
NC_Tensor *nc_nn_identity(NC_Tape *tape, NC_Arena *scratch,
                          const NC_Tensor *x);

NC_Tensor *nc_nn_conv2d(NC_Tape *tape, NC_Arena *scratch,
                        const NC_Tensor *input, const NC_Tensor *kernel,
                        size_t stride, size_t padding);
NC_Tensor *nc_nn_conv2d_transpose(NC_Tape *tape, NC_Arena *scratch,
                                  const NC_Tensor *input, const NC_Tensor *kernel,
                                  size_t stride, size_t padding);
NC_Tensor *nc_nn_max_pool2d(NC_Tape *tape, NC_Arena *scratch,
                            const NC_Tensor *input, size_t kernel_size,
                            size_t stride);
NC_Tensor *nc_nn_avg_pool2d(NC_Tape *tape, NC_Arena *scratch,
                            const NC_Tensor *input, size_t kernel_size,
                            size_t stride);

/* Semente de gradiente do erro de reconstrução (MSE). */
void nc_autograd_record_mse(NC_Tape *tape, const NC_Tensor *pred,
                            const NC_Tensor *target);

#endif /* NC_AUTOGRAD_H */
