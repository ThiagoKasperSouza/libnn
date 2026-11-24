#ifndef NC_TOPOLOGY_H
#define NC_TOPOLOGY_H

#include <stdbool.h>
#include <stddef.h>

#include "nc_autograd.h"
#include "nc_cells.h"

typedef enum {
    NC_EDGE_FORWARD,
    NC_EDGE_RECURRENT,
    NC_EDGE_RESIDUAL,
    NC_EDGE_BIDIRECTIONAL,
    NC_EDGE_FROZEN
} NC_EdgeType;

typedef struct NC_Edge {
    size_t src_cell_id;
    size_t dst_cell_id;
    NC_EdgeType type;
    NC_Tensor *weight_tensor;
} NC_Edge;

/* Estado do Adam por parâmetro (criado na primeira chamada do treinador). */
typedef struct {
    NC_Tensor *param;
    NC_Tensor *m;
    NC_Tensor *v;
    size_t step;
} NC_AdamState;

typedef struct NC_Graph {
    NC_Arena *model_arena;
    NC_Cell *cells;
    size_t cell_count;
    size_t cell_capacity;
    NC_Edge *edges;
    size_t edge_count;
    size_t edge_capacity;
    NC_Tape *tape;
    size_t input_cell_id;
    size_t output_cell_id;
    NC_AdamState *adam;
    size_t adam_count;
    size_t adam_capacity;
    bool adam_ready;
    float sparsity_target;
    float sparsity_beta;
    NC_Cell *sparse_cell;
    NC_Tensor *last_output;
} NC_Graph;

/* Toda a memória vem da model arena; tape_capacity é o máximo de nós por iteração. */
NC_Graph *nc_graph_create(NC_Arena *model_arena, size_t cell_capacity,
                          size_t edge_capacity, size_t tape_capacity);
NC_Cell *nc_graph_add_cell(NC_Graph *graph, NC_CellType type,
                           NC_Activation activation);
bool nc_graph_connect(NC_Graph *graph, size_t src, size_t dst, NC_EdgeType type);

/* Achata a entrada contígua para [batch, features], executa as células em
 * ordem e grava as operações na fita. Retorna NULL se a entrada não tiver
 * layout contíguo. */
NC_Tensor *nc_graph_forward(NC_Graph *graph, const NC_Tensor *input,
                            NC_Arena *scratch);

/* Aplica a penalidade de sparsity (se houver) e roda a fita em ordem reversa. */
void nc_graph_backward(NC_Graph *graph);

/* Variante sem zerar gradientes: usada quando o gradiente de saída já foi
 * semeado por outro grafo (treino adversarial do gerador). */
void nc_graph_backward_no_zero(NC_Graph *graph);

/* Cria os estados Adam para os parâmetros treináveis (idempotente). */
void nc_graph_ensure_adam_states(NC_Graph *graph);

/* Não libera memória individual: a model arena é a dona de tudo. */
void nc_graph_destroy(NC_Graph *graph);

size_t nc_graph_cell_count(const NC_Graph *graph);
NC_Cell *nc_graph_cell(NC_Graph *graph, size_t id);

#endif /* NC_TOPOLOGY_H */
