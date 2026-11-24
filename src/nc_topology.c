#include "nc_topology.h"

#include <math.h>
#include <stdalign.h>
#include <string.h>

NC_Graph *nc_graph_create(NC_Arena *model_arena, size_t cell_capacity,
                          size_t edge_capacity, size_t tape_capacity)
{
    if (model_arena == NULL || cell_capacity == 0 || edge_capacity == 0 ||
        tape_capacity == 0) {
        return NULL;
    }
    const size_t saved = model_arena->offset;

    NC_Graph *graph = (NC_Graph *)nc_arena_alloc(model_arena, sizeof(*graph),
                                                 alignof(NC_Graph));
    NC_Cell *cells = (NC_Cell *)nc_arena_alloc(model_arena,
                                               cell_capacity * sizeof(NC_Cell),
                                               alignof(NC_Cell));
    NC_Edge *edges = (NC_Edge *)nc_arena_alloc(model_arena,
                                               edge_capacity * sizeof(NC_Edge),
                                               alignof(NC_Edge));
    NC_AdamState *adam = (NC_AdamState *)nc_arena_alloc(
        model_arena, cell_capacity * 16 * sizeof(NC_AdamState),
        alignof(NC_AdamState));
    NC_Tape *tape = nc_autograd_tape_init(model_arena, tape_capacity);

    if (graph == NULL || cells == NULL || edges == NULL || adam == NULL ||
        tape == NULL) {
        model_arena->offset = saved;
        return NULL;
    }

    memset(graph, 0, sizeof(*graph));
    memset(cells, 0, cell_capacity * sizeof(NC_Cell));
    memset(edges, 0, edge_capacity * sizeof(NC_Edge));
    memset(adam, 0, cell_capacity * 16 * sizeof(NC_AdamState));

    graph->model_arena = model_arena;
    graph->cells = cells;
    graph->cell_capacity = cell_capacity;
    graph->edges = edges;
    graph->edge_capacity = edge_capacity;
    graph->adam = adam;
    graph->adam_capacity = cell_capacity * 16;
    graph->tape = tape;
    return graph;
}

NC_Cell *nc_graph_add_cell(NC_Graph *graph, NC_CellType type,
                           NC_Activation activation)
{
    if (graph == NULL || graph->cell_count >= graph->cell_capacity) {
        return NULL;
    }
    NC_Cell *cell = &graph->cells[graph->cell_count];
    memset(cell, 0, sizeof(*cell));
    cell->id = graph->cell_count;
    cell->type = type;
    cell->activation = activation;
    graph->input_cell_id = 0;
    graph->output_cell_id = graph->cell_count;
    ++graph->cell_count;
    return cell;
}

bool nc_graph_connect(NC_Graph *graph, size_t src, size_t dst, NC_EdgeType type)
{
    if (graph == NULL || src >= graph->cell_count || dst >= graph->cell_count ||
        graph->edge_count >= graph->edge_capacity) {
        return false;
    }
    NC_Edge *edge = &graph->edges[graph->edge_count++];
    edge->src_cell_id = src;
    edge->dst_cell_id = dst;
    edge->type = type;
    edge->weight_tensor = NULL;
    return true;
}

static NC_Tensor *gather_input(NC_Graph *graph, size_t cell_id,
                               const NC_Tensor *flat, NC_Arena *scratch)
{
    NC_Tensor *acc = NULL;
    for (size_t i = 0; i < graph->edge_count; ++i) {
        const NC_Edge *edge = &graph->edges[i];
        if (edge->dst_cell_id != cell_id) {
            continue;
        }
        NC_Tensor *source = graph->cells[edge->src_cell_id].activity;
        if (source == NULL) {
            return NULL;
        }
        if (acc == NULL) {
            acc = source;
        } else {
            /* Soma de entradas (skip/residual): gradientes voltam para ambas. */
            acc = nc_nn_add(graph->tape, scratch, acc, source);
            if (acc == NULL) {
                return NULL;
            }
        }
    }

    if (acc == NULL) {
        const NC_CellType type = graph->cells[cell_id].type;
        if (type == NC_CELL_INPUT || type == NC_CELL_NOISY_INPUT) {
            acc = (NC_Tensor *)flat;
        }
    }
    return acc;
}

static bool graph_is_spatial(const NC_Graph *graph)
{
    for (size_t i = 0; i < graph->cell_count; ++i) {
        const NC_CellType type = graph->cells[i].type;
        if (type == NC_CELL_KERNEL || type == NC_CELL_CONV_OR_POOL) {
            return true;
        }
    }
    return false;
}

NC_Tensor *nc_graph_step(NC_Graph *graph, const NC_Tensor *input,
                         NC_Arena *scratch)
{
    if (graph == NULL || graph->tape == NULL || graph->cell_count == 0 ||
        input == NULL || scratch == NULL || !nc_tensor_layout_valid(input)) {
        return NULL;
    }

    for (size_t i = 0; i < graph->cell_count; ++i) {
        NC_Tensor *cell_input = gather_input(graph, i, input, scratch);
        if (cell_input == NULL) {
            return NULL;
        }
        NC_Tensor *activity = nc_cell_forward(&graph->cells[i], cell_input,
                                              graph->tape, scratch);
        if (activity == NULL) {
            return NULL;
        }
    }

    graph->last_output = graph->cells[graph->output_cell_id].activity;
    return graph->last_output;
}

NC_Tensor *nc_graph_forward(NC_Graph *graph, const NC_Tensor *input,
                            NC_Arena *scratch)
{
    if (graph == NULL || graph->tape == NULL || input == NULL ||
        !nc_tensor_layout_valid(input)) {
        return NULL;
    }
    nc_autograd_tape_reset(graph->tape);

    NC_Tensor *input_view = NULL;
    if (graph_is_spatial(graph)) {
        if (input->ndim != 4) {
            return NULL;
        }
        input_view = (NC_Tensor *)input;
    } else {
        /* requires_grad=true mantém o caminho de volta à saída do gerador
         * quando um grafo é alimentado por outro (treino adversarial). */
        input_view = nc_tensor_flatten(scratch, input, input->grad != NULL);
    }
    if (input_view == NULL) {
        return NULL;
    }
    return nc_graph_step(graph, input_view, scratch);
}

void nc_graph_zero_grads(NC_Graph *graph)
{
    if (graph == NULL) {
        return;
    }
    for (size_t i = 0; i < graph->cell_count; ++i) {
        NC_Cell *cell = &graph->cells[i];
        NC_Tensor *params[7] = {
            cell->weights, cell->recurrent_weights, cell->bias,
            cell->gate_weights[0], cell->gate_weights[1],
            cell->gate_weights[2], cell->gate_weights[3]
        };
        for (size_t p = 0; p < 7; ++p) {
            if (params[p] != NULL && params[p]->grad != NULL) {
                memset(params[p]->grad, 0, params[p]->size * sizeof(float));
            }
        }
        for (size_t g = 0; g < 4; ++g) {
            if (cell->gate_recurrent[g] != NULL &&
                cell->gate_recurrent[g]->grad != NULL) {
                memset(cell->gate_recurrent[g]->grad, 0,
                       cell->gate_recurrent[g]->size * sizeof(float));
            }
            if (cell->gate_bias[g] != NULL && cell->gate_bias[g]->grad != NULL) {
                memset(cell->gate_bias[g]->grad, 0,
                       cell->gate_bias[g]->size * sizeof(float));
            }
        }
        if (cell->activity != NULL && cell->activity->grad != NULL) {
            memset(cell->activity->grad, 0, cell->activity->size * sizeof(float));
        }
    }
}

static void graph_backward_impl(NC_Graph *graph, bool zero_grads)
{
    if (graph == NULL || graph->tape == NULL) {
        return;
    }
    if (zero_grads) {
        nc_graph_zero_grads(graph);
    }
    /* Regularização L1 (SAE): semeia o gradiente da ativação oculta. */
    if (graph->sparse_cell != NULL && graph->sparse_cell->activity != NULL &&
        graph->sparse_cell->activity->grad != NULL && graph->sparsity_beta > 0.0f) {
        NC_Tensor *h = graph->sparse_cell->activity;
        const float factor = graph->sparsity_beta / (float)h->size;
        for (size_t i = 0; i < h->size; ++i) {
            const float value = h->data[i];
            h->grad[i] += value > 0.0f ? factor : (value < 0.0f ? -factor : 0.0f);
        }
    }

    nc_autograd_backward(graph->tape);
}

void nc_graph_backward(NC_Graph *graph)
{
    graph_backward_impl(graph, true);
}

void nc_graph_backward_no_zero(NC_Graph *graph)
{
    graph_backward_impl(graph, false);
}

void nc_graph_ensure_adam_states(NC_Graph *graph)
{
    if (graph == NULL || graph->adam_ready) {
        return;
    }
    for (size_t i = 0; i < graph->cell_count; ++i) {
        NC_Cell *cell = &graph->cells[i];
        NC_Tensor *params[15] = {
            cell->weights, cell->recurrent_weights, cell->bias,
            cell->gate_weights[0], cell->gate_weights[1],
            cell->gate_weights[2], cell->gate_weights[3],
            cell->gate_recurrent[0], cell->gate_recurrent[1],
            cell->gate_recurrent[2], cell->gate_recurrent[3],
            cell->gate_bias[0], cell->gate_bias[1],
            cell->gate_bias[2], cell->gate_bias[3]
        };
        for (size_t p = 0; p < 15; ++p) {
            NC_Tensor *param = params[p];
            if (param == NULL || !param->requires_grad ||
                graph->adam_count >= graph->adam_capacity) {
                continue;
            }
            NC_AdamState *state = &graph->adam[graph->adam_count++];
            state->param = param;
            state->m = nc_tensor_create(graph->model_arena, param->shape,
                                        param->ndim, false);
            state->v = nc_tensor_create(graph->model_arena, param->shape,
                                        param->ndim, false);
            state->step = 0;
        }
    }
    graph->adam_ready = true;
}

void nc_graph_destroy(NC_Graph *graph)
{
    if (graph == NULL) {
        return;
    }
    /* A model arena é a dona de toda a memória; apenas invalidamos os ponteiros. */
    graph->cells = NULL;
    graph->cell_count = 0;
    graph->edges = NULL;
    graph->edge_count = 0;
    graph->tape = NULL;
    graph->adam = NULL;
    graph->adam_count = 0;
    graph->adam_ready = false;
    graph->last_output = NULL;
    graph->sparse_cell = NULL;
}

size_t nc_graph_cell_count(const NC_Graph *graph)
{
    return graph != NULL ? graph->cell_count : 0;
}

NC_Cell *nc_graph_cell(NC_Graph *graph, size_t id)
{
    if (graph == NULL || id >= graph->cell_count) {
        return NULL;
    }
    return &graph->cells[id];
}
