#include "nc_sequence.h"

#include <stdalign.h>

NC_Tensor **nc_sequence_forward(NC_Graph *graph, const NC_Tensor *x_seq,
                                size_t steps, NC_Arena *scratch)
{
    if (graph == NULL || graph->tape == NULL || graph->cell_count == 0 ||
        x_seq == NULL || x_seq->ndim != 3 || x_seq->shape[0] != steps ||
        steps == 0 || scratch == NULL) {
        return NULL;
    }

    nc_autograd_tape_reset(graph->tape);

    NC_Tensor **outputs = (NC_Tensor **)nc_arena_alloc(
        scratch, steps * sizeof(NC_Tensor *), alignof(NC_Tensor *));
    if (outputs == NULL) {
        return NULL;
    }

    const size_t batch = x_seq->shape[1];
    const size_t features = x_seq->shape[2];
    if (batch == 0 || features == 0) {
        return NULL;
    }

    for (size_t t = 0; t < steps; ++t) {
        NC_Tensor *view = (NC_Tensor *)nc_arena_alloc(scratch, sizeof(NC_Tensor),
                                                      alignof(NC_Tensor));
        if (view == NULL) {
            return NULL;
        }
        *view = *x_seq;
        view->ndim = 2;
        view->shape[0] = batch;
        view->shape[1] = features;
        view->shape[2] = 0;
        view->shape[3] = 0;
        view->strides[0] = features;
        view->strides[1] = 1;
        view->strides[2] = 0;
        view->strides[3] = 0;
        view->size = batch * features;
        view->data = x_seq->data + t * batch * features;
        view->grad = NULL;
        view->requires_grad = false;

        outputs[t] = nc_graph_step(graph, view, scratch);
        if (outputs[t] == NULL) {
            return NULL;
        }
    }
    return outputs;
}

void nc_sequence_backward(NC_Graph *graph)
{
    if (graph == NULL || graph->tape == NULL) {
        return;
    }
    nc_graph_zero_grads(graph);
    nc_autograd_backward(graph->tape);
}

void nc_sequence_reset_state(NC_Graph *graph)
{
    if (graph == NULL) {
        return;
    }
    for (size_t i = 0; i < graph->cell_count; ++i) {
        nc_cell_reset_state(&graph->cells[i]);
    }
    graph->last_output = NULL;
}
