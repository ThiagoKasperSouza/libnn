#include "nc_zoo.h"

/* Capacidade generosa o suficiente para as topologias feedforward do Marco 2. */
#define ZOO_TAPE_CAPACITY 128

static NC_Graph *new_graph(NC_Arena *model, size_t cell_count, size_t edge_count)
{
    return nc_graph_create(model, cell_count, edge_count, ZOO_TAPE_CAPACITY);
}

static NC_Cell *add_layer(NC_Graph *graph, NC_CellType type,
                          NC_Activation activation, size_t in_dim, size_t out_dim)
{
    NC_Cell *cell = nc_graph_add_cell(graph, type, activation);
    if (cell == NULL || !nc_cell_init_params(cell, graph->model_arena,
                                             in_dim, out_dim, true)) {
        return NULL;
    }
    return cell;
}

/* --------------------------- P / FF / DFF ---------------------------- */

NC_Graph *nc_zoo_create_perceptron(NC_Arena *model, size_t input_dim,
                                   size_t output_dim)
{
    NC_Graph *g = new_graph(model, 2, 1);
    if (g == NULL) {
        return NULL;
    }
    if (nc_graph_add_cell(g, NC_CELL_INPUT, NC_ACT_LINEAR) == NULL ||
        add_layer(g, NC_CELL_OUTPUT, NC_ACT_LINEAR, input_dim, output_dim) == NULL ||
        !nc_graph_connect(g, 0, 1, NC_EDGE_FORWARD)) {
        return NULL;
    }
    return g;
}

NC_Graph *nc_zoo_create_ff(NC_Arena *model, size_t input_dim,
                           size_t hidden_dim, size_t output_dim)
{
    const size_t dims[1] = { hidden_dim };
    return nc_zoo_create_dff(model, input_dim, dims, 1, output_dim);
}

NC_Graph *nc_zoo_create_dff(NC_Arena *model, size_t input_dim,
                            const size_t hidden_dims[], size_t hidden_count,
                            size_t output_dim)
{
    if (hidden_dims == NULL || hidden_count == 0) {
        return NULL;
    }
    NC_Graph *g = new_graph(model, hidden_count + 2, hidden_count + 1);
    if (g == NULL) {
        return NULL;
    }
    if (nc_graph_add_cell(g, NC_CELL_INPUT, NC_ACT_LINEAR) == NULL) {
        return NULL;
    }

    size_t current_dim = input_dim;
    for (size_t i = 0; i < hidden_count; ++i) {
        NC_Cell *hidden = add_layer(g, NC_CELL_HIDDEN, NC_ACT_RELU,
                                    current_dim, hidden_dims[i]);
        if (hidden == NULL || !nc_graph_connect(g, i, i + 1, NC_EDGE_FORWARD)) {
            return NULL;
        }
        current_dim = hidden_dims[i];
    }

    const size_t out_id = hidden_count + 1;
    if (add_layer(g, NC_CELL_OUTPUT, NC_ACT_LINEAR, current_dim, output_dim) == NULL ||
        !nc_graph_connect(g, hidden_count, out_id, NC_EDGE_FORWARD)) {
        return NULL;
    }
    return g;
}

/* ------------------------------ Autoencoders -------------------------- */

static NC_Graph *create_autoencoder(NC_Arena *model, size_t input_dim,
                                    size_t hidden_dim, NC_CellType input_type)
{
    NC_Graph *g = new_graph(model, 3, 2);
    if (g == NULL) {
        return NULL;
    }
    NC_Cell *input = nc_graph_add_cell(g, input_type, NC_ACT_LINEAR);
    NC_Cell *hidden = add_layer(g, NC_CELL_HIDDEN, NC_ACT_RELU,
                                input_dim, hidden_dim);
    NC_Cell *recon = add_layer(g, NC_CELL_MATCH_INPUT_OUTPUT, NC_ACT_LINEAR,
                               hidden_dim, input_dim);
    if (input == NULL || hidden == NULL || recon == NULL ||
        !nc_graph_connect(g, 0, 1, NC_EDGE_FORWARD) ||
        !nc_graph_connect(g, 1, 2, NC_EDGE_FORWARD)) {
        return NULL;
    }
    return g;
}

NC_Graph *nc_zoo_create_ae(NC_Arena *model, size_t input_dim, size_t hidden_dim)
{
    return create_autoencoder(model, input_dim, hidden_dim, NC_CELL_INPUT);
}

NC_Graph *nc_zoo_create_dae(NC_Arena *model, size_t input_dim,
                            size_t hidden_dim, float noise_level)
{
    NC_Graph *g = create_autoencoder(model, input_dim, hidden_dim,
                                     NC_CELL_NOISY_INPUT);
    if (g != NULL) {
        g->cells[0].state.noisy.noise_std = noise_level;
        g->cells[0].state.noisy.dropout_rate = 0.0f;
    }
    return g;
}

NC_Graph *nc_zoo_create_sae(NC_Arena *model, size_t input_dim,
                            size_t hidden_dim, float sparsity_target,
                            float beta)
{
    NC_Graph *g = create_autoencoder(model, input_dim, hidden_dim, NC_CELL_INPUT);
    if (g != NULL) {
        g->sparsity_target = sparsity_target;
        g->sparsity_beta = beta;
        g->sparse_cell = &g->cells[1];
    }
    return g;
}

/* --------------------------------- DRN -------------------------------- */

NC_Graph *nc_zoo_create_drn(NC_Arena *model, size_t input_dim,
                            const size_t hidden_dims[], size_t hidden_count,
                            size_t output_dim)
{
    if (hidden_dims == NULL || hidden_count == 0) {
        return NULL;
    }
    /* Uma conexão residual de 2 camadas para cada par elegível. */
    const size_t max_edges = hidden_count + (hidden_count >= 2 ? hidden_count - 1 : 0);
    NC_Graph *g = new_graph(model, hidden_count + 2, max_edges + 1);
    if (g == NULL) {
        return NULL;
    }
    if (nc_graph_add_cell(g, NC_CELL_INPUT, NC_ACT_LINEAR) == NULL) {
        return NULL;
    }

    size_t current_dim = input_dim;
    for (size_t i = 0; i < hidden_count; ++i) {
        NC_Cell *hidden = add_layer(g, NC_CELL_HIDDEN, NC_ACT_RELU,
                                    current_dim, hidden_dims[i]);
        if (hidden == NULL || !nc_graph_connect(g, i, i + 1, NC_EDGE_FORWARD)) {
            return NULL;
        }
        /* Skip de identidade: só é válido quando as dimensionalidades casam. */
        if (i >= 2 && hidden_dims[i] == hidden_dims[i - 2]) {
            if (!nc_graph_connect(g, i - 1, i + 1, NC_EDGE_RESIDUAL)) {
                return NULL;
            }
        }
        current_dim = hidden_dims[i];
    }

    const size_t out_id = hidden_count + 1;
    if (add_layer(g, NC_CELL_OUTPUT, NC_ACT_LINEAR, current_dim, output_dim) == NULL ||
        !nc_graph_connect(g, hidden_count, out_id, NC_EDGE_FORWARD)) {
        return NULL;
    }
    return g;
}
