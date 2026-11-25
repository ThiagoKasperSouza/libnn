#include "nc_zoo.h"

#include <math.h>
#include <stdalign.h>

/* Capacidade suficiente para sequências desdobradas com LSTM/GRU. */
#define ZOO_TAPE_CAPACITY 8192

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

/* ------------------------- Topologias do Marco 3 ----------------------- */

NC_Graph *nc_zoo_create_dcn(NC_Arena *model, size_t channels, size_t h,
                            size_t w, size_t num_classes)
{
    if (h < 2 || w < 2) {
        return NULL;
    }
    const size_t filters = 8;
    NC_Graph *g = new_graph(model, 4, 3);
    if (g == NULL) {
        return NULL;
    }

    NC_Cell *input = nc_graph_add_cell(g, NC_CELL_INPUT, NC_ACT_LINEAR);
    NC_Cell *conv = nc_graph_add_cell(g, NC_CELL_KERNEL, NC_ACT_RELU);
    NC_Cell *pool = nc_graph_add_cell(g, NC_CELL_CONV_OR_POOL, NC_ACT_LINEAR);
    NC_Cell *out = add_layer(g, NC_CELL_OUTPUT, NC_ACT_LINEAR,
                             filters * (h / 2) * (w / 2), num_classes);
    if (input == NULL || conv == NULL || pool == NULL || out == NULL ||
        !nc_cell_init_conv_params(conv, model, filters, channels, 3, 3, true)) {
        return NULL;
    }
    conv->state.conv.stride = 1;
    conv->state.conv.padding = 1;
    pool->state.conv.is_pool = true;
    pool->state.conv.pool_kernel = 2;
    pool->state.conv.pool_stride = 2;

    return nc_graph_connect(g, 0, 1, NC_EDGE_FORWARD) &&
           nc_graph_connect(g, 1, 2, NC_EDGE_FORWARD) &&
           nc_graph_connect(g, 2, 3, NC_EDGE_FORWARD) ? g : NULL;
}

NC_Graph *nc_zoo_create_dn(NC_Arena *model, size_t channels, size_t h, size_t w)
{
    if (channels == 0 || h == 0 || w == 0) {
        return NULL;
    }
    NC_Graph *g = new_graph(model, 3, 2);
    if (g == NULL) {
        return NULL;
    }
    NC_Cell *input = nc_graph_add_cell(g, NC_CELL_INPUT, NC_ACT_LINEAR);
    NC_Cell *deconv = nc_graph_add_cell(g, NC_CELL_CONV_OR_POOL, NC_ACT_RELU);
    NC_Cell *match = nc_graph_add_cell(g, NC_CELL_MATCH_INPUT_OUTPUT, NC_ACT_LINEAR);
    if (input == NULL || deconv == NULL || match == NULL ||
        !nc_cell_init_conv_params(deconv, model, channels, channels, 2, 2, true)) {
        return NULL;
    }
    deconv->state.conv.transpose = true;
    deconv->state.conv.stride = 2;
    deconv->state.conv.padding = 0;

    return nc_graph_connect(g, 0, 1, NC_EDGE_FORWARD) &&
           nc_graph_connect(g, 1, 2, NC_EDGE_FORWARD) ? g : NULL;
}

NC_Graph *nc_zoo_create_dcign(NC_Arena *model, size_t channels, size_t h,
                              size_t w, size_t latent_channels)
{
    if (channels == 0 || h == 0 || w == 0 || latent_channels == 0) {
        return NULL;
    }
    NC_Graph *g = new_graph(model, 5, 4);
    if (g == NULL) {
        return NULL;
    }
    NC_Cell *input = nc_graph_add_cell(g, NC_CELL_INPUT, NC_ACT_LINEAR);
    NC_Cell *enc = nc_graph_add_cell(g, NC_CELL_KERNEL, NC_ACT_RELU);
    NC_Cell *latent = nc_graph_add_cell(g, NC_CELL_PROBABILISTIC_HIDDEN, NC_ACT_LINEAR);
    NC_Cell *dec = nc_graph_add_cell(g, NC_CELL_CONV_OR_POOL, NC_ACT_RELU);
    NC_Cell *match = nc_graph_add_cell(g, NC_CELL_MATCH_INPUT_OUTPUT, NC_ACT_LINEAR);
    if (input == NULL || enc == NULL || latent == NULL || dec == NULL ||
        match == NULL ||
        !nc_cell_init_conv_params(enc, model, latent_channels, channels, 3, 3, true) ||
        /* A transposta lê o kernel como [C_in, C_out, kH, kW]. */
        !nc_cell_init_conv_params(dec, model, latent_channels, channels, 3, 3, true)) {
        return NULL;
    }
    enc->state.conv.stride = 1;
    enc->state.conv.padding = 1;
    dec->state.conv.transpose = true;
    dec->state.conv.stride = 1;
    dec->state.conv.padding = 1;

    return nc_graph_connect(g, 0, 1, NC_EDGE_FORWARD) &&
           nc_graph_connect(g, 1, 2, NC_EDGE_FORWARD) &&
           nc_graph_connect(g, 2, 3, NC_EDGE_FORWARD) &&
           nc_graph_connect(g, 3, 4, NC_EDGE_FORWARD) ? g : NULL;
}

NC_GAN *nc_zoo_create_gan(NC_Arena *model, size_t latent_dim, size_t data_dim)
{
    if (model == NULL || latent_dim == 0 || data_dim == 0) {
        return NULL;
    }
    NC_GAN *gan = (NC_GAN *)nc_arena_alloc(model, sizeof(*gan), alignof(NC_GAN));
    if (gan == NULL) {
        return NULL;
    }
    const size_t hidden[2] = { 32, 32 };
    gan->generator = nc_zoo_create_dff(model, latent_dim, hidden, 2, data_dim);
    gan->discriminator = nc_zoo_create_dff(model, data_dim, hidden, 2, 1);
    if (gan->generator == NULL || gan->discriminator == NULL) {
        gan->generator = NULL;
        gan->discriminator = NULL;
        return NULL;
    }
    return gan;
}

/* ------------------------- Topologias do Marco 4 ----------------------- */

static NC_Graph *create_recurrent(NC_Arena *model, NC_CellType memory_type,
                                  size_t input_dim, size_t hidden_dim,
                                  size_t output_dim, size_t gates)
{
    if (model == NULL || input_dim == 0 || hidden_dim == 0 || output_dim == 0) {
        return NULL;
    }
    NC_Graph *g = new_graph(model, 3, 2);
    if (g == NULL) {
        return NULL;
    }
    NC_Cell *input = nc_graph_add_cell(g, NC_CELL_INPUT, NC_ACT_LINEAR);
    NC_Cell *memory = nc_graph_add_cell(g, memory_type, NC_ACT_LINEAR);
    NC_Cell *out = add_layer(g, NC_CELL_OUTPUT, NC_ACT_LINEAR,
                             hidden_dim, output_dim);
    if (input == NULL || memory == NULL || out == NULL) {
        return NULL;
    }

    bool params_ok = false;
    if (gates == 0) {
        params_ok = nc_cell_init_params(memory, model, input_dim, hidden_dim, true);
        if (params_ok) {
            const size_t rs[2] = { hidden_dim, hidden_dim };
            memory->recurrent_weights =
                nc_tensor_create(model, rs, 2, true);
            params_ok = memory->recurrent_weights != NULL;
            if (params_ok) {
                nc_tensor_fill_random_normal(memory->recurrent_weights, 0.0f,
                    sqrtf(2.0f / (float)(hidden_dim + hidden_dim)));
            }
        }
    } else {
        params_ok = nc_cell_init_gated_params(memory, model, input_dim,
                                              hidden_dim, gates);
    }
    if (!params_ok ||
        !nc_graph_connect(g, 0, 1, NC_EDGE_RECURRENT) ||
        !nc_graph_connect(g, 1, 2, NC_EDGE_FORWARD)) {
        return NULL;
    }
    return g;
}

NC_Graph *nc_zoo_create_rnn(NC_Arena *model, size_t input_dim,
                            size_t hidden_dim, size_t output_dim)
{
    return create_recurrent(model, NC_CELL_RECURRENT, input_dim, hidden_dim,
                            output_dim, 0);
}

NC_Graph *nc_zoo_create_lstm(NC_Arena *model, size_t input_dim,
                             size_t hidden_dim, size_t output_dim)
{
    return create_recurrent(model, NC_CELL_MEMORY, input_dim, hidden_dim,
                            output_dim, 4);
}

NC_Graph *nc_zoo_create_gru(NC_Arena *model, size_t input_dim,
                            size_t hidden_dim, size_t output_dim)
{
    return create_recurrent(model, NC_CELL_DIFFERENT_MEMORY, input_dim,
                            hidden_dim, output_dim, 3);
}

NC_Graph *nc_zoo_create_ntm(NC_Arena *model, size_t input_dim, size_t slots,
                            size_t mem_dim, size_t output_dim)
{
    if (model == NULL || input_dim == 0 || slots == 0 || mem_dim == 0 ||
        output_dim == 0) {
        return NULL;
    }
    NC_Graph *g = new_graph(model, 3, 2);
    if (g == NULL) {
        return NULL;
    }
    NC_Cell *input = nc_graph_add_cell(g, NC_CELL_INPUT, NC_ACT_LINEAR);
    NC_Cell *memory = nc_graph_add_cell(g, NC_CELL_DIFFERENT_MEMORY, NC_ACT_LINEAR);
    NC_Cell *out = add_layer(g, NC_CELL_OUTPUT, NC_ACT_LINEAR,
                             mem_dim, output_dim);
    if (input == NULL || memory == NULL || out == NULL) {
        return NULL;
    }
    const size_t ws[2] = { input_dim, mem_dim };
    memory->gate_weights[0] = nc_tensor_create(model, ws, 2, true);
    memory->gate_weights[1] = nc_tensor_create(model, ws, 2, true);
    memory->memory_slots = slots;
    if (memory->gate_weights[0] == NULL || memory->gate_weights[1] == NULL) {
        return NULL;
    }
    const float stddev = sqrtf(2.0f / (float)(input_dim + mem_dim));
    nc_tensor_fill_random_normal(memory->gate_weights[0], 0.0f, stddev);
    nc_tensor_fill_random_normal(memory->gate_weights[1], 0.0f, stddev);

    return nc_graph_connect(g, 0, 1, NC_EDGE_RECURRENT) &&
           nc_graph_connect(g, 1, 2, NC_EDGE_FORWARD) ? g : NULL;
}

/* ------------------------- Topologias do Marco 5 ----------------------- */

static NC_Graph *simple_chain(NC_Arena *model, NC_CellType hidden_type,
                              size_t in_dim, size_t hidden_dim, size_t out_dim)
{
    (void)in_dim; /* pesos da célula oculta são inicializados pelo chamador */
    NC_Graph *g = new_graph(model, 3, 2);
    if (g == NULL) return NULL;
    NC_Cell *in = nc_graph_add_cell(g, NC_CELL_INPUT, NC_ACT_LINEAR);
    NC_Cell *hidden = nc_graph_add_cell(g, hidden_type, NC_ACT_LINEAR);
    NC_Cell *out = add_layer(g, NC_CELL_OUTPUT, NC_ACT_LINEAR, hidden_dim, out_dim);
    if (in == NULL || hidden == NULL || out == NULL ||
        !nc_graph_connect(g, 0, 1, NC_EDGE_FORWARD) ||
        !nc_graph_connect(g, 1, 2, NC_EDGE_FORWARD)) return NULL;
    return g;
}

static bool init_cd_cell(NC_Cell *cell, NC_Arena *model, size_t visible,
                         size_t hidden)
{
    if (!nc_cell_init_params(cell, model, visible, hidden, true)) return false;
    const size_t vb_shape[2] = { 1, visible };
    cell->recurrent_weights = nc_tensor_create(model, vb_shape, 2, true);
    if (cell->recurrent_weights == NULL) return false;
    nc_tensor_fill_zeros(cell->recurrent_weights);
    return true;
}

NC_Graph *nc_zoo_create_markov_chain(NC_Arena *model, size_t states, size_t hidden)
{
    NC_Graph *g = simple_chain(model, NC_CELL_PROBABILISTIC_HIDDEN,
                               states, hidden, states);
    if (g != NULL && !init_cd_cell(&g->cells[1], model, states, hidden)) return NULL;
    return g;
}

NC_Graph *nc_zoo_create_hopfield(NC_Arena *model, size_t dim)
{
    if (dim == 0) return NULL;
    NC_Graph *g = new_graph(model, 2, 1);
    if (g == NULL) return NULL;
    NC_Cell *in = nc_graph_add_cell(g, NC_CELL_BACKFED_INPUT, NC_ACT_TANH);
    NC_Cell *out = nc_graph_add_cell(g, NC_CELL_MATCH_INPUT_OUTPUT, NC_ACT_LINEAR);
    if (in == NULL || out == NULL ||
        !nc_cell_init_params(in, model, dim, dim, false) ||
        !nc_graph_connect(g, 0, 1, NC_EDGE_BIDIRECTIONAL)) return NULL;
    nc_tensor_fill_random_normal(in->weights, 0.0f, 0.1f);
    return g;
}

NC_Graph *nc_zoo_create_boltzmann(NC_Arena *model, size_t visible, size_t hidden)
{
    NC_Graph *g = simple_chain(model, NC_CELL_PROBABILISTIC_HIDDEN,
                               visible, hidden, visible);
    if (g != NULL && !init_cd_cell(&g->cells[1], model, visible, hidden)) return NULL;
    return g;
}

NC_Graph *nc_zoo_create_rbm(NC_Arena *model, size_t visible, size_t hidden)
{
    return nc_zoo_create_boltzmann(model, visible, hidden);
}

NC_Graph *nc_zoo_create_dbn(NC_Arena *model, size_t visible,
                            const size_t hidden_dims[], size_t layers)
{
    if (hidden_dims == NULL || layers == 0) return NULL;
    NC_Graph *g = new_graph(model, layers + 2, layers + 1);
    if (g == NULL) return NULL;
    if (nc_graph_add_cell(g, NC_CELL_INPUT, NC_ACT_LINEAR) == NULL) return NULL;
    size_t current = visible;
    for (size_t i = 0; i < layers; ++i) {
        NC_Cell *h = nc_graph_add_cell(g, NC_CELL_PROBABILISTIC_HIDDEN, NC_ACT_LINEAR);
        if (h == NULL || !init_cd_cell(h, model, current, hidden_dims[i]) ||
            !nc_graph_connect(g, i, i + 1, NC_EDGE_FORWARD)) return NULL;
        current = hidden_dims[i];
    }
    NC_Cell *out = add_layer(g, NC_CELL_OUTPUT, NC_ACT_LINEAR, current, visible);
    return (out != NULL && nc_graph_connect(g, layers, layers + 1, NC_EDGE_FORWARD))
        ? g : NULL;
}

NC_Graph *nc_zoo_create_vae(NC_Arena *model, size_t input_dim, size_t latent_dim)
{
    NC_Graph *g = nc_zoo_create_ae(model, input_dim, latent_dim);
    if (g == NULL) return NULL;
    NC_Cell *latent = &g->cells[1];
    const size_t ws[2] = { input_dim, latent_dim };
    latent->gate_weights[0] = nc_tensor_create(model, ws, 2, true);
    latent->gate_weights[1] = nc_tensor_create(model, ws, 2, true);
    latent->gate_bias[0] = nc_tensor_create(model, (size_t[2]){1, latent_dim}, 2, true);
    latent->gate_bias[1] = nc_tensor_create(model, (size_t[2]){1, latent_dim}, 2, true);
    if (latent->gate_weights[0] == NULL || latent->gate_weights[1] == NULL ||
        latent->gate_bias[0] == NULL || latent->gate_bias[1] == NULL) return NULL;
    const float std = sqrtf(2.0f / (float)(input_dim + latent_dim));
    nc_tensor_fill_random_normal(latent->gate_weights[0], 0.0f, std);
    nc_tensor_fill_random_normal(latent->gate_weights[1], 0.0f, std);
    return g;
}

static NC_Graph *create_reservoir(NC_Arena *model, NC_CellType reservoir_type,
                                   size_t input_dim, size_t neurons,
                                   size_t output_dim, bool spiking)
{
    if (input_dim == 0 || neurons == 0 || output_dim == 0) return NULL;
    NC_Graph *g = new_graph(model, 3, 2);
    if (g == NULL) return NULL;
    NC_Cell *input = nc_graph_add_cell(g, NC_CELL_INPUT, NC_ACT_LINEAR);
    NC_Cell *res = nc_graph_add_cell(g, reservoir_type, NC_ACT_LINEAR);
    NC_Cell *out = nc_graph_add_cell(g, NC_CELL_OUTPUT, NC_ACT_LINEAR);
    if (input == NULL || res == NULL || out == NULL) return NULL;
    const size_t ws[2] = { input_dim, neurons };
    res->weights = nc_tensor_create(model, ws, 2, false);
    out->weights = nc_tensor_create(model, (size_t[2]){ neurons, output_dim }, 2, false);
    if (res->weights == NULL || out->weights == NULL) return NULL;
    nc_tensor_fill_random_normal(res->weights, 0.0f,
                                 sqrtf(1.0f / (float)input_dim));
    nc_tensor_fill_zeros(out->weights);
    if (spiking) {
        res->state.spiking.v_threshold = 0.3f;
        res->state.spiking.v_reset = 0.0f;
        res->state.spiking.decay_rate = 0.9f;
        res->state.spiking.refractory_steps = 1;
    }
    return (nc_graph_connect(g, 0, 1, NC_EDGE_FORWARD) &&
            nc_graph_connect(g, 1, 2, NC_EDGE_FORWARD)) ? g : NULL;
}

NC_Graph *nc_zoo_create_lsm(NC_Arena *model, size_t input_dim, size_t neurons,
                            size_t output_dim)
{
    return create_reservoir(model, NC_CELL_SPIKING_HIDDEN, input_dim,
                            neurons, output_dim, true);
}

NC_Graph *nc_zoo_create_elm(NC_Arena *model, size_t input_dim, size_t neurons,
                            size_t output_dim)
{
    return create_reservoir(model, NC_CELL_HIDDEN, input_dim, neurons,
                            output_dim, false);
}

NC_Graph *nc_zoo_create_esn(NC_Arena *model, size_t input_dim, size_t neurons,
                            size_t output_dim)
{
    NC_Graph *g = create_reservoir(model, NC_CELL_RECURRENT, input_dim,
                                   neurons, output_dim, false);
    if (g == NULL) return NULL;
    const size_t rs[2] = { neurons, neurons };
    g->cells[1].recurrent_weights = nc_tensor_create(model, rs, 2, false);
    if (g->cells[1].recurrent_weights == NULL) return NULL;
    nc_tensor_fill_random_normal(g->cells[1].recurrent_weights, 0.0f, 0.1f);
    return g;
}

NC_Graph *nc_zoo_create_kohonen(NC_Arena *model, size_t input_dim, size_t neurons)
{
    if (input_dim == 0 || neurons == 0) return NULL;
    NC_Graph *g = new_graph(model, 2, 1);
    if (g == NULL) return NULL;
    nc_graph_add_cell(g, NC_CELL_INPUT, NC_ACT_LINEAR);
    NC_Cell *map = nc_graph_add_cell(g, NC_CELL_HIDDEN, NC_ACT_LINEAR);
    if (map == NULL) return NULL;
    map->competitive = true;
    map->weights = nc_tensor_create(model, (size_t[2]){ neurons, input_dim }, 2, true);
    if (map->weights == NULL) return NULL;
    nc_tensor_fill_random_normal(map->weights, 0.0f, 0.1f);
    return nc_graph_connect(g, 0, 1, NC_EDGE_FORWARD) ? g : NULL;
}

NC_Graph *nc_zoo_create_rbf(NC_Arena *model, size_t input_dim, size_t centers,
                            size_t output_dim)
{
    if (input_dim == 0 || centers == 0 || output_dim == 0) return NULL;
    NC_Graph *g = new_graph(model, 3, 2);
    if (g == NULL) return NULL;
    nc_graph_add_cell(g, NC_CELL_INPUT, NC_ACT_LINEAR);
    NC_Cell *kernel = nc_graph_add_cell(g, NC_CELL_KERNEL, NC_ACT_LINEAR);
    NC_Cell *out = nc_graph_add_cell(g, NC_CELL_OUTPUT, NC_ACT_LINEAR);
    if (kernel == NULL || out == NULL) return NULL;
    kernel->weights = nc_tensor_create(model, (size_t[2]){ centers, input_dim }, 2, false);
    out->weights = nc_tensor_create(model, (size_t[2]){ centers, output_dim }, 2, false);
    if (kernel->weights == NULL || out->weights == NULL) return NULL;
    nc_tensor_fill_random_normal(kernel->weights, 0.0f, 1.0f);
    nc_tensor_fill_zeros(out->weights);
    return (nc_graph_connect(g, 0, 1, NC_EDGE_FORWARD) &&
            nc_graph_connect(g, 1, 2, NC_EDGE_FORWARD)) ? g : NULL;
}

NC_Graph *nc_zoo_create_svm(NC_Arena *model, size_t input_dim)
{
    if (input_dim == 0) return NULL;
    NC_Graph *g = new_graph(model, 2, 1);
    if (g == NULL) return NULL;
    nc_graph_add_cell(g, NC_CELL_INPUT, NC_ACT_LINEAR);
    NC_Cell *out = nc_graph_add_cell(g, NC_CELL_OUTPUT, NC_ACT_LINEAR);
    if (out == NULL || !nc_cell_init_params(out, model, input_dim, 1, true)) return NULL;
    nc_tensor_fill_zeros(out->weights);
    return nc_graph_connect(g, 0, 1, NC_EDGE_FORWARD) ? g : NULL;
}
