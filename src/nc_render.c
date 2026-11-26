#include "nc_render.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Paleta do infográfico "A mostly complete chart of Neural Networks"  */
/* (Fjodor van Veen, Asimov Institute) — cores vivas e saturadas.      */
/* ------------------------------------------------------------------ */

#define NC_COLOR_INPUT   "#FFD700" /* amarelo */
#define NC_COLOR_HIDDEN  "#33CC33" /* verde   */
#define NC_COLOR_OUTPUT  "#FF6600" /* laranja */
#define NC_COLOR_STATE   "#3399FF" /* azul    */
#define NC_COLOR_KERNEL  "#FF66B2" /* rosa    */
#define NC_COLOR_OUTLINE "#333333"
#define NC_COLOR_LINK    "#666666"

static const char *cell_fill_color(NC_CellType type)
{
    switch (type) {
    case NC_CELL_BACKFED_INPUT:
    case NC_CELL_INPUT:
    case NC_CELL_NOISY_INPUT:
        return NC_COLOR_INPUT;
    case NC_CELL_HIDDEN:
    case NC_CELL_PROBABILISTIC_HIDDEN:
    case NC_CELL_SPIKING_HIDDEN:
        return NC_COLOR_HIDDEN;
    case NC_CELL_OUTPUT:
    case NC_CELL_MATCH_INPUT_OUTPUT:
        return NC_COLOR_OUTPUT;
    case NC_CELL_RECURRENT:
    case NC_CELL_MEMORY:
    case NC_CELL_DIFFERENT_MEMORY:
        return NC_COLOR_STATE;
    case NC_CELL_KERNEL:
    case NC_CELL_CONV_OR_POOL:
        return NC_COLOR_KERNEL;
    default:
        return "#CCCCCC";
    }
}

static int cell_has_inner_ring(NC_CellType type)
{
    switch (type) {
    case NC_CELL_BACKFED_INPUT:
    case NC_CELL_PROBABILISTIC_HIDDEN:
    case NC_CELL_MATCH_INPUT_OUTPUT:
    case NC_CELL_MEMORY:
    case NC_CELL_CONV_OR_POOL:
        return 1;
    default:
        return 0;
    }
}

static int cell_has_triangle(NC_CellType type)
{
    switch (type) {
    case NC_CELL_NOISY_INPUT:
    case NC_CELL_SPIKING_HIDDEN:
    case NC_CELL_DIFFERENT_MEMORY:
        return 1;
    default:
        return 0;
    }
}

/* ------------------------------------------------------------------ */
/* Helpers de escrita em buffer                                        */
/* ------------------------------------------------------------------ */

#define A0(s) do { \
        if (!truncated) { \
            int _r = snprintf(buf ? buf + pos : NULL, buf ? bufsize - pos : 0, "%s", s); \
            if (_r < 0 || (buf && (size_t)_r >= bufsize - pos)) { truncated = 1; if (buf) pos = bufsize; } \
            else pos += (size_t)_r; \
        } \
    } while (0)
#define AF(f, ...) do { \
        if (!truncated) { \
            int _r = snprintf(buf ? buf + pos : NULL, buf ? bufsize - pos : 0, f, ##__VA_ARGS__); \
            if (_r < 0 || (buf && (size_t)_r >= bufsize - pos)) { truncated = 1; if (buf) pos = bufsize; } \
            else pos += (size_t)_r; \
        } \
    } while (0)

/* Desenha um neurônio (círculo) no estilo da legenda do infográfico. */
static void append_neuron(char *buf, size_t bufsize, size_t *pos_ptr, int *tr_ptr,
                          double cx, double cy, NC_CellType type, double radius)
{
    /* Cópias locais para as macros A0/AF (que usam os nomes pos/truncated). */
    size_t pos = *pos_ptr;
    int truncated = *tr_ptr;
    const char *fill = cell_fill_color(type);
    int ring = cell_has_inner_ring(type);
    int tri = cell_has_triangle(type);

    AF("    <circle cx=\"%.1f\" cy=\"%.1f\" r=\"%.1f\" fill=\"%s\" stroke=\"%s\" stroke-width=\"1.4\"/>\n",
       cx, cy, radius, fill, NC_COLOR_OUTLINE);
    if (ring) {
        AF("    <circle cx=\"%.1f\" cy=\"%.1f\" r=\"%.1f\" fill=\"none\" stroke=\"%s\" stroke-width=\"1.4\"/>\n",
           cx, cy, radius * 0.55, NC_COLOR_OUTLINE);
    }
    if (tri) {
        double h = radius * 0.75;
        AF("    <polygon points=\"%.1f,%.1f %.1f,%.1f %.1f,%.1f\" fill=\"%s\"/>\n",
           cx, cy - h * 0.6, cx - h * 0.55, cy + h * 0.45, cx + h * 0.55, cy + h * 0.45,
           NC_COLOR_OUTLINE);
    }

#undef A0
#undef AF
    *pos_ptr = pos;
    *tr_ptr = truncated;
    /* Redefine as macros para o escopo seguinte. */
#define A0(s) do { \
        if (!truncated) { \
            int _r = snprintf(buf ? buf + pos : NULL, buf ? bufsize - pos : 0, "%s", s); \
            if (_r < 0 || (buf && (size_t)_r >= bufsize - pos)) { truncated = 1; if (buf) pos = bufsize; } \
            else pos += (size_t)_r; \
        } \
    } while (0)
#define AF(f, ...) do { \
        if (!truncated) { \
            int _r = snprintf(buf ? buf + pos : NULL, buf ? bufsize - pos : 0, f, ##__VA_ARGS__); \
            if (_r < 0 || (buf && (size_t)_r >= bufsize - pos)) { truncated = 1; if (buf) pos = bufsize; } \
            else pos += (size_t)_r; \
        } \
    } while (0)
}

static int cell_is_recurrent(NC_CellType type)
{
    switch (type) {
    case NC_CELL_BACKFED_INPUT:
    case NC_CELL_RECURRENT:
    case NC_CELL_MEMORY:
    case NC_CELL_DIFFERENT_MEMORY:
        return 1;
    default:
        return 0;
    }
}

/* ------------------------------------------------------------------ */
/* Layout: uma coluna por camada, neurônios empilhados verticalmente   */
/* ------------------------------------------------------------------ */

#define MAX_CELLS 4096
#define MAX_LAYERS 256
#define MAX_NEURONS_PER_LAYER 8
#define NEURON_R 14.0
#define NEURON_SPACING 36.0
#define LAYER_SPACING 95.0
#define MARGIN 55.0
#define TITLE_H 45.0
#define LEGEND_H 230.0

/* Número de neurônios representados por uma célula (fan-out). */
static int neuron_count(const NC_Graph *graph, size_t cell_id)
{
    const NC_Cell *c = &graph->cells[cell_id];
    if (c->weights != NULL && c->weights->ndim == 2) {
        int dim = (int)c->weights->shape[1];
        if (dim < 1) dim = 1;
        if (dim > MAX_NEURONS_PER_LAYER) dim = MAX_NEURONS_PER_LAYER;
        return dim;
    }
    /* Célula de entrada: usa a dimensão de entrada da próxima célula.
     * Células com portas (LSTM/GRU/NTM) guardam os pesos em gate_weights. */
    for (size_t e = 0; e < graph->edge_count; ++e) {
        if (graph->edges[e].src_cell_id == cell_id) {
            const NC_Cell *dst_cell = &graph->cells[graph->edges[e].dst_cell_id];
            const NC_Tensor *w = dst_cell->weights;
            if (w == NULL) w = dst_cell->gate_weights[0];
            if (w != NULL && w->ndim == 2) {
                int dim = (int)w->shape[0];
                if (dim < 1) dim = 1;
                if (dim > MAX_NEURONS_PER_LAYER) dim = MAX_NEURONS_PER_LAYER;
                return dim;
            }
        }
    }
    return 2;
}

static int compute_layers(const NC_Graph *graph, int *layer_of)
{
    size_t n = graph->cell_count;
    for (size_t i = 0; i < n; ++i) layer_of[i] = -1;

    int changed = 1;
    int max_layer = 0;

    for (size_t i = 0; i < n; ++i) {
        int has_pred = 0;
        for (size_t e = 0; e < graph->edge_count; ++e) {
            if (graph->edges[e].dst_cell_id == i) { has_pred = 1; break; }
        }
        if (!has_pred) layer_of[i] = 0;
    }

    int iterations = 0;
    while (changed && iterations < (int)n + 1) {
        changed = 0;
        for (size_t e = 0; e < graph->edge_count; ++e) {
            size_t src = graph->edges[e].src_cell_id;
            size_t dst = graph->edges[e].dst_cell_id;
            if (src >= n || dst >= n) continue;
            if (layer_of[src] >= 0 && layer_of[dst] < 0) {
                layer_of[dst] = layer_of[src] + 1;
                if (layer_of[dst] > max_layer) max_layer = layer_of[dst];
                changed = 1;
            } else if (layer_of[src] >= 0 && layer_of[dst] >= 0 &&
                       layer_of[src] + 1 > layer_of[dst]) {
                layer_of[dst] = layer_of[src] + 1;
                if (layer_of[dst] > max_layer) max_layer = layer_of[dst];
                changed = 1;
            }
        }
        iterations++;
    }

    for (size_t i = 0; i < n; ++i) {
        if (layer_of[i] < 0) layer_of[i] = 0;
    }
    return max_layer;
}

static long render_to_buffer(const NC_Graph *graph, char *buf, size_t bufsize)
{
    if (graph == NULL || graph->cell_count == 0) return -1;
    size_t n = graph->cell_count;
    if (n > MAX_CELLS) return -1;

    int layer_of[MAX_CELLS];
    int max_layer = compute_layers(graph, layer_of);
    if (max_layer >= MAX_LAYERS) max_layer = MAX_LAYERS - 1;

    /* neurônios por camada e por célula */
    int cells_in_layer[MAX_LAYERS];
    int neurons_in_layer[MAX_LAYERS];
    int cell_neurons[MAX_CELLS];
    for (int l = 0; l < MAX_LAYERS; ++l) {
        cells_in_layer[l] = 0;
        neurons_in_layer[l] = 0;
    }
    for (size_t i = 0; i < n; ++i) {
        int l = layer_of[i];
        cell_neurons[i] = neuron_count(graph, i);
        cells_in_layer[l]++;
        neurons_in_layer[l] += cell_neurons[i];
    }

    /* posição de cada célula dentro da sua camada */
    int cell_idx[MAX_CELLS];
    int seen[MAX_LAYERS];
    for (int l = 0; l < MAX_LAYERS; ++l) seen[l] = 0;
    for (size_t i = 0; i < n; ++i) cell_idx[i] = seen[layer_of[i]]++;

    /* altura máxima de uma camada (em neurônios) */
    int max_neurons = 1;
    for (int l = 0; l <= max_layer; ++l) {
        if (neurons_in_layer[l] > max_neurons) max_neurons = neurons_in_layer[l];
    }

    double graph_w = MARGIN * 2.0 + (max_layer + 1) * LAYER_SPACING;
    double graph_h = TITLE_H + MARGIN + (max_neurons - 1) * NEURON_SPACING + NEURON_R * 2.0;
    double canvas_w = graph_w;
    double canvas_h = graph_h + LEGEND_H;
    if (canvas_w < 400) canvas_w = 400;
    if (canvas_h < 400) canvas_h = 400;

    /* centros dos neurônios de cada célula */
    double cx[MAX_CELLS], cy[MAX_CELLS][MAX_NEURONS_PER_LAYER];
    for (size_t i = 0; i < n; ++i) {
        int l = layer_of[i];
        int cnt = cells_in_layer[l];
        /* base vertical desta célula dentro da coluna */
        int neurons_before = 0;
        for (int j = 0; j < cell_idx[i]; ++j) {
            /*cells antes na mesma camada - reconstrói soma */
        }
        /* soma dos neurônios das células anteriores na camada */
        neurons_before = 0;
        for (size_t j = 0; j < i; ++j) {
            if (layer_of[j] == l) neurons_before += cell_neurons[j];
        }
        int total = neurons_in_layer[l];
        double start_y = TITLE_H + MARGIN +
                         (max_neurons - total) * NEURON_SPACING / 2.0 + NEURON_R;
        cx[i] = MARGIN + l * LAYER_SPACING + NEURON_R;
        for (int k = 0; k < cell_neurons[i]; ++k) {
            cy[i][k] = start_y + (neurons_before + k) * NEURON_SPACING;
        }
        (void)cnt;
    }

    size_t pos = 0;
    int truncated = 0;

    A0("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    AF("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%.0f\" height=\"%.0f\" viewBox=\"0 0 %.0f %.0f\">\n",
       canvas_w, canvas_h, canvas_w, canvas_h);
    A0("  <rect width=\"100%\" height=\"100%\" fill=\"#FFFFFF\"/>\n");
    A0("  <defs>\n");
    A0("    <marker id=\"arrow\" markerWidth=\"8\" markerHeight=\"8\" refX=\"7\" refY=\"3\" orient=\"auto\" markerUnits=\"strokeWidth\">\n");
    AF("      <path d=\"M0,0 L0,6 L7,3 z\" fill=\"%s\"/>\n", NC_COLOR_LINK);
    A0("    </marker>\n");
    A0("  </defs>\n");

    AF("  <text x=\"%.1f\" y=\"%.0f\" text-anchor=\"middle\" font-family=\"Georgia, serif\" font-size=\"18\" font-weight=\"bold\" fill=\"#222222\">Neural Network Topology</text>\n",
       canvas_w / 2.0, 30.0);

    /* Arestas: conecta todos os neurônios de src a todos os de dst */
    for (size_t e = 0; e < graph->edge_count; ++e) {
        size_t src = graph->edges[e].src_cell_id;
        size_t dst = graph->edges[e].dst_cell_id;
        if (src >= n || dst >= n) continue;
        const char *dash = graph->edges[e].type == NC_EDGE_RECURRENT ? " stroke-dasharray=\"4,3\"" : "";
        for (int a = 0; a < cell_neurons[src]; ++a) {
            for (int b = 0; b < cell_neurons[dst]; ++b) {
                AF("    <line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"%s\" stroke-width=\"1\" marker-end=\"url(#arrow)\"%s/>\n",
                   cx[src], cy[src][a], cx[dst], cy[dst][b], NC_COLOR_LINK, dash);
            }
        }
    }

    /* Skip connections: primeira camada → última (como no guia LSTM) */
    if (max_layer >= 2) {
        for (size_t src = 0; src < n; ++src) {
            if (layer_of[src] != 0) continue;
            for (size_t dst = 0; dst < n; ++dst) {
                if (layer_of[dst] != max_layer) continue;
                for (int a = 0; a < cell_neurons[src]; ++a) {
                    for (int b = 0; b < cell_neurons[dst]; ++b) {
                        AF("    <line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"%s\" stroke-width=\"0.8\" stroke-dasharray=\"3,3\" marker-end=\"url(#arrow)\"/>\n",
                           cx[src], cy[src][a], cx[dst], cy[dst][b], NC_COLOR_LINK);
                    }
                }
            }
        }
    }

    /* Neurônios */
    for (size_t i = 0; i < n; ++i) {
        for (int k = 0; k < cell_neurons[i]; ++k) {
            append_neuron(buf, bufsize, &pos, &truncated, cx[i], cy[i][k],
                          graph->cells[i].type, NEURON_R);
        }
        /* Loops recorrentes (setas circulares acima das células de estado) */
        if (cell_is_recurrent(graph->cells[i].type)) {
            for (int k = 0; k < cell_neurons[i]; ++k) {
                double lx = cx[i];
                double ly = cy[i][k] - NEURON_R - 3.0;
                AF("    <path d=\"M %.1f,%.1f A 9,9 0 1,1 %.1f,%.1f\" fill=\"none\" stroke=\"%s\" stroke-width=\"1.2\" marker-end=\"url(#arrow)\"/>\n",
                   lx - 7.0, ly, lx + 7.0, ly, NC_COLOR_LINK);
            }
        }
    }

    /* Legenda — 13 variantes do infográfico, em 2 colunas */
    AF("  <text x=\"20\" y=\"%.0f\" font-family=\"sans-serif\" font-size=\"13\" font-weight=\"bold\" fill=\"#222222\">Legend</text>\n",
       graph_h + 24.0);
    static const NC_CellType legend_types[13] = {
        NC_CELL_BACKFED_INPUT, NC_CELL_INPUT, NC_CELL_NOISY_INPUT,
        NC_CELL_HIDDEN, NC_CELL_PROBABILISTIC_HIDDEN, NC_CELL_SPIKING_HIDDEN,
        NC_CELL_OUTPUT, NC_CELL_MATCH_INPUT_OUTPUT,
        NC_CELL_RECURRENT, NC_CELL_MEMORY, NC_CELL_DIFFERENT_MEMORY,
        NC_CELL_KERNEL, NC_CELL_CONV_OR_POOL
    };
    static const char *legend_names[13] = {
        "Backfed Input Cell", "Input Cell", "Noisy Input Cell",
        "Hidden Cell", "Probabilistic Hidden Cell", "Spiking Hidden Cell",
        "Output Cell", "Match Input Output Cell",
        "Recurrent Cell", "Memory Cell", "Different Memory Cell",
        "Kernel", "Convolution or Pool"
    };
    for (int i = 0; i < 13; ++i) {
        int col = i / 7;
        int row = i % 7;
        double lx = 36.0 + col * 200.0;
        double ly = graph_h + 48.0 + row * 26.0;
        append_neuron(buf, bufsize, &pos, &truncated, lx, ly, legend_types[i], 10.0);
        AF("    <text x=\"%.0f\" y=\"%.0f\" font-family=\"sans-serif\" font-size=\"10\" fill=\"#222222\">%s</text>\n",
           lx + 18.0, ly + 4.0, legend_names[i]);
    }

    A0("</svg>\n");

#undef A0
#undef AF

    if (truncated) return -1;
    if (buf && bufsize > 0) {
        if (pos < bufsize) buf[pos] = '\0';
        else buf[bufsize - 1] = '\0';
    }
    return (long)pos;
}

int nc_render_svg(const NC_Graph *graph, const char *filename)
{
    if (graph == NULL || filename == NULL) return -1;
    long size = render_to_buffer(graph, NULL, 0);
    if (size < 0) return -1;
    char *buf = (char *)malloc((size_t)size + 1);
    if (buf == NULL) return -1;
    long written = render_to_buffer(graph, buf, (size_t)size + 1);
    if (written < 0) { free(buf); return -1; }
    FILE *f = fopen(filename, "w");
    if (f == NULL) { free(buf); return -1; }
    size_t nw = fwrite(buf, 1, (size_t)written, f);
    fclose(f);
    free(buf);
    return (nw == (size_t)written) ? 0 : -1;
}

long nc_render_svg_string(const NC_Graph *graph, char *buffer, size_t buffer_size)
{
    if (graph == NULL) return -1;
    if (buffer == NULL) return render_to_buffer(graph, NULL, 0);
    return render_to_buffer(graph, buffer, buffer_size);
}