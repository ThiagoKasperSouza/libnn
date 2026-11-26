#include "nc_render.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Paleta de cores do infográfico (Fjodor van Veen, Asimov Institute) */
/* ------------------------------------------------------------------ */

static const char *cell_fill_color(NC_CellType type)
{
    switch (type) {
    case NC_CELL_BACKFED_INPUT:
    case NC_CELL_INPUT:
    case NC_CELL_NOISY_INPUT:
        return "#FFD700"; /* Amarelo — entrada */
    case NC_CELL_HIDDEN:
    case NC_CELL_PROBABILISTIC_HIDDEN:
    case NC_CELL_SPIKING_HIDDEN:
        return "#90EE90"; /* Verde — oculta */
    case NC_CELL_OUTPUT:
    case NC_CELL_MATCH_INPUT_OUTPUT:
        return "#FF6347"; /* Vermelho/laranja — saída */
    case NC_CELL_RECURRENT:
    case NC_CELL_MEMORY:
    case NC_CELL_DIFFERENT_MEMORY:
        return "#87CEEB"; /* Azul — recorrente/memória */
    case NC_CELL_KERNEL:
    case NC_CELL_CONV_OR_POOL:
        return "#FFB6C1"; /* Rosa — convolucional */
    default:
        return "#CCCCCC";
    }
}

static const char *cell_stroke_color(NC_CellType type)
{
    switch (type) {
    case NC_CELL_BACKFED_INPUT:
    case NC_CELL_NOISY_INPUT:
    case NC_CELL_PROBABILISTIC_HIDDEN:
    case NC_CELL_SPIKING_HIDDEN:
    case NC_CELL_MATCH_INPUT_OUTPUT:
    case NC_CELL_MEMORY:
    case NC_CELL_DIFFERENT_MEMORY:
    case NC_CELL_CONV_OR_POOL:
        return "#333333"; /* borda destacada para variantes */
    default:
        return "#555555";
    }
}

static int cell_has_border(NC_CellType type)
{
    switch (type) {
    case NC_CELL_BACKFED_INPUT:
    case NC_CELL_NOISY_INPUT:
    case NC_CELL_PROBABILISTIC_HIDDEN:
    case NC_CELL_SPIKING_HIDDEN:
    case NC_CELL_MATCH_INPUT_OUTPUT:
    case NC_CELL_MEMORY:
    case NC_CELL_DIFFERENT_MEMORY:
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

static const char *cell_label(NC_CellType type)
{
    switch (type) {
    case NC_CELL_BACKFED_INPUT:       return "BFI";
    case NC_CELL_INPUT:               return "IN";
    case NC_CELL_NOISY_INPUT:         return "NI";
    case NC_CELL_HIDDEN:              return "H";
    case NC_CELL_PROBABILISTIC_HIDDEN:return "PH";
    case NC_CELL_SPIKING_HIDDEN:      return "SH";
    case NC_CELL_OUTPUT:              return "OUT";
    case NC_CELL_MATCH_INPUT_OUTPUT:  return "MIO";
    case NC_CELL_RECURRENT:           return "R";
    case NC_CELL_MEMORY:              return "MEM";
    case NC_CELL_DIFFERENT_MEMORY:    return "DM";
    case NC_CELL_KERNEL:              return "K";
    case NC_CELL_CONV_OR_POOL:        return "CP";
    default:                          return "?";
    }
}

/* ------------------------------------------------------------------ */
/* Layout topológico simples: camadas por distância dos inputs         */
/* ------------------------------------------------------------------ */

#define MAX_CELLS 4096
#define MAX_LAYERS 256

typedef struct {
    int layer;       /* camada topológica (0 = input) */
    int index;       /* índice dentro da camada */
    int layer_count;  /* total de células na camada */
} CellLayout;

static int compute_layers(const NC_Graph *graph, int *layer_of)
{
    size_t n = graph->cell_count;
    /* Inicializa */
    for (size_t i = 0; i < n; ++i) layer_of[i] = -1;

    /* BFS a partir das células de entrada (sem arestas de entrada) */
    int changed = 1;
    int max_layer = 0;
    /* Células sem predecessoras são camada 0 */
    for (size_t i = 0; i < n; ++i) {
        int has_pred = 0;
        for (size_t e = 0; e < graph->edge_count; ++e) {
            if (graph->edges[e].dst_cell_id == i) { has_pred = 1; break; }
        }
        if (!has_pred) {
            layer_of[i] = 0;
        }
    }

    /* Propaga camadas (ignora arestas recorrentes para evitar ciclos) */
    int iterations = 0;
    while (changed && iterations < (int)n + 1) {
        changed = 0;
        for (size_t e = 0; e < graph->edge_count; ++e) {
            /* Ignora arestas recorrentes (criam ciclos) */
            if (graph->edges[e].type == NC_EDGE_RECURRENT) continue;
            size_t src = graph->edges[e].src_cell_id;
            size_t dst = graph->edges[e].dst_cell_id;
            if (src >= n || dst >= n) continue;
            if (layer_of[src] >= 0 && layer_of[dst] < 0) {
                layer_of[dst] = layer_of[src] + 1;
                if (layer_of[dst] > max_layer) max_layer = layer_of[dst];
                changed = 1;
            } else if (layer_of[src] >= 0 && layer_of[dst] >= 0) {
                if (layer_of[src] + 1 > layer_of[dst]) {
                    layer_of[dst] = layer_of[src] + 1;
                    if (layer_of[dst] > max_layer) max_layer = layer_of[dst];
                    changed = 1;
                }
            }
        }
        iterations++;
    }

    /* Células sem camada (isoladas) → camada 0 */
    for (size_t i = 0; i < n; ++i) {
        if (layer_of[i] < 0) layer_of[i] = 0;
    }

    return max_layer;
}

/* ------------------------------------------------------------------ */
/* Renderização SVG                                                    */
/* ------------------------------------------------------------------ */

#define CELL_RADIUS 22.0
#define LAYER_SPACING 120.0
#define CELL_SPACING 70.0
#define MARGIN 40.0

static long render_to_buffer(const NC_Graph *graph, char *buf, size_t bufsize)
{
    if (graph == NULL || graph->cell_count == 0) return -1;

    size_t n = graph->cell_count;

    int layer_of[MAX_CELLS];
    if (n > MAX_CELLS) return -1;
    int max_layer = compute_layers(graph, layer_of);

    /* Conta células por camada (com guard contra OOB) */
    int layer_counts[MAX_LAYERS];
    for (int i = 0; i < MAX_LAYERS; ++i) layer_counts[i] = 0;
    for (size_t i = 0; i < n; ++i) {
        if (layer_of[i] >= 0 && layer_of[i] < MAX_LAYERS)
            layer_counts[layer_of[i]]++;
        else
            layer_of[i] = 0; /* fallback seguro */
    }
    if (max_layer >= MAX_LAYERS) max_layer = MAX_LAYERS - 1;

    /* Posições */
    double px[MAX_CELLS], py[MAX_CELLS];
    /* Índice dentro da camada */
    int layer_idx[MAX_CELLS];
    for (size_t i = 0; i < n; ++i) layer_idx[i] = 0;

    /* Primeira passagem: atribui índice */
    int seen[MAX_LAYERS];
    for (int i = 0; i <= max_layer && i < MAX_LAYERS; ++i) seen[i] = 0;
    for (size_t i = 0; i < n; ++i) {
        int l = layer_of[i];
        layer_idx[i] = seen[l]++;
    }

    /* Calcula dimensões do canvas */
    double max_layer_height = 0;
    for (int l = 0; l <= max_layer && l < MAX_LAYERS; ++l) {
        double h = (double)layer_counts[l] * CELL_SPACING;
        if (h > max_layer_height) max_layer_height = h;
    }
    double canvas_w = MARGIN * 2 + (max_layer + 1) * LAYER_SPACING;
    /* Garante espaço para a legenda (5 itens x 130px) */
    double legend_w = 5 * 130.0 + MARGIN * 2;
    if (legend_w > canvas_w) canvas_w = legend_w;
    double canvas_h = MARGIN * 2 + max_layer_height + 70; /* espaço para legenda */
    if (canvas_w < 200) canvas_w = 200;
    if (canvas_h < 200) canvas_h = 200;

    /* Posiciona células */
    for (size_t i = 0; i < n; ++i) {
        int l = layer_of[i];
        int cnt = layer_counts[l];
        double x = MARGIN + l * LAYER_SPACING + CELL_RADIUS;
        double total_h = (double)(cnt - 1) * CELL_SPACING;
        double start_y = MARGIN + (max_layer_height - total_h) / 2.0 + CELL_RADIUS;
        double y = start_y + layer_idx[i] * CELL_SPACING;
        px[i] = x;
        py[i] = y;
    }

    /* Escreve SVG */
    /* Usamos um buffer dinâmico via fwrite-like approach com snprintf */
    /* Para simplicidade, escrevemos em um FILE* se buf==NULL, ou em buf. */
    /* Esta função sempre escreve em buf. */

    size_t pos = 0;
    int truncated = 0;

#define APPEND0(str) do { \
        if (!truncated) { \
            int _r = snprintf(buf ? buf + pos : NULL, buf ? bufsize - pos : 0, "%s", str); \
            if (_r < 0) { truncated = 1; } \
            else if (buf && (size_t)_r >= bufsize - pos) { truncated = 1; pos = bufsize; } \
            else { pos += (size_t)_r; } \
        } \
    } while(0)
#define APPEND(fmt, ...) do { \
        if (!truncated) { \
            int _r = snprintf(buf ? buf + pos : NULL, buf ? bufsize - pos : 0, fmt, ##__VA_ARGS__); \
            if (_r < 0) { truncated = 1; } \
            else if (buf && (size_t)_r >= bufsize - pos) { truncated = 1; pos = bufsize; } \
            else { pos += (size_t)_r; } \
        } \
    } while(0)

    APPEND0("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    APPEND("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%.0f\" height=\"%.0f\" viewBox=\"0 0 %.0f %.0f\">\n",
           canvas_w, canvas_h, canvas_w, canvas_h);
    APPEND0("  <rect width=\"100%\" height=\"100%\" fill=\"#FFFFFF\"/>\n");
    APPEND0("  <defs>\n");
    APPEND0("    <marker id=\"arrow\" markerWidth=\"10\" markerHeight=\"10\" refX=\"9\" refY=\"3\" orient=\"auto\" markerUnits=\"strokeWidth\">\n");
    APPEND0("      <path d=\"M0,0 L0,6 L9,3 z\" fill=\"#555555\"/>\n");
    APPEND0("    </marker>\n");
    APPEND0("  </defs>\n");

    /* Arestas (desenhadas primeiro, atrás dos nós) */
    for (size_t e = 0; e < graph->edge_count; ++e) {
        size_t src = graph->edges[e].src_cell_id;
        size_t dst = graph->edges[e].dst_cell_id;
        if (src >= n || dst >= n) continue;
        APPEND("  <line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"#555555\" stroke-width=\"1.5\" marker-end=\"url(#arrow)\"/>\n",
               px[src], py[src], px[dst], py[dst]);
    }

    /* Nós */
    for (size_t i = 0; i < n; ++i) {
        NC_CellType t = graph->cells[i].type;
        const char *fill = cell_fill_color(t);
        const char *stroke = cell_stroke_color(t);
        int border = cell_has_border(t);
        int tri = cell_has_triangle(t);
        const char *label = cell_label(t);

        APPEND("  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"%.1f\" fill=\"%s\" stroke=\"%s\" stroke-width=\"%s\"/>\n",
               px[i], py[i], CELL_RADIUS, fill, stroke, border ? "3" : "1.5");

        /* Marcador de triângulo para variantes "com triângulo" */
        if (tri) {
            double r = CELL_RADIUS * 0.5;
            double cx = px[i];
            double cy = py[i] - CELL_RADIUS * 0.3;
            APPEND("  <polygon points=\"%.1f,%.1f %.1f,%.1f %.1f,%.1f\" fill=\"#333333\"/>\n",
                   cx, cy - r, cx - r * 0.866, cy + r * 0.5, cx + r * 0.866, cy + r * 0.5);
        }

        /* Marcador de borda (anel) para variantes "com borda" */
        if (border) {
            APPEND("  <circle cx=\"%.1f\" cy=\"%.1f\" r=\"%.1f\" fill=\"none\" stroke=\"#333333\" stroke-width=\"1\" stroke-dasharray=\"3,2\"/>\n",
                   px[i], py[i], CELL_RADIUS + 4);
        }

        /* Label */
        APPEND("  <text x=\"%.1f\" y=\"%.1f\" text-anchor=\"middle\" dominant-baseline=\"central\" font-family=\"sans-serif\" font-size=\"10\" font-weight=\"bold\" fill=\"#333333\">%s</text>\n",
               px[i], py[i] + CELL_RADIUS + 12, label);
    }

    /* Legenda */
    APPEND("  <g transform=\"translate(10, %.0f)\">\n", canvas_h - 65);
    APPEND0("    <text x=\"0\" y=\"0\" font-family=\"sans-serif\" font-size=\"11\" font-weight=\"bold\" fill=\"#333333\">Legenda:</text>\n");
    const char *legend_items[] = {
        "Entrada (amarelo)", "Oculta (verde)", "Saída (vermelho)",
        "Recorrente (azul)", "Convolucional (rosa)"
    };
    const char *legend_colors[] = {
        "#FFD700", "#90EE90", "#FF6347", "#87CEEB", "#FFB6C1"
    };
    for (int i = 0; i < 5; ++i) {
        double lx = i * 130.0;
        APPEND("    <circle cx=\"%.0f\" cy=\"18\" r=\"6\" fill=\"%s\" stroke=\"#555555\" stroke-width=\"1\"/>\n",
               lx + 6, legend_colors[i]);
        APPEND("    <text x=\"%.0f\" y=\"22\" font-family=\"sans-serif\" font-size=\"9\" fill=\"#333333\">%s</text>\n",
               lx + 16, legend_items[i]);
    }
    APPEND0("  </g>\n");

    APPEND0("</svg>\n");

#undef APPEND

    if (truncated) return -1;
    /* pos inclui o '\0' implícito? snprintf não adiciona '\0' no limite exato;
       garantimos terminação. */
    if (buf && bufsize > 0) {
        if (pos < bufsize) buf[pos] = '\0';
        else buf[bufsize - 1] = '\0';
    }
    return (long)pos;
}

int nc_render_svg(const NC_Graph *graph, const char *filename)
{
    if (graph == NULL || filename == NULL) return -1;

    /* Primeira passagem: tamanho necessário */
    long size = render_to_buffer(graph, NULL, 0);
    if (size < 0) return -1;

    char *buf = (char *)malloc((size_t)size + 1);
    if (buf == NULL) return -1;

    long written = render_to_buffer(graph, buf, (size_t)size + 1);
    if (written < 0) {
        free(buf);
        return -1;
    }

    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        free(buf);
        return -1;
    }
    size_t nw = fwrite(buf, 1, (size_t)written, f);
    fclose(f);
    free(buf);

    if (nw != (size_t)written) return -1;
    return 0;
}

long nc_render_svg_string(const NC_Graph *graph, char *buffer, size_t buffer_size)
{
    if (graph == NULL) return -1;
    if (buffer == NULL) {
        /* Retorna tamanho necessário */
        return render_to_buffer(graph, NULL, 0);
    }
    return render_to_buffer(graph, buffer, buffer_size);
}