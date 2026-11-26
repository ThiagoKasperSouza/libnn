/*
 * render-graph.c — exporta a topologia de uma rede para SVG (Marco 6).
 * Build: make run_apps APP=render-graph
 */
#include <stdio.h>
#include <stdlib.h>

#include "neurochart.h"

int main(void)
{
    printf("=== libnn Marco 6: Render SVG ===\n");

    NC_Arena model = nc_arena_init(4u << 20);
    if (model.buffer == NULL) return EXIT_FAILURE;

    /* Rede de exemplo: autoencoder denoising */
    NC_Graph *g = nc_zoo_create_dae(&model, 8, 4, 0.2f);
    if (g == NULL) {
        nc_arena_free(&model);
        return EXIT_FAILURE;
    }

    const char *out = "dae_architecture.svg";
    if (nc_render_svg(g, out) != 0) {
        fprintf(stderr, "Falha ao escrever %s\n", out);
        nc_arena_free(&model);
        return EXIT_FAILURE;
    }
    printf("  SVG exportado: %s\n", out);

    /* Também exporta uma LSTM */
    NC_Graph *lstm = nc_zoo_create_lstm(&model, 4, 6, 2);
    if (lstm != NULL) {
        const char *out2 = "lstm_architecture.svg";
        if (nc_render_svg(lstm, out2) == 0)
            printf("  SVG exportado: %s\n", out2);
    }

    nc_graph_destroy(g);
    if (lstm != NULL) nc_graph_destroy(lstm);
    nc_arena_free(&model);
    printf("=== Concluído. ===\n");
    return EXIT_SUCCESS;
}