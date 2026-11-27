#include "test_common.h"
#include "neurochart.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_render_basic(void)
{
    NC_Arena model = nc_arena_init(4 * 1024 * 1024);
    NC_Arena scratch = nc_arena_init(4 * 1024 * 1024);

    NC_Graph *g = nc_zoo_create_ff(&model, 3, 4, 2);
    NC_CHECK(g != NULL);

    /* Tamanho necessário */
    long sz = nc_render_svg_string(g, NULL, 0);
    printf("  SVG tamanho: %ld bytes\n", sz);
    NC_CHECK(sz > 0);

    /* Renderiza em buffer */
    char *buf = (char *)malloc((size_t)sz + 1);
    NC_CHECK(buf != NULL);
    long written = nc_render_svg_string(g, buf, (size_t)sz + 1);
    NC_CHECK(written == sz);
    NC_CHECK(strstr(buf, "<svg") != NULL);
    NC_CHECK(strstr(buf, "</svg>") != NULL);
    NC_CHECK(strstr(buf, "#FFD700") != NULL); /* amarelo entrada */
    NC_CHECK(strstr(buf, "#33CC33") != NULL); /* verde oculta */
    NC_CHECK(strstr(buf, "#FF6600") != NULL); /* laranja saída */
    NC_CHECK(strstr(buf, "#3399FF") != NULL); /* azul recorrente */
    NC_CHECK(strstr(buf, "#FF66B2") != NULL); /* rosa kernel */
    /* Legenda completa com as 13 variantes */
    NC_CHECK(strstr(buf, "Backfed Input Cell") != NULL);
    NC_CHECK(strstr(buf, "Noisy Input Cell") != NULL);
    NC_CHECK(strstr(buf, "Probabilistic Hidden Cell") != NULL);
    NC_CHECK(strstr(buf, "Spiking Hidden Cell") != NULL);
    NC_CHECK(strstr(buf, "Match Input Output Cell") != NULL);
    NC_CHECK(strstr(buf, "Different Memory Cell") != NULL);
    NC_CHECK(strstr(buf, "Convolution or Pool") != NULL);

    /* Arquivo */
    const char *fname = "test_render_output.svg";
    NC_CHECK(nc_render_svg(g, fname) == 0);

    /* Verifica conteúdo do arquivo */
    FILE *f = fopen(fname, "r");
    NC_CHECK(f != NULL);
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    NC_CHECK(fsize == sz);
    fclose(f);

    free(buf);
    nc_arena_free(&scratch);
    nc_arena_free(&model);
}

static void test_render_all_types(void)
{
    NC_Arena model = nc_arena_init(8 * 1024 * 1024);
    /* Testa que cada tipo de célula renderiza sem crash */
    NC_CellType types[] = {
        NC_CELL_BACKFED_INPUT, NC_CELL_INPUT, NC_CELL_NOISY_INPUT,
        NC_CELL_HIDDEN, NC_CELL_PROBABILISTIC_HIDDEN, NC_CELL_SPIKING_HIDDEN,
        NC_CELL_OUTPUT, NC_CELL_MATCH_INPUT_OUTPUT, NC_CELL_RECURRENT,
        NC_CELL_MEMORY, NC_CELL_DIFFERENT_MEMORY, NC_CELL_KERNEL,
        NC_CELL_CONV_OR_POOL
    };
    /* Cria um grafo com uma célula de cada tipo encadeada */
    NC_Graph *g = nc_graph_create(&model, 14, 13, 256);
    NC_CHECK(g != NULL);
    for (int i = 0; i < 13; ++i) {
        NC_CHECK(nc_graph_add_cell(g, types[i], NC_ACT_LINEAR) != NULL);
    }
    for (int i = 0; i < 12; ++i) {
        NC_CHECK(nc_graph_connect(g, (size_t)i, (size_t)i + 1, NC_EDGE_FORWARD));
    }
    long sz = nc_render_svg_string(g, NULL, 0);
    NC_CHECK(sz > 0);
    char *buf = (char *)malloc((size_t)sz + 1);
    NC_CHECK(nc_render_svg_string(g, buf, (size_t)sz + 1) == sz);
    NC_CHECK(strstr(buf, "</svg>") != NULL);
    free(buf);

    nc_arena_free(&model);
}

static void test_render_null(void)
{
    NC_CHECK(nc_render_svg(NULL, "test_render_null.svg") == -1);
    NC_CHECK(nc_render_svg_string(NULL, NULL, 0) == -1);
}

int main(void)
{
    printf("=== test_render (Marco 6) ===\n");
    test_render_basic();
    test_render_all_types();
    test_render_null();
    NC_TEST_SUMMARY("test_render");
    return 0;
}