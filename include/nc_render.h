#ifndef NC_RENDER_H
#define NC_RENDER_H

#include "nc_topology.h"

/*
 * nc_render_svg — Exporta a topologia de um NC_Graph para um arquivo SVG
 * vetorial no estilo do infográfico "A mostly complete chart of Neural
 * Networks" (Fjodor van Veen, Asimov Institute).
 *
 * Cada célula é desenhada como um círculo colorido segundo seu tipo
 * (amarelo=entrada, verde=oculta, vermelho=saída, azul=recorrente/memória,
 * rosa=convolucional), com marcadores de borda/triângulo para as variantes
 * da legenda. As arestas são linhas com setas indicando o fluxo de dados.
 *
 * Retorna 0 em sucesso, -1 em erro (arquivo não pode ser criado).
 */
int nc_render_svg(const NC_Graph *graph, const char *filename);

/*
 * nc_render_svg_string — Variante que escreve o SVG em um buffer fornecido
 * pelo chamador (útil para testes e uso em memória). Retorna o número de
 * bytes escritos (sem o '\0' final) ou -1 em erro. Se buffer for NULL,
 * retorna apenas o tamanho necessário.
 */
long nc_render_svg_string(const NC_Graph *graph, char *buffer, size_t buffer_size);

#endif /* NC_RENDER_H */