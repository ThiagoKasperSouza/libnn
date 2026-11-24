#ifndef NC_SEQUENCE_H
#define NC_SEQUENCE_H

#include <stddef.h>

#include "nc_topology.h"

/*
 * Executa `steps` passos temporais acumulando todas as operações numa única
 * fita (BPTT). `x_seq` é um tensor contíguo [T, N, F]; o retorno é um vetor
 * de `steps` ponteiros para as saídas de cada passo, alocado em `scratch`.
 */
NC_Tensor **nc_sequence_forward(NC_Graph *graph, const NC_Tensor *x_seq,
                                size_t steps, NC_Arena *scratch);

/* Zera gradientes e propaga o erro por toda a fita desdobrada. */
void nc_sequence_backward(NC_Graph *graph);

/* Libera os estados recorrentes antes de uma nova sequência. */
void nc_sequence_reset_state(NC_Graph *graph);

#endif /* NC_SEQUENCE_H */
