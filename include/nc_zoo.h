#ifndef NC_ZOO_H
#define NC_ZOO_H

#include "nc_topology.h"

/* Todas as fábricas alocam o grafo na model arena e retornam NULL em falha. */
NC_Graph *nc_zoo_create_perceptron(NC_Arena *model, size_t input_dim,
                                   size_t output_dim);
NC_Graph *nc_zoo_create_ff(NC_Arena *model, size_t input_dim,
                           size_t hidden_dim, size_t output_dim);
NC_Graph *nc_zoo_create_dff(NC_Arena *model, size_t input_dim,
                            const size_t hidden_dims[], size_t hidden_count,
                            size_t output_dim);
NC_Graph *nc_zoo_create_ae(NC_Arena *model, size_t input_dim, size_t hidden_dim);
NC_Graph *nc_zoo_create_dae(NC_Arena *model, size_t input_dim,
                            size_t hidden_dim, float noise_level);
NC_Graph *nc_zoo_create_sae(NC_Arena *model, size_t input_dim,
                            size_t hidden_dim, float sparsity_target,
                            float beta);
NC_Graph *nc_zoo_create_drn(NC_Arena *model, size_t input_dim,
                            const size_t hidden_dims[], size_t hidden_count,
                            size_t output_dim);

#endif /* NC_ZOO_H */
