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

/* ------------------------- Topologias do Marco 3 ----------------------- */

/* Rede convolucional profunda: INPUT -> CONV -> POOL -> OUTPUT densa. */
NC_Graph *nc_zoo_create_dcn(NC_Arena *model, size_t channels, size_t h,
                            size_t w, size_t num_classes);

/* Rede deconvolucional: INPUT -> CONV_TRANSPOSE -> MATCH (upsample). */
NC_Graph *nc_zoo_create_dn(NC_Arena *model, size_t channels, size_t h, size_t w);

/* Autoencoder convolucional: INPUT -> CONV -> latente -> CONV_T -> MATCH. */
NC_Graph *nc_zoo_create_dcign(NC_Arena *model, size_t channels, size_t h,
                              size_t w, size_t latent_channels);

/* GAN: par gerador/discriminador acoplado para treino adversarial. */
typedef struct {
    NC_Graph *generator;
    NC_Graph *discriminator;
} NC_GAN;

NC_GAN *nc_zoo_create_gan(NC_Arena *model, size_t latent_dim, size_t data_dim);

/* ------------------------- Topologias do Marco 4 ----------------------- */

NC_Graph *nc_zoo_create_rnn(NC_Arena *model, size_t input_dim,
                            size_t hidden_dim, size_t output_dim);
NC_Graph *nc_zoo_create_lstm(NC_Arena *model, size_t input_dim,
                             size_t hidden_dim, size_t output_dim);
NC_Graph *nc_zoo_create_gru(NC_Arena *model, size_t input_dim,
                            size_t hidden_dim, size_t output_dim);
NC_Graph *nc_zoo_create_ntm(NC_Arena *model, size_t input_dim, size_t slots,
                            size_t mem_dim, size_t output_dim);

/* ------------------------- Topologias do Marco 5 ----------------------- */

NC_Graph *nc_zoo_create_markov_chain(NC_Arena *model, size_t states, size_t hidden);
NC_Graph *nc_zoo_create_hopfield(NC_Arena *model, size_t dim);
NC_Graph *nc_zoo_create_boltzmann(NC_Arena *model, size_t visible, size_t hidden);
NC_Graph *nc_zoo_create_rbm(NC_Arena *model, size_t visible, size_t hidden);
NC_Graph *nc_zoo_create_dbn(NC_Arena *model, size_t visible,
                            const size_t hidden_dims[], size_t layers);
NC_Graph *nc_zoo_create_vae(NC_Arena *model, size_t input_dim, size_t latent_dim);
NC_Graph *nc_zoo_create_lsm(NC_Arena *model, size_t input_dim, size_t neurons,
                            size_t output_dim);
NC_Graph *nc_zoo_create_elm(NC_Arena *model, size_t input_dim, size_t neurons,
                            size_t output_dim);
NC_Graph *nc_zoo_create_esn(NC_Arena *model, size_t input_dim, size_t neurons,
                            size_t output_dim);
NC_Graph *nc_zoo_create_kohonen(NC_Arena *model, size_t input_dim, size_t neurons);
NC_Graph *nc_zoo_create_rbf(NC_Arena *model, size_t input_dim, size_t centers,
                            size_t output_dim);
NC_Graph *nc_zoo_create_svm(NC_Arena *model, size_t input_dim);

#endif /* NC_ZOO_H */
