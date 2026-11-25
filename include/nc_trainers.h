#ifndef NC_TRAINERS_H
#define NC_TRAINERS_H

#include "nc_autograd.h"
#include "nc_topology.h"

typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
} NC_OptimizerAdam;

/* Atualiza todos os parâmetros treináveis do grafo com Adam. Deve ser
 * chamado após nc_graph_forward() + nc_autograd_record_mse() + nc_graph_backward(). */
void nc_train_backprop_step(NC_Graph *graph, NC_OptimizerAdam *opt);

/*
 * Um passo minimax: treina o discriminador com dados reais (alvo 1) e falsos
 * (alvo 0), depois treina o gerador para enganar o discriminador (alvo 1).
 * `scratch` deve ser resetada pelo chamador no início de cada iteração.
 */
void nc_train_gan_step(NC_Graph *generator, NC_Graph *discriminator,
                       NC_Tensor *real_data, NC_Arena *scratch,
                       NC_OptimizerAdam *opt);

/* ---------------- Treinadores não-diferenciáveis (Marco 5) --------------- */

/* Divergência contrastiva CD-k para RBM/BM: W, viés oculto (bias) e viés
 * visível (recurrent_weights) são atualizados por amostragem de Gibbs. */
void nc_train_contrastive_divergence(NC_Graph *graph, NC_Tensor *input,
                                     size_t k_steps, float lr);

/* Hebbiano: ΔW += lr · (atividade pré · atividade pós) nas células com pesos. */
void nc_train_hebbian_step(NC_Graph *graph, float lr);

/* STDP: atualiza pesos spiking segundo a diferença de tempo pré/pós. */
void nc_train_stdp_step(NC_Graph *graph, float lr, float tau);

/* Ridge/pseudoinversa: resolve (ΦᵀΦ + αI)W = ΦᵀY para ELM/ESN/LSM/RBF. */
bool nc_train_reservoir_ridge_regression(NC_Graph *graph, NC_Tensor *states,
                                         NC_Tensor *targets, float alpha);

/* Kohonen/SOM: encontra o BMU e atualiza os protótipos da vizinhança. */
void nc_train_kohonen_bmu_step(NC_Graph *graph, NC_Tensor *input_sample,
                               float lr, float radius);

/* Hinge loss (SVM) com amostragem de margem; sinais em {-1,+1}. */
void nc_train_kernel_hinge_loss(NC_Graph *graph, NC_Tensor *X,
                                NC_Tensor *y, float C);

/* Reexportados de nc_autograd.h para conveniência (o header é incluído acima). */
float nc_compute_mse_loss(const NC_Tensor *pred, const NC_Tensor *target);
void nc_autograd_record_mse(NC_Tape *tape, const NC_Tensor *pred,
                            const NC_Tensor *target);

#endif /* NC_TRAINERS_H */
