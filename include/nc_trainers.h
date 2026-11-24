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

/* Reexportados de nc_autograd.h para conveniência (o header é incluído acima). */
float nc_compute_mse_loss(const NC_Tensor *pred, const NC_Tensor *target);
void nc_autograd_record_mse(NC_Tape *tape, const NC_Tensor *pred,
                            const NC_Tensor *target);

#endif /* NC_TRAINERS_H */
