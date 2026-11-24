#include "nc_trainers.h"

#include <math.h>
#include <string.h>

void nc_train_backprop_step(NC_Graph *graph, NC_OptimizerAdam *opt)
{
    if (graph == NULL || opt == NULL || opt->learning_rate <= 0.0f ||
        opt->beta1 <= 0.0f || opt->beta1 >= 1.0f ||
        opt->beta2 <= 0.0f || opt->beta2 >= 1.0f || opt->epsilon <= 0.0f) {
        return;
    }

    nc_graph_ensure_adam_states(graph);

    for (size_t s = 0; s < graph->adam_count; ++s) {
        NC_AdamState *state = &graph->adam[s];
        NC_Tensor *param = state->param;
        if (param == NULL || param->grad == NULL || state->m == NULL ||
            state->v == NULL) {
            continue;
        }

        ++state->step;
        const float beta1_power = powf(opt->beta1, (float)state->step);
        const float beta2_power = powf(opt->beta2, (float)state->step);
        const float corr1 = 1.0f - beta1_power;
        const float corr2 = 1.0f - beta2_power;
        if (corr1 <= 0.0f || corr2 <= 0.0f) {
            continue;
        }

        for (size_t i = 0; i < param->size; ++i) {
            const float g = param->grad[i];
            const float m = opt->beta1 * state->m->data[i] +
                            (1.0f - opt->beta1) * g;
            const float v = opt->beta2 * state->v->data[i] +
                            (1.0f - opt->beta2) * g * g;
            state->m->data[i] = m;
            state->v->data[i] = v;

            const float m_hat = m / corr1;
            const float v_hat = v / corr2;
            param->data[i] -= opt->learning_rate * m_hat /
                              (sqrtf(v_hat) + opt->epsilon);
        }
    }
}


static float discriminator_step(NC_Graph *discriminator, const NC_Tensor *input,
                                const NC_Tensor *target, NC_Arena *scratch,
                                NC_OptimizerAdam *opt)
{
    NC_Tensor *pred = nc_graph_forward(discriminator, input, scratch);
    if (pred == NULL) {
        return NAN;
    }
    const float loss = nc_compute_mse_loss(pred, target);
    nc_autograd_record_mse(discriminator->tape, pred, target);
    nc_graph_backward(discriminator);
    if (opt != NULL) {
        nc_train_backprop_step(discriminator, opt);
    }
    return loss;
}

static size_t first_param_dim(const NC_Graph *graph)
{
    for (size_t i = 0; i < graph->cell_count; ++i) {
        if (graph->cells[i].weights != NULL) {
            return graph->cells[i].weights->shape[0];
        }
    }
    return 0;
}

void nc_train_gan_step(NC_Graph *generator, NC_Graph *discriminator,
                       NC_Tensor *real_data, NC_Arena *scratch,
                       NC_OptimizerAdam *opt)
{
    if (generator == NULL || discriminator == NULL || real_data == NULL ||
        scratch == NULL || opt == NULL || real_data->ndim != 2) {
        return;
    }
    const size_t batch = real_data->shape[0];
    const size_t data_dim = real_data->shape[1];
    const size_t latent_dim = first_param_dim(generator);
    if (batch == 0 || data_dim == 0 || latent_dim == 0) {
        return;
    }

    const size_t noise_shape[2] = { batch, latent_dim };
    const size_t target_shape[2] = { batch, 1 };
    NC_Tensor *noise = nc_tensor_create(scratch, noise_shape, 2, false);
    NC_Tensor *ones = nc_tensor_create(scratch, target_shape, 2, false);
    NC_Tensor *zeros = nc_tensor_create(scratch, target_shape, 2, false);
    NC_Tensor *fake_copy = nc_tensor_create(scratch, real_data->shape, 2, false);
    if (noise == NULL || ones == NULL || zeros == NULL || fake_copy == NULL) {
        return;
    }
    nc_tensor_fill_random_normal(noise, 0.0f, 1.0f);
    for (size_t i = 0; i < ones->size; ++i) {
        ones->data[i] = 1.0f;
    }

    /* 1) Discriminador com dados reais e com dados falsos (desacoplados). */
    discriminator_step(discriminator, real_data, ones, scratch, opt);

    NC_Tensor *fake = nc_graph_forward(generator, noise, scratch);
    if (fake == NULL || fake->size != fake_copy->size) {
        return;
    }
    memcpy(fake_copy->data, fake->data, fake->size * sizeof(float));
    discriminator_step(discriminator, fake_copy, zeros, scratch, opt);

    /* 2) Gerador: o erro no discriminador volta até os pesos do gerador,
     *     sem atualizar o discriminador nesta etapa. */
    NC_Tensor *fake_for_g = nc_graph_forward(generator, noise, scratch);
    if (fake_for_g == NULL) {
        return;
    }
    NC_Tensor *verdict = nc_graph_forward(discriminator, fake_for_g, scratch);
    if (verdict == NULL) {
        return;
    }
    nc_autograd_record_mse(discriminator->tape, verdict, ones);
    nc_graph_backward(discriminator); /* sem otimizar o discriminador */
    nc_graph_backward_no_zero(generator);
    nc_train_backprop_step(generator, opt);
}
