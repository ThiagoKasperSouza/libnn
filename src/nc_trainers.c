#include "nc_trainers.h"

#include <math.h>

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
