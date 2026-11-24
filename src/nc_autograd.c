#include "nc_autograd.h"

#include <math.h>
#include <stdalign.h>
#include <stdlib.h>

static const double NC_SQRT_2PI = 2.506628274631000502415765284811;

NC_Tape *nc_autograd_tape_init(NC_Arena *arena, size_t capacity)
{
    if (arena == NULL || capacity == 0) {
        return NULL;
    }
    NC_Tape *tape = (NC_Tape *)nc_arena_alloc(arena, sizeof(*tape),
                                              alignof(NC_Tape));
    if (tape == NULL) {
        return NULL;
    }
    tape->nodes = (NC_TapeNode *)nc_arena_alloc(arena, capacity * sizeof(NC_TapeNode),
                                                alignof(NC_TapeNode));
    if (tape->nodes == NULL) {
        return NULL;
    }
    tape->count = 0;
    tape->capacity = capacity;
    tape->arena = arena;
    return tape;
}

void nc_autograd_tape_reset(NC_Tape *tape)
{
    if (tape != NULL) {
        tape->count = 0;
    }
}

void nc_autograd_record(NC_Tape *tape, NC_Tensor *in1, NC_Tensor *in2,
                        NC_Tensor *out, NC_BackwardFn fn, void *params)
{
    if (tape == NULL || out == NULL || fn == NULL || tape->count >= tape->capacity) {
        return;
    }
    NC_TapeNode *node = &tape->nodes[tape->count++];
    node->inputs[0] = in1;
    node->inputs[1] = in2;
    node->inputs[2] = NULL;
    node->output = out;
    node->backward_fn = fn;
    node->custom_params = params;
}

void nc_autograd_backward(NC_Tape *tape)
{
    if (tape == NULL) {
        return;
    }
    for (size_t i = tape->count; i-- > 0;) {
        tape->nodes[i].backward_fn(&tape->nodes[i]);
    }
}

static NC_OpParams *new_params(NC_Arena *scratch, NC_OpType op)
{
    NC_OpParams *p = (NC_OpParams *)nc_arena_alloc(scratch, sizeof(*p),
                                                   alignof(NC_OpParams));
    if (p != NULL) {
        p->op = op;
        p->target = NULL;
        p->mask = NULL;
        p->scale = 1.0f;
    }
    return p;
}

static void accumulate(NC_Tensor *t, size_t index, float value)
{
    if (t != NULL && t->grad != NULL) {
        t->grad[index] += value;
    }
}

/* Índice físico do i-ésimo elemento de um tensor 2D contíguo/em view. */
static size_t flat_index(const NC_Tensor *t, size_t i)
{
    const size_t cols = t->shape[1];
    return (i / cols) * t->strides[0] + (i % cols) * t->strides[1];
}

/* ------------------------------ Backward ------------------------------- */

static void matmul_backward(struct NC_TapeNode *node)
{
    const NC_Tensor *a = node->inputs[0];
    const NC_Tensor *b = node->inputs[1];
    const NC_Tensor *dy = node->output;
    if (a == NULL || b == NULL || dy == NULL || dy->grad == NULL) {
        return;
    }

    const size_t m = a->shape[0];
    const size_t k = a->shape[1];
    const size_t n = b->shape[1];
    const size_t as0 = a->strides[0], as1 = a->strides[1];
    const size_t bs0 = b->strides[0], bs1 = b->strides[1];
    const size_t ds0 = dy->strides[0], ds1 = dy->strides[1];

    /* dA[i,k] += sum_j dy[i,j] * B[k,j] */
    for (size_t i = 0; i < m; ++i) {
        for (size_t kk = 0; kk < k; ++kk) {
            float sum = 0.0f;
            for (size_t j = 0; j < n; ++j) {
                sum += dy->grad[i * ds0 + j * ds1] * b->data[kk * bs0 + j * bs1];
            }
            accumulate((NC_Tensor *)a, i * as0 + kk * as1, sum);
        }
    }

    /* dB[k,j] += sum_i A[i,k] * dy[i,j] */
    for (size_t kk = 0; kk < k; ++kk) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < m; ++i) {
                sum += a->data[i * as0 + kk * as1] * dy->grad[i * ds0 + j * ds1];
            }
            accumulate((NC_Tensor *)b, kk * bs0 + j * bs1, sum);
        }
    }
}

static void add_backward(struct NC_TapeNode *node)
{
    NC_Tensor *a = node->inputs[0];
    NC_Tensor *b = node->inputs[1];
    const NC_Tensor *dy = node->output;
    if (dy == NULL || dy->grad == NULL) {
        return;
    }

    const size_t rows = dy->shape[0];
    const size_t cols = dy->shape[1];
    const bool broadcast_b = b != NULL && b->shape[0] == 1 && rows != 1;
    const bool broadcast_a = a != NULL && a->shape[0] == 1 && rows != 1;

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const float g = dy->grad[i * dy->strides[0] + j * dy->strides[1]];
            if (a != NULL) {
                const size_t idx = (broadcast_a ? 0 : i) * a->strides[0] +
                                   j * a->strides[1];
                accumulate(a, idx, g);
            }
            if (b != NULL) {
                const size_t idx = (broadcast_b ? 0 : i) * b->strides[0] +
                                   j * b->strides[1];
                accumulate(b, idx, g);
            }
        }
    }
}

static void elementwise_backward(struct NC_TapeNode *node)
{
    NC_Tensor *x = node->inputs[0];
    const NC_Tensor *y = node->output;
    if (x == NULL || y == NULL || y->grad == NULL) {
        return;
    }
    const NC_OpParams *params = (const NC_OpParams *)node->custom_params;
    const NC_OpType op = params != NULL ? params->op : NC_OP_RELU;

    for (size_t i = 0; i < x->size; ++i) {
        const size_t xi = flat_index(x, i);
        const size_t yi = flat_index(y, i);
        const float xv = x->data[xi];
        float derivative = 0.0f;

        switch (op) {
        case NC_OP_RELU:
            derivative = xv > 0.0f ? 1.0f : 0.0f;
            break;
        case NC_OP_TANH: {
            const float t = y->data[yi];
            derivative = 1.0f - t * t;
            break;
        }
        case NC_OP_SIGMOID: {
            const float s = y->data[yi];
            derivative = s * (1.0f - s);
            break;
        }
        case NC_OP_GELU: {
            const double cdf = 0.5 * (1.0 + erf(xv / sqrt(2.0)));
            const double pdf = exp(-(double)xv * xv / 2.0) / NC_SQRT_2PI;
            derivative = (float)(cdf + xv * pdf);
            break;
        }
        default:
            derivative = 1.0f;
            break;
        }
        accumulate(x, xi, y->grad[yi] * derivative);
    }
}

static void dropout_backward(struct NC_TapeNode *node)
{
    NC_Tensor *x = node->inputs[0];
    const NC_Tensor *y = node->output;
    if (x == NULL || y == NULL || y->grad == NULL) {
        return;
    }
    const NC_OpParams *params = (const NC_OpParams *)node->custom_params;
    if (params == NULL || params->mask == NULL) {
        return;
    }
    for (size_t i = 0; i < x->size; ++i) {
        const size_t idx = flat_index(x, i);
        accumulate(x, idx, y->grad[flat_index(y, i)] *
                    params->mask->data[i] * params->scale);
    }
}

static void mse_backward(struct NC_TapeNode *node)
{
    NC_Tensor *pred = node->inputs[0];
    const NC_Tensor *target = node->inputs[1];
    if (pred == NULL || pred->grad == NULL || pred->size == 0 || target == NULL) {
        return;
    }
    const size_t count = pred->size < target->size ? pred->size : target->size;
    const float scale = 2.0f / (float)pred->size;
    for (size_t i = 0; i < count; ++i) {
        const size_t pi = flat_index(pred, i);
        accumulate(pred, pi,
                   scale * (pred->data[pi] - target->data[flat_index(target, i)]));
    }
}

/* ------------------------------ Forward -------------------------------- */

NC_Tensor *nc_nn_matmul(NC_Tape *tape, NC_Arena *scratch,
                        const NC_Tensor *A, const NC_Tensor *B)
{
    if (tape == NULL || A == NULL || B == NULL || scratch == NULL ||
        A->ndim != 2 || B->ndim != 2 || A->shape[1] != B->shape[0]) {
        return NULL;
    }
    const size_t shape[2] = { A->shape[0], B->shape[1] };
    NC_Tensor *out = nc_tensor_create(scratch, shape, 2, true);
    if (out == NULL) {
        return NULL;
    }
    nc_matmul(A, B, out);

    NC_OpParams *params = new_params(scratch, NC_OP_MATMUL);
    nc_autograd_record(tape, (NC_Tensor *)A, (NC_Tensor *)B, out,
                       matmul_backward, params);
    return out;
}

NC_Tensor *nc_nn_add(NC_Tape *tape, NC_Arena *scratch,
                     const NC_Tensor *A, const NC_Tensor *B)
{
    if (tape == NULL || A == NULL || B == NULL || scratch == NULL ||
        A->ndim != 2 || B->ndim != 2 || A->shape[1] != B->shape[1] ||
        (A->shape[0] != B->shape[0] && A->shape[0] != 1 && B->shape[0] != 1)) {
        return NULL;
    }
    const size_t rows = A->shape[0] > B->shape[0] ? A->shape[0] : B->shape[0];
    const size_t shape[2] = { rows, A->shape[1] };
    NC_Tensor *out = nc_tensor_create(scratch, shape, 2, true);
    if (out == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            const float av = A->data[(A->shape[0] == 1 ? 0 : i) * A->strides[0] +
                                     j * A->strides[1]];
            const float bv = B->data[(B->shape[0] == 1 ? 0 : i) * B->strides[0] +
                                     j * B->strides[1]];
            out->data[i * out->strides[0] + j * out->strides[1]] = av + bv;
        }
    }

    NC_OpParams *params = new_params(scratch, NC_OP_ADD);
    nc_autograd_record(tape, (NC_Tensor *)A, (NC_Tensor *)B, out,
                       add_backward, params);
    return out;
}

NC_Tensor *nc_nn_activation(NC_Tape *tape, NC_Arena *scratch,
                            const NC_Tensor *x, NC_OpType op)
{
    if (tape == NULL || x == NULL || scratch == NULL || x->ndim != 2) {
        return NULL;
    }
    NC_Tensor *out = nc_tensor_create(scratch, x->shape, x->ndim, true);
    if (out == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < x->size; ++i) {
        const size_t xi = flat_index(x, i);
        const size_t oi = flat_index(out, i);
        const float v = x->data[xi];
        float result = v;
        switch (op) {
        case NC_OP_RELU:
            result = v > 0.0f ? v : 0.0f;
            break;
        case NC_OP_TANH:
            result = (float)tanh(v);
            break;
        case NC_OP_SIGMOID:
            result = 1.0f / (1.0f + (float)exp(-v));
            break;
        case NC_OP_GELU:
            result = (float)(0.5 * v * (1.0 + erf(v / sqrt(2.0))));
            break;
        default:
            break;
        }
        out->data[oi] = result;
    }

    NC_OpParams *params = new_params(scratch, op);
    nc_autograd_record(tape, (NC_Tensor *)x, NULL, out,
                       elementwise_backward, params);
    return out;
}

NC_Tensor *nc_nn_dropout(NC_Tape *tape, NC_Arena *scratch,
                         const NC_Tensor *x, float keep_prob)
{
    if (tape == NULL || x == NULL || scratch == NULL || x->ndim != 2 ||
        keep_prob <= 0.0f || keep_prob > 1.0f) {
        return NULL;
    }
    NC_Tensor *out = nc_tensor_create(scratch, x->shape, x->ndim, true);
    NC_Tensor *mask = nc_tensor_create(scratch, x->shape, x->ndim, false);
    if (out == NULL || mask == NULL) {
        return NULL;
    }

    const float scale = 1.0f / keep_prob;
    for (size_t i = 0; i < x->size; ++i) {
        const float keep = nc_rng_next_double() < keep_prob ? 1.0f : 0.0f;
        mask->data[i] = keep;
        out->data[i] = x->data[flat_index(x, i)] * keep * scale;
    }

    NC_OpParams *params = new_params(scratch, NC_OP_DROPOUT);
    params->mask = mask;
    params->scale = scale;
    nc_autograd_record(tape, (NC_Tensor *)x, NULL, out,
                       dropout_backward, params);
    return out;
}

float nc_compute_mse_loss(const NC_Tensor *pred, const NC_Tensor *target)
{
    if (pred == NULL || target == NULL || pred->size == 0) {
        return 0.0f;
    }
    const size_t count = pred->size < target->size ? pred->size : target->size;
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double diff = (double)pred->data[flat_index(pred, i)] -
                            (double)target->data[flat_index(target, i)];
        sum += diff * diff;
    }
    return (float)(sum / (double)count);
}

void nc_autograd_record_mse(NC_Tape *tape, const NC_Tensor *pred,
                            const NC_Tensor *target)
{
    if (tape == NULL || pred == NULL || pred->size == 0 || target == NULL ||
        tape->count >= tape->capacity) {
        return;
    }
    /*
     * Sem alocação por iteração: o alvo viaja em inputs[1] e o fator 2/|pred|
     * é recalculado no backward. A saída do nó é a própria previsão.
     */
    nc_autograd_record(tape, (NC_Tensor *)pred, (NC_Tensor *)target,
                       (NC_Tensor *)pred, mse_backward, NULL);
}
