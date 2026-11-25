#include "nc_autograd.h"

#include <math.h>
#include <stdalign.h>
#include <stdlib.h>
#include <string.h>

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
    tape->overflow = false;
    return tape;
}

void nc_autograd_tape_reset(NC_Tape *tape)
{
    if (tape != NULL) {
        tape->count = 0;
        tape->overflow = false;
    }
}

bool nc_autograd_tape_overflow(const NC_Tape *tape)
{
    return tape != NULL && tape->overflow;
}

void nc_autograd_record(NC_Tape *tape, NC_Tensor *in1, NC_Tensor *in2,
                        NC_Tensor *out, NC_BackwardFn fn, void *params)
{
    if (tape == NULL || out == NULL || fn == NULL) {
        return;
    }
    if (tape->count >= tape->capacity) {
        tape->overflow = true; /* nunca descartar silenciosamente */
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
        p->pool_indices = NULL;
        p->scale = 1.0f;
        p->stride = 0;
        p->padding = 0;
        p->kernel_size = 0;
    }
    return p;
}

static void accumulate(NC_Tensor *t, size_t index, float value)
{
    if (t != NULL && t->grad != NULL) {
        t->grad[index] += value;
    }
}

/* Índice físico do i-ésimo elemento, para qualquer ndim (1–4). */
static size_t flat_index(const NC_Tensor *t, size_t i)
{
    size_t offset = 0;
    for (size_t d = t->ndim; d-- > 0;) {
        const size_t dim = t->shape[d];
        if (dim == 0) {
            return 0;
        }
        offset += (i % dim) * t->strides[d];
        i /= dim;
    }
    return offset;
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

static void identity_backward(struct NC_TapeNode *node)
{
    NC_Tensor *x = node->inputs[0];
    const NC_Tensor *y = node->output;
    if (x == NULL || y == NULL || y->grad == NULL) {
        return;
    }
    for (size_t i = 0; i < x->size; ++i) {
        accumulate(x, flat_index(x, i), y->grad[flat_index(y, i)]);
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

/* --------------- Backward elementwise (Marco 4) ------------------------ */

static void mul_backward(struct NC_TapeNode *node)
{
    NC_Tensor *a = node->inputs[0];
    NC_Tensor *b = node->inputs[1];
    const NC_Tensor *dy = node->output;
    if (a == NULL || b == NULL || dy == NULL || dy->grad == NULL) {
        return;
    }
    for (size_t i = 0; i < dy->size; ++i) {
        const float g = dy->grad[flat_index(dy, i)];
        accumulate(a, flat_index(a, i), g * b->data[flat_index(b, i)]);
        accumulate(b, flat_index(b, i), g * a->data[flat_index(a, i)]);
    }
}

static void scale_backward(struct NC_TapeNode *node)
{
    NC_Tensor *x = node->inputs[0];
    const NC_Tensor *dy = node->output;
    const NC_OpParams *p = (const NC_OpParams *)node->custom_params;
    if (x == NULL || dy == NULL || dy->grad == NULL || p == NULL) {
        return;
    }
    for (size_t i = 0; i < x->size; ++i) {
        accumulate(x, flat_index(x, i), p->scale * dy->grad[flat_index(dy, i)]);
    }
}

static void exp_backward(struct NC_TapeNode *node)
{
    NC_Tensor *x = node->inputs[0];
    const NC_Tensor *y = node->output;
    if (x == NULL || y == NULL || y->grad == NULL) return;
    for (size_t i = 0; i < x->size; ++i) {
        accumulate(x, flat_index(x, i),
                   y->grad[flat_index(y, i)] * y->data[flat_index(y, i)]);
    }
}

static void softmax_backward(struct NC_TapeNode *node)
{
    NC_Tensor *x = node->inputs[0];
    const NC_Tensor *y = node->output;
    if (x == NULL || y == NULL || y->grad == NULL) {
        return;
    }
    if (y->ndim != 2 || y->shape[1] <= 1) {
        double dot = 0.0;
        for (size_t i = 0; i < y->size; ++i) {
            dot += (double)y->grad[flat_index(y, i)] * y->data[flat_index(y, i)];
        }
        for (size_t i = 0; i < x->size; ++i) {
            const size_t yi = flat_index(y, i);
            accumulate(x, flat_index(x, i),
                       (float)(y->data[yi] * ((double)y->grad[yi] - dot)));
        }
        return;
    }

    /* Softmax 2D com mais de uma coluna é por linha; [slots,1] é global. */
    const size_t rows = y->shape[0];
    const size_t cols = y->shape[1];
    for (size_t r = 0; r < rows; ++r) {
        double dot = 0.0;
        for (size_t c = 0; c < cols; ++c) {
            const size_t yi = r * y->strides[0] + c * y->strides[1];
            dot += (double)y->grad[yi] * y->data[yi];
        }
        for (size_t c = 0; c < cols; ++c) {
            const size_t yi = r * y->strides[0] + c * y->strides[1];
            const size_t xi = r * x->strides[0] + c * x->strides[1];
            accumulate(x, xi, (float)(y->data[yi] * ((double)y->grad[yi] - dot)));
        }
    }
}

/* --------------------- Backward espacial (Marco 3) --------------------- */

static void conv2d_backward(struct NC_TapeNode *node)
{
    NC_Tensor *input = node->inputs[0];
    NC_Tensor *kernel = node->inputs[1];
    const NC_Tensor *dy = node->output;
    const NC_OpParams *p = (const NC_OpParams *)node->custom_params;
    if (input == NULL || kernel == NULL || dy == NULL || dy->grad == NULL ||
        p == NULL) {
        return;
    }
    const size_t stride = p->stride, pad = p->padding;
    const size_t c_out = kernel->shape[0];
    const size_t k_h = kernel->shape[2], k_w = kernel->shape[3];
    const size_t o_h = dy->shape[2], o_w = dy->shape[3];

    /* dKernel = corr(im2col(input), dy) */
    for (size_t oc = 0; oc < c_out; ++oc) {
        for (size_t ic = 0; ic < input->shape[1]; ++ic) {
            for (size_t kh = 0; kh < k_h; ++kh) {
                for (size_t kw = 0; kw < k_w; ++kw) {
                    float sum = 0.0f;
                    for (size_t n = 0; n < input->shape[0]; ++n) {
                        for (size_t oh = 0; oh < o_h; ++oh) {
                            const size_t ih = oh * stride + kh;
                            if (ih < pad || ih >= input->shape[2] + pad) {
                                continue;
                            }
                            for (size_t ow = 0; ow < o_w; ++ow) {
                                const size_t iw = ow * stride + kw;
                                if (iw < pad || iw >= input->shape[3] + pad) {
                                    continue;
                                }
                                const size_t d_off =
                                    n * dy->strides[0] + oc * dy->strides[1] +
                                    oh * dy->strides[2] + ow * dy->strides[3];
                                const size_t in_off =
                                    n * input->strides[0] + ic * input->strides[1] +
                                    (ih - pad) * input->strides[2] +
                                    (iw - pad) * input->strides[3];
                                sum += dy->grad[d_off] * input->data[in_off];
                            }
                        }
                    }
                    const size_t k_off =
                        oc * kernel->strides[0] + ic * kernel->strides[1] +
                        kh * kernel->strides[2] + kw * kernel->strides[3];
                    accumulate(kernel, k_off, sum);
                }
            }
        }
    }

    /* dInput = conv_transpose(dy, kernel), com um laço explícito por canal
     * de entrada: cada canal usa os filtros correspondentes do kernel. */
    for (size_t n = 0; n < input->shape[0]; ++n) {
        for (size_t ic = 0; ic < input->shape[1]; ++ic) {
            for (size_t oh = 0; oh < o_h; ++oh) {
                for (size_t ow = 0; ow < o_w; ++ow) {
                    for (size_t oc = 0; oc < c_out; ++oc) {
                        const size_t d_off =
                            n * dy->strides[0] + oc * dy->strides[1] +
                            oh * dy->strides[2] + ow * dy->strides[3];
                        const float g = dy->grad[d_off];
                        for (size_t kh = 0; kh < k_h; ++kh) {
                            const size_t ih = oh * stride + kh;
                            if (ih < pad || ih >= input->shape[2] + pad) {
                                continue;
                            }
                            for (size_t kw = 0; kw < k_w; ++kw) {
                                const size_t iw = ow * stride + kw;
                                if (iw < pad || iw >= input->shape[3] + pad) {
                                    continue;
                                }
                                const size_t k_off =
                                    oc * kernel->strides[0] +
                                    ic * kernel->strides[1] +
                                    kh * kernel->strides[2] +
                                    kw * kernel->strides[3];
                                const size_t in_off =
                                    n * input->strides[0] + ic * input->strides[1] +
                                    (ih - pad) * input->strides[2] +
                                    (iw - pad) * input->strides[3];
                                accumulate(input, in_off, g * kernel->data[k_off]);
                            }
                        }
                    }
                }
            }
        }
    }
}

static void conv_transpose_backward(struct NC_TapeNode *node)
{
    NC_Tensor *input = node->inputs[0];
    NC_Tensor *kernel = node->inputs[1];
    const NC_Tensor *dy = node->output;
    const NC_OpParams *p = (const NC_OpParams *)node->custom_params;
    if (input == NULL || kernel == NULL || dy == NULL || dy->grad == NULL ||
        p == NULL) {
        return;
    }
    const size_t stride = p->stride, pad = p->padding;
    const size_t c_out = kernel->shape[1];
    const size_t k_h = kernel->shape[2], k_w = kernel->shape[3];
    const size_t out_h = dy->shape[2], out_w = dy->shape[3];

    /* dInput = conv(dy, kernel, stride, padding) */
    for (size_t n = 0; n < input->shape[0]; ++n) {
        for (size_t ic = 0; ic < input->shape[1]; ++ic) {
            for (size_t ih = 0; ih < input->shape[2]; ++ih) {
                for (size_t iw = 0; iw < input->shape[3]; ++iw) {
                    for (size_t kh = 0; kh < k_h; ++kh) {
                        const size_t oh = ih * stride + kh;
                        if (oh < pad || oh >= out_h + pad) {
                            continue;
                        }
                        for (size_t kw = 0; kw < k_w; ++kw) {
                            const size_t ow = iw * stride + kw;
                            if (ow < pad || ow >= out_w + pad) {
                                continue;
                            }
                            for (size_t oc = 0; oc < c_out; ++oc) {
                                const size_t d_off =
                                    n * dy->strides[0] + oc * dy->strides[1] +
                                    (oh - pad) * dy->strides[2] +
                                    (ow - pad) * dy->strides[3];
                                const size_t in_off =
                                    n * input->strides[0] + ic * input->strides[1] +
                                    ih * input->strides[2] + iw * input->strides[3];
                                const size_t k_off =
                                    ic * kernel->strides[0] + oc * kernel->strides[1] +
                                    kh * kernel->strides[2] + kw * kernel->strides[3];
                                accumulate(input, in_off,
                                           dy->grad[d_off] * kernel->data[k_off]);
                            }
                        }
                    }
                }
            }
        }
    }

    /* dKernel = correlação entre input e dy. */
    for (size_t ic = 0; ic < input->shape[1]; ++ic) {
        for (size_t oc = 0; oc < c_out; ++oc) {
            for (size_t kh = 0; kh < k_h; ++kh) {
                for (size_t kw = 0; kw < k_w; ++kw) {
                    float sum = 0.0f;
                    for (size_t n = 0; n < input->shape[0]; ++n) {
                        for (size_t ih = 0; ih < input->shape[2]; ++ih) {
                            const size_t oh = ih * stride + kh;
                            if (oh < pad || oh >= out_h + pad) {
                                continue;
                            }
                            for (size_t iw = 0; iw < input->shape[3]; ++iw) {
                                const size_t ow = iw * stride + kw;
                                if (ow < pad || ow >= out_w + pad) {
                                    continue;
                                }
                                const size_t d_off =
                                    n * dy->strides[0] + oc * dy->strides[1] +
                                    (oh - pad) * dy->strides[2] +
                                    (ow - pad) * dy->strides[3];
                                const size_t in_off =
                                    n * input->strides[0] + ic * input->strides[1] +
                                    ih * input->strides[2] + iw * input->strides[3];
                                sum += dy->grad[d_off] * input->data[in_off];
                            }
                        }
                    }
                    const size_t k_off =
                        ic * kernel->strides[0] + oc * kernel->strides[1] +
                        kh * kernel->strides[2] + kw * kernel->strides[3];
                    accumulate(kernel, k_off, sum);
                }
            }
        }
    }
}

static void maxpool_backward(struct NC_TapeNode *node)
{
    NC_Tensor *input = node->inputs[0];
    const NC_Tensor *dy = node->output;
    const NC_OpParams *p = (const NC_OpParams *)node->custom_params;
    if (input == NULL || dy == NULL || dy->grad == NULL || p == NULL ||
        p->pool_indices == NULL) {
        return;
    }
    for (size_t i = 0; i < dy->size; ++i) {
        accumulate(input, p->pool_indices[i], dy->grad[i]);
    }
}

static void avgpool_backward(struct NC_TapeNode *node)
{
    NC_Tensor *input = node->inputs[0];
    const NC_Tensor *dy = node->output;
    const NC_OpParams *p = (const NC_OpParams *)node->custom_params;
    if (input == NULL || dy == NULL || dy->grad == NULL || p == NULL ||
        p->kernel_size == 0) {
        return;
    }
    const size_t k = p->kernel_size;
    const float scale = 1.0f / (float)(k * k);
    size_t index = 0;
    for (size_t n = 0; n < dy->shape[0]; ++n) {
        for (size_t c = 0; c < dy->shape[1]; ++c) {
            for (size_t oh = 0; oh < dy->shape[2]; ++oh) {
                for (size_t ow = 0; ow < dy->shape[3]; ++ow, ++index) {
                    const float g = dy->grad[index] * scale;
                    for (size_t kh = 0; kh < k; ++kh) {
                        for (size_t kw = 0; kw < k; ++kw) {
                            const size_t in_off =
                                n * input->strides[0] + c * input->strides[1] +
                                (oh * p->stride + kh) * input->strides[2] +
                                (ow * p->stride + kw) * input->strides[3];
                            accumulate(input, in_off, g);
                        }
                    }
                }
            }
        }
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
    if (tape == NULL || pred == NULL || pred->size == 0 || target == NULL) {
        return;
    }
    if (tape->count >= tape->capacity) {
        tape->overflow = true;
        return;
    }
    /*
     * Sem alocação por iteração: o alvo viaja em inputs[1] e o fator 2/|pred|
     * é recalculado no backward. A saída do nó é a própria previsão.
     */
    nc_autograd_record(tape, (NC_Tensor *)pred, (NC_Tensor *)target,
                       (NC_Tensor *)pred, mse_backward, NULL);
}

/* --------------------- Forward espacial (Marco 3) ---------------------- */

NC_Tensor *nc_nn_identity(NC_Tape *tape, NC_Arena *scratch,
                          const NC_Tensor *x)
{
    if (tape == NULL || scratch == NULL || x == NULL || x->ndim == 0 ||
        x->ndim > 4) {
        return NULL;
    }
    NC_Tensor *out = nc_tensor_create(scratch, x->shape, x->ndim, true);
    if (out == NULL) {
        return NULL;
    }
    memcpy(out->data, x->data, x->size * sizeof(float));
    nc_autograd_record(tape, (NC_Tensor *)x, NULL, out, identity_backward, NULL);
    return out;
}

NC_Tensor *nc_nn_conv2d(NC_Tape *tape, NC_Arena *scratch,
                        const NC_Tensor *input, const NC_Tensor *kernel,
                        size_t stride, size_t padding)
{
    size_t shape[4];
    if (tape == NULL || scratch == NULL ||
        !nc_conv2d_output_shape(input, kernel, stride, padding, shape)) {
        return NULL;
    }
    NC_Tensor *out = nc_tensor_create(scratch, shape, 4, true);
    if (out == NULL) {
        return NULL;
    }
    nc_conv2d_im2col(scratch, input, kernel, out, stride, padding);

    NC_OpParams *p = new_params(scratch, NC_OP_CONV2D);
    p->stride = stride;
    p->padding = padding;
    nc_autograd_record(tape, (NC_Tensor *)input, (NC_Tensor *)kernel, out,
                       conv2d_backward, p);
    return out;
}

NC_Tensor *nc_nn_conv2d_transpose(NC_Tape *tape, NC_Arena *scratch,
                                  const NC_Tensor *input, const NC_Tensor *kernel,
                                  size_t stride, size_t padding)
{
    size_t shape[4];
    if (tape == NULL || scratch == NULL ||
        !nc_conv2d_transpose_output_shape(input, kernel, stride, padding, shape)) {
        return NULL;
    }
    NC_Tensor *out = nc_tensor_create(scratch, shape, 4, true);
    if (out == NULL) {
        return NULL;
    }
    nc_conv2d_transpose_im2col(scratch, input, kernel, out, stride, padding);

    NC_OpParams *p = new_params(scratch, NC_OP_CONV_TRANSPOSE);
    p->stride = stride;
    p->padding = padding;
    nc_autograd_record(tape, (NC_Tensor *)input, (NC_Tensor *)kernel, out,
                       conv_transpose_backward, p);
    return out;
}

NC_Tensor *nc_nn_max_pool2d(NC_Tape *tape, NC_Arena *scratch,
                            const NC_Tensor *input, size_t kernel_size,
                            size_t stride)
{
    size_t shape[4];
    if (tape == NULL || scratch == NULL ||
        !nc_pool_output_shape(input, kernel_size, stride, shape)) {
        return NULL;
    }
    NC_Tensor *out = nc_tensor_create(scratch, shape, 4, true);
    if (out == NULL) {
        return NULL;
    }
    size_t *indices = (size_t *)nc_arena_alloc(scratch,
                                               out->size * sizeof(size_t),
                                               alignof(size_t));
    if (indices == NULL) {
        return NULL;
    }

    size_t index = 0;
    for (size_t n = 0; n < out->shape[0]; ++n) {
        for (size_t c = 0; c < out->shape[1]; ++c) {
            for (size_t oh = 0; oh < out->shape[2]; ++oh) {
                for (size_t ow = 0; ow < out->shape[3]; ++ow, ++index) {
                    float best = -INFINITY;
                    size_t best_off = 0;
                    for (size_t kh = 0; kh < kernel_size; ++kh) {
                        for (size_t kw = 0; kw < kernel_size; ++kw) {
                            const size_t in_off =
                                n * input->strides[0] + c * input->strides[1] +
                                (oh * stride + kh) * input->strides[2] +
                                (ow * stride + kw) * input->strides[3];
                            if (input->data[in_off] > best) {
                                best = input->data[in_off];
                                best_off = in_off;
                            }
                        }
                    }
                    out->data[index] = best;
                    indices[index] = best_off;
                }
            }
        }
    }

    NC_OpParams *p = new_params(scratch, NC_OP_MAXPOOL);
    p->pool_indices = indices;
    p->kernel_size = kernel_size;
    p->stride = stride;
    nc_autograd_record(tape, (NC_Tensor *)input, NULL, out,
                       maxpool_backward, p);
    return out;
}

NC_Tensor *nc_nn_avg_pool2d(NC_Tape *tape, NC_Arena *scratch,
                            const NC_Tensor *input, size_t kernel_size,
                            size_t stride)
{
    size_t shape[4];
    if (tape == NULL || scratch == NULL ||
        !nc_pool_output_shape(input, kernel_size, stride, shape)) {
        return NULL;
    }
    NC_Tensor *out = nc_tensor_create(scratch, shape, 4, true);
    if (out == NULL) {
        return NULL;
    }
    nc_avg_pool2d(input, out, kernel_size, stride);

    NC_OpParams *p = new_params(scratch, NC_OP_AVGPOOL);
    p->kernel_size = kernel_size;
    p->stride = stride;
    nc_autograd_record(tape, (NC_Tensor *)input, NULL, out,
                       avgpool_backward, p);
    return out;
}

NC_Tensor *nc_nn_mul(NC_Tape *tape, NC_Arena *scratch,
                     const NC_Tensor *A, const NC_Tensor *B)
{
    if (tape == NULL || scratch == NULL || A == NULL || B == NULL ||
        A->ndim != B->ndim || A->size != B->size) {
        return NULL;
    }
    for (size_t i = 0; i < A->ndim; ++i) {
        if (A->shape[i] != B->shape[i]) {
            return NULL;
        }
    }
    NC_Tensor *out = nc_tensor_create(scratch, A->shape, A->ndim, true);
    if (out == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < A->size; ++i) {
        out->data[i] = A->data[flat_index(A, i)] * B->data[flat_index(B, i)];
    }
    nc_autograd_record(tape, (NC_Tensor *)A, (NC_Tensor *)B, out,
                       mul_backward, NULL);
    return out;
}

NC_Tensor *nc_nn_scale(NC_Tape *tape, NC_Arena *scratch,
                       const NC_Tensor *x, float scale)
{
    if (tape == NULL || scratch == NULL || x == NULL || x->ndim == 0 ||
        x->ndim > 4) {
        return NULL;
    }
    NC_Tensor *out = nc_tensor_create(scratch, x->shape, x->ndim, true);
    if (out == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < x->size; ++i) {
        out->data[i] = scale * x->data[flat_index(x, i)];
    }
    NC_OpParams *p = new_params(scratch, NC_OP_SCALE);
    p->scale = scale;
    nc_autograd_record(tape, (NC_Tensor *)x, NULL, out, scale_backward, p);
    return out;
}

NC_Tensor *nc_nn_softmax(NC_Tape *tape, NC_Arena *scratch,
                         const NC_Tensor *x)
{
    if (tape == NULL || scratch == NULL || x == NULL || x->size == 0 ||
        x->ndim == 0 || x->ndim > 4) {
        return NULL;
    }
    NC_Tensor *out = nc_tensor_create(scratch, x->shape, x->ndim, true);
    if (out == NULL) {
        return NULL;
    }
    if (x->ndim == 2 && x->shape[1] > 1) {
        const size_t rows = x->shape[0];
        const size_t cols = x->shape[1];
        for (size_t r = 0; r < rows; ++r) {
            float max_value = -INFINITY;
            for (size_t c = 0; c < cols; ++c) {
                const float v = x->data[r * x->strides[0] + c * x->strides[1]];
                if (v > max_value) max_value = v;
            }
            double sum = 0.0;
            for (size_t c = 0; c < cols; ++c) {
                const double e = exp((double)x->data[r * x->strides[0] +
                                                     c * x->strides[1]] - max_value);
                out->data[r * out->strides[0] + c * out->strides[1]] = (float)e;
                sum += e;
            }
            if (sum <= 0.0) {
                return NULL;
            }
            for (size_t c = 0; c < cols; ++c) {
                const size_t oi = r * out->strides[0] + c * out->strides[1];
                out->data[oi] = (float)((double)out->data[oi] / sum);
            }
        }
    } else {
        float max_value = -INFINITY;
        for (size_t i = 0; i < x->size; ++i) {
            const float v = x->data[flat_index(x, i)];
            if (v > max_value) max_value = v;
        }
        double sum = 0.0;
        for (size_t i = 0; i < x->size; ++i) {
            const double e = exp((double)x->data[flat_index(x, i)] - max_value);
            out->data[i] = (float)e;
            sum += e;
        }
        if (sum <= 0.0) {
            return NULL;
        }
        for (size_t i = 0; i < out->size; ++i) {
            out->data[i] = (float)((double)out->data[i] / sum);
        }
    }
    nc_autograd_record(tape, (NC_Tensor *)x, NULL, out, softmax_backward, NULL);
    return out;
}

NC_Tensor *nc_nn_exp(NC_Tape *tape, NC_Arena *scratch, const NC_Tensor *x)
{
    if (tape == NULL || scratch == NULL || x == NULL || x->ndim == 0 ||
        x->ndim > 4) return NULL;
    NC_Tensor *out = nc_tensor_create(scratch, x->shape, x->ndim, true);
    if (out == NULL) return NULL;
    for (size_t i = 0; i < x->size; ++i) {
        out->data[i] = (float)exp((double)x->data[flat_index(x, i)]);
    }
    nc_autograd_record(tape, (NC_Tensor *)x, NULL, out, exp_backward, NULL);
    return out;
}

static void transpose_backward(struct NC_TapeNode *node)
{
    NC_Tensor *x = node->inputs[0];
    const NC_Tensor *y = node->output;
    if (x == NULL || y == NULL || y->grad == NULL || x->ndim != 2) {
        return;
    }
    const size_t rows = x->shape[0];
    const size_t cols = x->shape[1];
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const size_t yi = j * y->strides[0] + i * y->strides[1];
            const size_t xi = i * x->strides[0] + j * x->strides[1];
            accumulate(x, xi, y->grad[yi]);
        }
    }
}

NC_Tensor *nc_nn_transpose(NC_Tape *tape, NC_Arena *scratch,
                           const NC_Tensor *x)
{
    if (tape == NULL || scratch == NULL || x == NULL || x->ndim != 2) {
        return NULL;
    }
    NC_Tensor *out = (NC_Tensor *)nc_arena_alloc(scratch, sizeof(*out),
                                                 alignof(NC_Tensor));
    if (out == NULL) {
        return NULL;
    }
    *out = *x;
    out->shape[0] = x->shape[1];
    out->shape[1] = x->shape[0];
    out->strides[0] = x->strides[1];
    out->strides[1] = x->strides[0];
    out->grad = (float *)nc_arena_alloc(scratch, x->size * sizeof(float),
                                        NC_TENSOR_ALIGNMENT);
    if (out->grad == NULL) {
        return NULL;
    }
    memset(out->grad, 0, x->size * sizeof(float));
    out->requires_grad = true;
    nc_autograd_record(tape, (NC_Tensor *)x, NULL, out, transpose_backward, NULL);
    return out;
}
