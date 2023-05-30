#include "nnrt_layers.h"

#include <stdlib.h>
#include <string.h>

#include "nnrt.h"

nnrt_Conv2DLayer *nnrt_conv_2d_layer_fread(FILE *fp) {
    nnrt_Conv2DLayer *layer = (nnrt_Conv2DLayer *)malloc(sizeof(nnrt_Conv2DLayer));
    layer->w = nnrt_tensor_fread(fp);
    layer->b = nnrt_tensor_fread(fp);
    fread(&layer->stride, sizeof(int), 1, fp);
    fread(&layer->pad, sizeof(int), 1, fp);
    return layer;
}

void nnrt_conv_2d_layer_forward(nnrt_Tensor *x, nnrt_Conv2DLayer *l, nnrt_Tensor *out) {
    nnrt_conv_2d(x, l->w, l->b, l->stride, l->pad, out);
}

void nnrt_conv_2d_layer_free(nnrt_Conv2DLayer *l) {
    nnrt_tensor_free(l->w);
    nnrt_tensor_free(l->b);
    free(l);
}

nnrt_LinearLayer *nnrt_linear_layer_fread(FILE *fp) {
    nnrt_LinearLayer *layer = (nnrt_LinearLayer *)malloc(sizeof(nnrt_LinearLayer));
    layer->w = nnrt_tensor_fread(fp);
    layer->b = nnrt_tensor_fread(fp);
    return layer;
}

void nnrt_linear_layer_forward(nnrt_Tensor *x, nnrt_LinearLayer *l, nnrt_Tensor *out) {
    nnrt_Tensor *wt = nnrt_tensor_alloc(l->w->ndim, l->w->shape);
    memcpy(wt->data, l->w->data, nnrt_tensor_size(l->w) * sizeof(NNRT_FLOAT));
    nnrt_transpose_inplace(wt);
    nnrt_affine(x, wt, l->b, out);
}

void nnrt_linear_layer_free(nnrt_LinearLayer *l) {
    nnrt_tensor_free(l->w);
    nnrt_tensor_free(l->b);
    free(l);
}
