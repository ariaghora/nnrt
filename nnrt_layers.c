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

nnrt_Tensor* nnrt_conv_2d_layer_forward(nnrt_Conv2DLayer *l, nnrt_Tensor *x) {
    return nnrt_conv_2d(x, l->w, l->b, l->stride, l->pad);
}

void nnrt_conv_2d_layer_free(nnrt_Conv2DLayer *l) {
    nnrt_tensor_free(l->w);
    nnrt_tensor_free(l->b);
    free(l);
}

nnrt_ConvTranspose2DLayer *nnrt_conv_transpose_2d_fread(FILE *fp) {
    nnrt_ConvTranspose2DLayer *layer = (nnrt_ConvTranspose2DLayer *)malloc(sizeof(nnrt_ConvTranspose2DLayer));
    layer->w = nnrt_tensor_fread(fp);
    layer->b = nnrt_tensor_fread(fp);
    fread(&layer->stride, sizeof(int), 1, fp);
    fread(&layer->pad, sizeof(int), 1, fp);
    return layer;
}

nnrt_Tensor *nnrt_conv_transpose_2d_forward(nnrt_ConvTranspose2DLayer *l, nnrt_Tensor *x) {
    return nnrt_conv_transpose_2d(x, l->w, l->b, l->stride, l->pad);
}

void nnrt_conv_transpose_2d_free(nnrt_Conv2DLayer *l) {
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

nnrt_Tensor *nnrt_linear_layer_forward(nnrt_LinearLayer *l, nnrt_Tensor *x) {
    return nnrt_affine(x, l->w, l->b);
}

void nnrt_linear_layer_free(nnrt_LinearLayer *l) {
    nnrt_tensor_free(l->w);
    nnrt_tensor_free(l->b);
    free(l);
}
