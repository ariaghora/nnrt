#ifndef _NNRT_LAYERS_H_
#define _NNRT_LAYERS_H_

#include "nnrt.h"

typedef struct {
    nnrt_Tensor *w;
    nnrt_Tensor *b;
    int stride;
    int pad;
} nnrt_Conv2DLayer;

typedef struct {
    nnrt_Tensor *w;
    nnrt_Tensor *b;
    int stride;
    int pad;
} nnrt_ConvTranspose2DLayer;

typedef struct {
    nnrt_Tensor *w;
    nnrt_Tensor *b;
} nnrt_LinearLayer;

nnrt_Conv2DLayer *nnrt_conv_2d_layer_fread(FILE *fp);
nnrt_Tensor *nnrt_conv_2d_layer_forward(nnrt_Conv2DLayer *l, nnrt_Tensor *x);
void nnrt_conv_2d_layer_free(nnrt_Conv2DLayer *l);

nnrt_ConvTranspose2DLayer *nnrt_conv_transpose_2d_fread(FILE *fp);
nnrt_Tensor *nnrt_conv_transpose_2d_forward(nnrt_ConvTranspose2DLayer *l, nnrt_Tensor *x);
void nnrt_conv_transpose_2d_free(nnrt_Conv2DLayer *l);

nnrt_LinearLayer *nnrt_linear_layer_fread(FILE *fp);
nnrt_Tensor *nnrt_linear_layer_forward(nnrt_LinearLayer *l, nnrt_Tensor *x);
void nnrt_linear_layer_free(nnrt_LinearLayer *l);

#endif
