#ifndef _NNRT_LAYERS_H_
#define _NNRT_LAYERS_H_

#include "nnrt.h"

typedef struct {
    nnrt_Tensor *w;
    nnrt_Tensor *b;
    int stride;
    int pad;
} nnrt_Conv2DLayer;

nnrt_Conv2DLayer *nnrt_conv_2d_layer_fread(FILE *fp);
void nnrt_conv_2d_layer_forward(nnrt_Conv2DLayer *l, nnrt_Tensor *out);
void nnrt_conv_2d_layer_free(nnrt_Conv2DLayer *l);

#endif
