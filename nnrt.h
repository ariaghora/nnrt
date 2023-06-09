#ifndef _NNRT_H_
#define _NNRT_H_

#ifndef NNRT_FLOAT
#define NNRT_FLOAT float
#endif

#include <ctype.h>
#include <stdio.h>

typedef struct {
        NNRT_FLOAT *data;
        int *shape;
        int ndim;
} nnrt_Tensor;

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


nnrt_Tensor *nnrt_adaptive_avg_pool2d(nnrt_Tensor *a, int output_h,
                                      int output_w);
nnrt_Tensor *nnrt_affine(nnrt_Tensor *a, nnrt_Tensor *b, nnrt_Tensor *c);
nnrt_Tensor *nnrt_conv_2d(nnrt_Tensor *a, nnrt_Tensor *kernel,
                          nnrt_Tensor *bias, int stride, int pad);
nnrt_Tensor* nnrt_conv_transpose_2d(nnrt_Tensor *a, nnrt_Tensor *kernel,
                                    nnrt_Tensor *bias, int stride, int pad);
nnrt_Tensor *nnrt_matmul(nnrt_Tensor *a, nnrt_Tensor *b);
nnrt_Tensor *nnrt_maxpool_2d(nnrt_Tensor *a, int kernel, int stride, int pad);
void         nnrt_reshape_inplace(nnrt_Tensor *a, int *new_shape, int new_ndim);
void         nnrt_transpose_inplace(nnrt_Tensor *a);
void         nnrt_batchnorm_2d(nnrt_Tensor *a, int num_features, NNRT_FLOAT *gamma,
                               NNRT_FLOAT *shift, nnrt_Tensor *out);

//// Activation functions
void nnrt_relu(nnrt_Tensor *a, nnrt_Tensor *out);
void nnrt_sigmoid(nnrt_Tensor *a, nnrt_Tensor *out);
void nnrt_softmax(nnrt_Tensor *a, int axis, nnrt_Tensor *out);

//// Misc
nnrt_Tensor *nnrt_argmax(nnrt_Tensor *a, int axis);

//// Image processing functions
//   - It is assumed that images are in 4-dimension, [N, H, W, C]
nnrt_Tensor *nnrt_image_load(char *filename);
nnrt_Tensor *nnrt_image_resize(nnrt_Tensor *a, int new_h, int new_w);
nnrt_Tensor *nnrt_image_hwc_to_chw(nnrt_Tensor *a);
nnrt_Tensor *nnrt_image_chw_to_hwc(nnrt_Tensor *a);
void nnrt_image_standardize(nnrt_Tensor *a, NNRT_FLOAT *mean, NNRT_FLOAT *stddev,
                            nnrt_Tensor *out);
void nnrt_image_to_gray(nnrt_Tensor *a, nnrt_Tensor *out);

//// Memory managements
nnrt_Tensor *nnrt_tensor_alloc(int ndim, int *shape);
nnrt_Tensor *nnrt_tensor_fread(FILE *fp);
size_t       nnrt_tensor_size(nnrt_Tensor *t);
void         nnrt_tensor_free(nnrt_Tensor *t);

//// High-level layer ops
nnrt_Conv2DLayer *nnrt_conv_2d_layer_fread(FILE *fp);
nnrt_Tensor      *nnrt_conv_2d_layer_forward(nnrt_Conv2DLayer *l, nnrt_Tensor *x);
void              nnrt_conv_2d_layer_free(nnrt_Conv2DLayer *l);

nnrt_ConvTranspose2DLayer *nnrt_conv_transpose_2d_fread(FILE *fp);
nnrt_Tensor               *nnrt_conv_transpose_2d_forward(nnrt_ConvTranspose2DLayer *l, nnrt_Tensor *x);
void                       nnrt_conv_transpose_2d_free(nnrt_ConvTranspose2DLayer *l);

nnrt_LinearLayer *nnrt_linear_layer_fread(FILE *fp);
nnrt_Tensor      *nnrt_linear_layer_forward(nnrt_LinearLayer *l, nnrt_Tensor *x);
void              nnrt_linear_layer_free(nnrt_LinearLayer *l);

#endif
