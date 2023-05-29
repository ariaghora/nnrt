#include "nnrt_layers.h"

#include <stdlib.h>

void nnrt_conv_2d_layer_free(nnrt_Conv2DLayer *l) {
    nnrt_tensor_free(l->w);
    nnrt_tensor_free(l->b);
    free(l);
}
