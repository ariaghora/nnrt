#include <math.h>
#include <pthread.h>

#include "nnrt.h"

#include <pthread.h>

typedef struct {
    size_t channel;
    nnrt_Tensor *a;
    nnrt_Tensor *out;
    int kernel_size;
    int stride;
    int pad;
    size_t batch_size;
    size_t num_channels;
    size_t height;
    size_t width;
    size_t outH;
    size_t outW;
} PoolTaskData;

void* pool_task_func(void* arg) {
    PoolTaskData* data = (PoolTaskData*)arg;
    size_t channel = data->channel;

    for (size_t batch = 0; batch < data->batch_size; ++batch) {
        for (size_t i = 0; i < data->outH; ++i) {
            for (size_t j = 0; j < data->outW; ++j) {
                NNRT_FLOAT max_val = -MAXFLOAT;
                for (size_t ki = 0; ki < data->kernel_size; ++ki) {
                    for (size_t kj = 0; kj < data->kernel_size; ++kj) {
                        long in_i = data->stride * i + ki - data->pad;  // use signed integer type
                        long in_j = data->stride * j + kj - data->pad;  // use signed integer type
                        if (in_i >= 0 && in_j >= 0 && in_i < data->height && in_j < data->width) {
                            float val = data->a->data[batch * data->num_channels * data->height * data->width +
                                                      channel * data->height * data->width +
                                                      in_i * data->width + in_j];
                            max_val = fmaxf(max_val, val);
                        }
                    }
                }
                int didx = batch * data->num_channels * data->outH * data->outW +
                           channel * data->outH * data->outW + i * data->outW + j;
                data->out->data[didx] = max_val;
            }
        }
    }
    return NULL;
}

nnrt_Tensor *nnrt_maxpool_2d(nnrt_Tensor *a, int kernel_size,
                             int stride, int pad) {
    size_t batch_size = a->shape[0], num_channels = a->shape[1], height = a->shape[2], width = a->shape[3];
    size_t kH = kernel_size, kW = kernel_size;
    size_t outH = (height - kH + 2 * pad) / stride + 1;
    size_t outW = (width - kW + 2 * pad) / stride + 1;

    nnrt_Tensor *out = nnrt_tensor_alloc(4, (int[]){batch_size, num_channels, outW, outH});

    pthread_t threads[num_channels];
    PoolTaskData data[num_channels];

    for (size_t channel = 0; channel < num_channels; ++channel) {
        data[channel].channel = channel;
        data[channel].a = a;
        data[channel].out = out;
        data[channel].kernel_size = kernel_size;
        data[channel].stride = stride;
        data[channel].pad = pad;
        data[channel].batch_size = batch_size;
        data[channel].num_channels = num_channels;
        data[channel].height = height;
        data[channel].width = width;
        data[channel].outH = outH;
        data[channel].outW = outW;
        pthread_create(&threads[channel], NULL, pool_task_func, &data[channel]);
    }

    for (size_t channel = 0; channel < num_channels; ++channel) {
        pthread_join(threads[channel], NULL);
    }

    return out;
}
