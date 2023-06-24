#include <pthread.h>

#include "nnrt.h"

typedef struct {
    int batch;
    int c;
    nnrt_Tensor *a;
    nnrt_Tensor *kernel;
    nnrt_Tensor *bias;
    nnrt_Tensor *out;
    int stride;
    int pad;
    int output_channels;
    int output_height;
    int output_width;
    int input_channels;
    int input_height;
    int input_width;
    int kernel_height;
    int kernel_width;
} ConvTaskData;

void* conv_task_func(void* arg) {
    ConvTaskData* data = (ConvTaskData*)arg;
    int batch = data->batch;
    int c = data->c;

    // initialize entire output channel to bias
    for (int h = 0; h < data->output_height; h++) {
        for (int w = 0; w < data->output_width; w++) {
            int out_idx = batch * data->output_channels * data->output_height * data->output_width +
                          c * data->output_height * data->output_width +
                          h * data->output_width + w;
            data->out->data[out_idx] = data->bias->data[c];
        }
    }

    // convolution operation
    for (int i = 0; i < data->kernel_height; i++) {
        for (int j = 0; j < data->kernel_width; j++) {
            for (int k = 0; k < data->input_channels; k++) {
                int k_idx = c * data->input_channels * data->kernel_height * data->kernel_width +
                            k * data->kernel_height * data->kernel_width +
                            i * data->kernel_width + j;

                for (int h = 0; h < data->output_height; h++) {
                    for (int w = 0; w < data->output_width; w++) {
                        // calculate input height and width
                        int h_in = h * data->stride - data->pad + i;
                        int w_in = w * data->stride - data->pad + j;

                        // if within padded input dimensions
                        if (h_in >= 0 && h_in < data->input_height && w_in >= 0 && w_in < data->input_width) {
                            int in_idx = batch * data->input_channels * data->input_height * data->input_width +
                                         k * data->input_height * data->input_width +
                                         h_in * data->input_width + w_in;

                            // increase current output element by input multiplied by kernel
                            int out_idx = batch * data->output_channels * data->output_height * data->output_width +
                                          c * data->output_height * data->output_width +
                                          h * data->output_width + w;
                            data->out->data[out_idx] += data->a->data[in_idx] * data->kernel->data[k_idx];
                        }
                    }
                }
            }
        }
    }
    return NULL;
}

nnrt_Tensor* nnrt_conv_2d(nnrt_Tensor *a, nnrt_Tensor *kernel, nnrt_Tensor *bias,
                          int stride, int pad) {
    // Get input dimensions 
    int batch_size = a->shape[0];
    int input_channels = a->shape[1];
    int input_height = a->shape[2];
    int input_width = a->shape[3];

    // get kernel dimensions
    int kernel_height = kernel->shape[2];
    int kernel_width = kernel->shape[3];
    int output_channels = kernel->shape[0];

    // output dimensions
    int output_height = (input_height + 2 * pad - kernel_height) / stride + 1;
    int output_width = (input_width + 2 * pad - kernel_width) / stride + 1;
    nnrt_Tensor *out = nnrt_tensor_alloc(4, (int[]){batch_size, output_channels, output_height, output_width});

    pthread_t threads[batch_size][output_channels];
    ConvTaskData data[batch_size][output_channels];

    // loop over each element of the output tensor
    for (int batch = 0; batch < batch_size; batch++) {
        for (int c = 0; c < output_channels; c++) {
            data[batch][c].batch = batch;
            data[batch][c].c = c;
            data[batch][c].a = a;
            data[batch][c].kernel = kernel;
            data[batch][c].bias = bias;
            data[batch][c].out = out;
            data[batch][c].stride = stride;
            data[batch][c].pad = pad;
            data[batch][c].output_channels = output_channels;
            data[batch][c].output_height = output_height;
            data[batch][c].output_width = output_width;
            data[batch][c].input_channels = input_channels;
            data[batch][c].input_height = input_height;
            data[batch][c].input_width = input_width;
            data[batch][c].kernel_height = kernel_height;
            data[batch][c].kernel_width = kernel_width;
            pthread_create(&threads[batch][c], NULL, conv_task_func, &data[batch][c]);
        }
    }

    for (int batch = 0; batch < batch_size; batch++) {
        for (int c = 0; c < output_channels; c++) {
            pthread_join(threads[batch][c], NULL);
        }
    }

    return out;
}


