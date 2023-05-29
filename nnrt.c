#include "nnrt.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

inline void nnrt_affine(nnrt_Tensor *a, nnrt_Tensor *b, nnrt_Tensor *c, nnrt_Tensor *out) {
    // A * B + c
    // A: [m, p]
    // B: [p, n]
    // c: [1, n]
    nnrt_matmul(a, b, out);

    if (c->ndim != 2) {
        printf("c must be of dimension 2\n");
        exit(1);
    }

    if (c->shape[0] != 1) {
        printf("leading dimension of c must be 1, got %d\n", c->shape[0]);
        exit(1);
    }

    if (b->shape[1] != c->shape[1]) {
        printf("cannot add intecept on tensor with incompatible shape: [m, %d], [1, %d]\n",
               b->shape[1], c->shape[1]);
        exit(1);
    }

    for (size_t i = 0; i < a->shape[0]; ++i) {
        for (size_t j = 0; j < b->shape[1]; ++j) {
            out->data[i * b->shape[1] + j] += c->data[j];
        }
    }
}

void nnrt_batchnorm_2d(nnrt_Tensor *a, int num_features, float *gamma, float *shift, nnrt_Tensor *out) {
    size_t batch_size = a->shape[0], num_channels = a->shape[1], height = a->shape[2], width = a->shape[3];
    size_t spatial_dim = height * width;

    for (int feature = 0; feature < num_features; ++feature) {
        for (size_t batch = 0; batch < batch_size; ++batch) {
            float sum = 0.0f;
            float sq_sum = 0.0f;

            for (size_t i = 0; i < spatial_dim; ++i) {
                float val = a->data[batch * num_channels * spatial_dim + feature * spatial_dim + i];
                sum += val;
                sq_sum += val * val;
            }

            float mean = sum / spatial_dim;
            float variance = sq_sum / spatial_dim - mean * mean;
            float std_dev = sqrtf(variance + 1e-5f);  // Adding a small value for numerical stability

            for (size_t i = 0; i < spatial_dim; ++i) {
                out->data[batch * num_channels * spatial_dim + feature * spatial_dim + i] =
                    (a->data[batch * num_channels * spatial_dim + feature * spatial_dim + i] - mean) * gamma[feature] / std_dev + shift[feature];
            }
        }
    }
}

inline void nnrt_conv_2d(nnrt_Tensor *a, nnrt_Tensor *kernel, int stride, int pad, nnrt_Tensor *out) {
    // input shape: [batch_size, height, width, in_channels]
    // kernel shape: [k_height, k_width, in_channels, out_channels]
    // output shape: [batch_size, out_height, out_width, out_channels]
    // Padding and stride apply to height and width

    int batch_size = a->shape[0];
    int in_height = a->shape[1];
    int in_width = a->shape[2];
    int in_channels = a->shape[3];

    int k_height = kernel->shape[0];
    int k_width = kernel->shape[1];
    int out_channels = kernel->shape[3];

    int out_height = (in_height - k_height + 2 * pad) / stride + 1;
    int out_width = (in_width - k_width + 2 * pad) / stride + 1;

    NNRT_FLOAT *pad_input = (NNRT_FLOAT *)calloc(batch_size * (in_height + 2 * pad) * (in_width + 2 * pad) * in_channels, sizeof(NNRT_FLOAT));

    for (int b = 0; b < batch_size; ++b)
        for (int h = 0; h < in_height; ++h)
            for (int w = 0; w < in_width; ++w)
                for (int c = 0; c < in_channels; ++c)
                    pad_input[(b * (in_height + 2 * pad) * (in_width + 2 * pad) * in_channels) + ((h + pad) * (in_width + 2 * pad) * in_channels) + ((w + pad) * in_channels) + c] = a->data[(b * in_height * in_width * in_channels) + (h * in_width * in_channels) + (w * in_channels) + c];

    for (int b = 0; b < batch_size; ++b)
        for (int h = 0; h < out_height; ++h)
            for (int w = 0; w < out_width; ++w)
                for (int c_out = 0; c_out < out_channels; ++c_out) {
                    out->data[(b * out_height * out_width * out_channels) + (h * out_width * out_channels) + (w * out_channels) + c_out] = 0.0f;
                    for (int k_h = 0; k_h < k_height; ++k_h)
                        for (int k_w = 0; k_w < k_width; ++k_w)
                            for (int c_in = 0; c_in < in_channels; ++c_in)
                                out->data[(b * out_height * out_width * out_channels) + (h * out_width * out_channels) + (w * out_channels) + c_out] += pad_input[(b * (in_height + 2 * pad) * (in_width + 2 * pad) * in_channels) + ((h * stride + k_h) * (in_width + 2 * pad) * in_channels) + ((w * stride + k_w) * in_channels) + c_in] * kernel->data[(k_h * k_width * in_channels * out_channels) + (k_w * in_channels * out_channels) + (c_in * out_channels) + c_out];
                }
    free(pad_input);
}

inline void nnrt_matmul(nnrt_Tensor *a, nnrt_Tensor *b, nnrt_Tensor *out) {
    if (a->shape[1] != b->shape[0]) {
        printf("cannot multiply tensors with incompatible shape: [m, %d], [%d, n]",
               a->shape[1], b->shape[0]);
        exit(1);
    }
    for (int i = 0; i < a->shape[0]; ++i) {
        for (int j = 0; j < b->shape[1]; ++j) {
            out->data[i * a->shape[1] + j] = 0.0f;
            for (int k = 0; k < a->shape[1]; ++k) {
                out->data[i * b->shape[1] + j] += a->data[i * a->shape[1] + k] * b->data[k * b->shape[1] + j];
            }
        }
    }
}

void nnrt_maxpool_2d(nnrt_Tensor *a, nnrt_Tensor *kernel, int stride, int pad, nnrt_Tensor *out) {
    size_t batch_size = a->shape[0], num_channels = a->shape[1], height = a->shape[2], width = a->shape[3];
    size_t kH = kernel->shape[0], kW = kernel->shape[1];
    size_t outH = (height - kH + 2 * pad) / stride + 1;
    size_t outW = (width - kW + 2 * pad) / stride + 1;

    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t channel = 0; channel < num_channels; ++channel) {
            for (size_t i = 0; i < outH; ++i) {
                for (size_t j = 0; j < outW; ++j) {
                    NNRT_FLOAT max_val = -MAXFLOAT;
                    for (size_t ki = 0; ki < kH; ++ki) {
                        for (size_t kj = 0; kj < kW; ++kj) {
                            size_t in_i = stride * i + ki - pad;
                            size_t in_j = stride * j + kj - pad;
                            if (in_i < height && in_j < width) {
                                float val = a->data[batch * num_channels * height * width + channel * height * width + in_i * width + in_j];
                                max_val = fmaxf(max_val, val);
                            }
                        }
                    }
                    out->data[batch * num_channels * outH * outW + channel * outH * outW + i * outW + j] = max_val;
                }
            }
        }
    }
}

inline void nnrt_relu(nnrt_Tensor *a, nnrt_Tensor *out) {
    size_t total_size = nnrt_tensor_size(a);
    for (size_t i = 0; i < total_size; ++i) {
        out->data[i] = a->data[i] < 0.0f ? 0.0f : a->data[i];
    }
}

inline void nnrt_sigmoid(nnrt_Tensor *a, nnrt_Tensor *out) {
    size_t total_size = nnrt_tensor_size(a);
    for (size_t i = 0; i < total_size; ++i) {
        out->data[i] = 1.0f / (1.0f + expf(-a->data[i]));
    }
}

inline void nnrt_argmax(nnrt_Tensor *a, int axis, nnrt_Tensor *out) {
    size_t m = a->shape[0], n = a->shape[1];
    if (axis == 0) {  // column-wise argmax
        for (size_t j = 0; j < n; ++j) {
            NNRT_FLOAT max_val = a->data[j];
            size_t max_idx = 0;
            for (size_t i = 1; i < m; ++i) {
                if (a->data[i * n + j] > max_val) {
                    max_val = a->data[i * n + j];
                    max_idx = i;
                }
            }
            out->data[j] = max_idx;
        }
    } else if (axis == 1) {  // row-wise argmax
        for (size_t i = 0; i < m; ++i) {
            NNRT_FLOAT max_val = a->data[i * n];
            size_t max_idx = 0;
            for (size_t j = 1; j < n; ++j) {
                if (a->data[i * n + j] > max_val) {
                    max_val = a->data[i * n + j];
                    max_idx = j;
                }
            }
            out->data[i] = max_idx;
        }
    }
}

inline void nnrt_softmax(nnrt_Tensor *a, int axis, nnrt_Tensor *out) {
    if (a->ndim != 2) {
        printf("Softmax is only defined on rank-2 tensors");
        exit(1);
    }
    size_t m = a->shape[0], n = a->shape[1];
    if (axis == 0) {  // column-wise softmax
        for (size_t j = 0; j < n; ++j) {
            NNRT_FLOAT max_val = a->data[j];
            NNRT_FLOAT sum = 0.0f;
            for (size_t i = 0; i < m; ++i) {
                max_val = fmaxf(max_val, a->data[i * n + j]);
            }
            for (size_t i = 0; i < m; ++i) {
                out->data[i * n + j] = expf(a->data[i * n + j] - max_val);
                sum += out->data[i * n + j];
            }
            for (size_t i = 0; i < m; ++i) {
                out->data[i * n + j] /= sum;
            }
        }
    } else if (axis == 1) {  // row-wise softmax
        for (size_t i = 0; i < m; ++i) {
            NNRT_FLOAT max_val = a->data[i * n];
            NNRT_FLOAT sum = 0.0f;
            for (size_t j = 0; j < n; ++j) {
                max_val = fmaxf(max_val, a->data[i * n + j]);
            }
            for (size_t j = 0; j < n; ++j) {
                out->data[i * n + j] = expf(a->data[i * n + j] - max_val);
                sum += out->data[i * n + j];
            }
            for (size_t j = 0; j < n; ++j) {
                out->data[i * n + j] /= sum;
            }
        }
    }
}

inline void nnrt_image_hwc_to_chw(nnrt_Tensor *a, nnrt_Tensor *out) {
    size_t batch_size = a->shape[0];
    size_t height = a->shape[1];
    size_t width = a->shape[2];
    size_t channels = a->shape[3];

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t i = 0; i < height; ++i) {
                for (size_t j = 0; j < width; ++j) {
                    size_t in_idx = (b * height * width * channels) + (i * width * channels) + (j * channels) + c;
                    size_t out_idx = (b * channels * height * width) + (c * height * width) + (i * width) + j;
                    out->data[out_idx] = a->data[in_idx];
                }
            }
        }
    }
}

inline void nnrt_image_chw_to_hwc(nnrt_Tensor *a, nnrt_Tensor *out) {
    size_t batch_size = a->shape[0];
    size_t channels = a->shape[1];
    size_t height = a->shape[2];
    size_t width = a->shape[3];

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < height; ++i) {
            for (size_t j = 0; j < width; ++j) {
                for (size_t c = 0; c < channels; ++c) {
                    size_t in_idx = (b * channels * height * width) + (c * height * width) + (i * width) + j;
                    size_t out_idx = (b * height * width * channels) + (i * width * channels) + (j * channels) + c;
                    out->data[out_idx] = a->data[in_idx];
                }
            }
        }
    }
}

inline void nnrt_image_standardize(nnrt_Tensor *a, float *mean, float *stddev, nnrt_Tensor *out) {
    size_t batch_size = a->shape[0], num_channels = a->shape[1], height = a->shape[2], width = a->shape[3];
    size_t spatial_dim = height * width;

    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t channel = 0; channel < num_channels; ++channel) {
            for (size_t i = 0; i < spatial_dim; ++i) {
                out->data[batch * num_channels * spatial_dim + channel * spatial_dim + i] =
                    (a->data[batch * num_channels * spatial_dim + channel * spatial_dim + i] - mean[channel]) / stddev[channel];
            }
        }
    }
}

inline void nnrt_image_to_gray(nnrt_Tensor *a, nnrt_Tensor *out) {
    size_t batch_size = a->shape[0];
    size_t height = a->shape[1];
    size_t width = a->shape[2];
    size_t channels = a->shape[3];

    // Check if the input image is a 3-channel RGB image
    if (channels != 3) {
        printf("Error: Expected a 3-channel image but got a %zu-channel image\n", channels);
        return;
    }

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < height; ++i) {
            for (size_t j = 0; j < width; ++j) {
                size_t idx = (b * height * width + i * width + j) * channels;
                NNRT_FLOAT gray = (a->data[idx] + a->data[idx + 1] + a->data[idx + 2]) / 3.0f;
                out->data[b * height * width + i * width + j] = gray;
            }
        }
    }
}

nnrt_Tensor *nnrt_tensor_alloc(int ndim, int *shape) {
    nnrt_Tensor *tensor = (nnrt_Tensor *)malloc(sizeof(nnrt_Tensor));
    tensor->ndim = ndim;
    tensor->shape = (int *)malloc(tensor->ndim * sizeof(int));
    memcpy(tensor->shape, shape, ndim * sizeof(int));
    tensor->data = (float *)malloc(nnrt_tensor_size(tensor) * sizeof(NNRT_FLOAT));

    return tensor;
}

nnrt_Tensor *nnrt_tensor_fread(FILE *fp) {
    nnrt_Tensor *tensor = (nnrt_Tensor *)malloc(sizeof(nnrt_Tensor));

    fread(&tensor->ndim, sizeof(int), 1, fp);

    tensor->shape = (int *)malloc(tensor->ndim * sizeof(int));
    fread(tensor->shape, sizeof(int), tensor->ndim, fp);

    size_t sz = nnrt_tensor_size(tensor);

    tensor->data = (float *)malloc(sz * sizeof(NNRT_FLOAT));
    fread(tensor->data, sizeof(float), sz, fp);

    return tensor;
}

size_t nnrt_tensor_size(nnrt_Tensor *t) {
    size_t len = 1;
    for (size_t i = 0; i < t->ndim; i++)
        len *= t->shape[i];
    return len;
}

void nnrt_tensor_free(nnrt_Tensor *t) {
    free(t->shape);
    free(t->data);
    free(t);
}
