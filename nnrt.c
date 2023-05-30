#include "nnrt.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

inline void nnrt_affine(nnrt_Tensor *x, nnrt_Tensor *w, nnrt_Tensor *b, nnrt_Tensor *out) {
    // x * w + c
    // x: [m, p]
    // w: [p, n]
    // b: [1, n]
    nnrt_matmul(x, w, out);

    if ((b->ndim > 2) || b->ndim < 1) {
        printf("c must be of dimension 1 or 2\n");
        exit(1);
    }

    if (b->ndim == 2 && b->shape[0] != 1) {
        printf("leading dimension of c must be 1, got %d\n", b->shape[0]);
        exit(1);
    }

    if (b->ndim == 2 && w->shape[1] != b->shape[1]) {
        printf("cannot add intecept on tensor with incompatible shape: [m, %d], [1, %d]\n",
               w->shape[1], b->shape[1]);
        exit(1);
    }

    for (size_t i = 0; i < x->shape[0]; ++i) {
        for (size_t j = 0; j < w->shape[1]; ++j) {
            out->data[i * w->shape[1] + j] += b->data[j];
        }
    }
}

void nnrt_batchnorm_2d(nnrt_Tensor *a, int num_features, float *gamma, float *shift, nnrt_Tensor *out) {
    size_t batch_size = a->shape[0];
    size_t num_channels = a->shape[1];
    size_t height = a->shape[2], width = a->shape[3];

    size_t spatial_dim = height * width;

    for (int feature = 0; feature < num_features; ++feature) {
        for (size_t batch = 0; batch < batch_size; ++batch) {
            float sum = 0.0f;
            float sq_sum = 0.0f;

            for (size_t i = 0; i < spatial_dim; ++i) {
                float val = a->data[batch * num_channels * spatial_dim +
                                    feature * spatial_dim + i];
                sum += val;
                sq_sum += val * val;
            }

            float mean = sum / spatial_dim;
            float variance = sq_sum / spatial_dim - mean * mean;
            float std_dev = sqrtf(variance + 1e-5f);  // Adding a small value for numerical stability

            for (size_t i = 0; i < spatial_dim; ++i) {
                int oidx = batch * num_channels * spatial_dim + feature * spatial_dim + i;
                out->data[oidx] = (a->data[oidx] - mean) * gamma[feature] / std_dev + shift[feature];
            }
        }
    }
}

inline void nnrt_conv_2d(nnrt_Tensor *a, nnrt_Tensor *kernel, nnrt_Tensor *bias,
                         int stride, int pad, nnrt_Tensor *out) {
    int N = a->shape[0];
    int C = a->shape[1];
    int H = a->shape[2];
    int W = a->shape[3];

    int KH = kernel->shape[2];
    int KW = kernel->shape[3];

    // extend H and W with padding
    int HP = H + 2 * pad;
    int WP = W + 2 * pad;

    // create padded input tensor
    NNRT_FLOAT *a_pad = (NNRT_FLOAT *)calloc(N * C * HP * WP, sizeof(NNRT_FLOAT));
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    a_pad[n * (C * HP * WP) + c * (HP * WP) + (h + pad) * WP + (w + pad)] =
                        a->data[n * (C * H * W) + c * (H * W) + h * W + w];
                }
            }
        }
    }

    for (int n = 0; n < N; n++) {
        for (int m = 0; m < out->shape[1]; m++) {
            for (int c = 0; c < C; c++) {
                for (int h = 0; h <= (HP - KH); h += stride) {
                    for (int w = 0; w <= (WP - KW); w += stride) {
                        NNRT_FLOAT sum = 0.0;
                        for (int i = 0; i < KH; i++) {
                            for (int j = 0; j < KW; j++) {
                                sum += a_pad[n * (C * HP * WP) + c * (HP * WP) + (h + i) * WP + (w + j)] *
                                       kernel->data[m * (C * KH * KW) + c * (KH * KW) + i * KW + j];
                            }
                        }
                        size_t oidx = n * (out->shape[1] * out->shape[2] * out->shape[3]) +
                                      m * (out->shape[2] * out->shape[3]) +
                                      (h / stride) * out->shape[3] +
                                      (w / stride);
                        out->data[oidx] += sum + bias->data[m];
                    }
                }
            }
        }
    }

    // free memory of padded input tensor
    free(a_pad);
}

inline void nnrt_conv_2d_calc_out_size(nnrt_Tensor *a, nnrt_Tensor *kernel,
                                       int stride, int pad, int *out_h, int *out_w, int *out_c) {
    // input shape: [batch_size, in_channels, height, width]
    // kernel shape: [out_channels, in_channels, k_height, k_width]

    int in_height = a->shape[2];
    int in_width = a->shape[3];

    int out_channels = kernel->shape[0];
    int k_height = kernel->shape[2];
    int k_width = kernel->shape[3];

    int out_height = (in_height - k_height + 2 * pad) / stride + 1;
    int out_width = (in_width - k_width + 2 * pad) / stride + 1;

    *out_h = out_height;
    *out_w = out_width;
    *out_c = out_channels;
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
                out->data[i * b->shape[1] + j] +=
                    a->data[i * a->shape[1] + k] * b->data[k * b->shape[1] + j];
            }
        }
    }
}

inline void nnrt_maxpool_2d(nnrt_Tensor *a, nnrt_Tensor *kernel, int stride, int pad, nnrt_Tensor *out) {
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

inline void nnrt_reshape_inplace(nnrt_Tensor *a, int *new_shape, int new_ndim) {
    // Calculate the total size of the original tensor
    size_t original_size = nnrt_tensor_size(a);

    // Calculate the total size of the new shape
    size_t new_size = 1;
    for (int i = 0; i < new_ndim; ++i) {
        new_size *= new_shape[i];
    }

    // Ensure the total size remains the same after reshaping
    if (original_size != new_size) {
        printf("Error: reshape would change the total size of the tensor (%ld vs %ld)\n",
               original_size, new_size);
        exit(1);
    }

    // Free the original shape array
    free(a->shape);

    // Set the new shape
    a->shape = (int *)malloc(new_ndim * sizeof(int));
    memcpy(a->shape, new_shape, new_ndim * sizeof(int));
    a->ndim = new_ndim;
}

inline void nnrt_transpose_inplace(nnrt_Tensor *a) {
    // Ensure the input tensor is 2D
    if (a->shape[1] == 0) {
        printf("Error: input tensor to nnrt_transpose_inplace is not 2D\n");
        exit(1);
    }

    int rows = a->shape[0];
    int cols = a->shape[1];

    // Allocate temporary storage for the transpose
    NNRT_FLOAT *temp = (NNRT_FLOAT *)calloc(rows * cols, sizeof(NNRT_FLOAT));

    // Compute the transpose
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            temp[j * rows + i] = a->data[i * cols + j];
        }
    }

    // Swap the data pointers and shapes
    free(a->data);
    a->data = temp;
    a->shape[0] = cols;
    a->shape[1] = rows;
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
    tensor->data = (NNRT_FLOAT *)calloc(nnrt_tensor_size(tensor), sizeof(NNRT_FLOAT));

    return tensor;
}

nnrt_Tensor *nnrt_tensor_fread(FILE *fp) {
    nnrt_Tensor *tensor = (nnrt_Tensor *)malloc(sizeof(nnrt_Tensor));

    fread(&tensor->ndim, sizeof(int), 1, fp);

    tensor->shape = (int *)malloc(tensor->ndim * sizeof(int));
    fread(tensor->shape, sizeof(int), tensor->ndim, fp);

    size_t sz = nnrt_tensor_size(tensor);

    tensor->data = (NNRT_FLOAT *)malloc(sz * sizeof(NNRT_FLOAT));
    fread(tensor->data, sizeof(NNRT_FLOAT), sz, fp);

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
