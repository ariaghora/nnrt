#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../../nnrt.h"
#include "../../nnrt_layers.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../vendor/stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../../vendor/stb/stb_image_resize.h"

#define DEBUG(label, t) (printf("%s: [%d, %d, %d, %d]\n", \
                                label,                    \
                                t->shape[0],              \
                                t->shape[1],              \
                                t->shape[2],              \
                                t->shape[3]))

#define IMG_SIZE 256
#define N_CHANNELS 3

nnrt_Tensor *load_image_alexnet(char *filename) {
    int img_h, img_w, img_c;
    float *img = stbi_loadf(filename, &img_w, &img_h, &img_c, 3);
    if (!img) {
        printf("Error: cannot load image\n");
        exit(1);
    }
    float *resized = (float *)calloc(IMG_SIZE * IMG_SIZE * 3, sizeof(float));
    int status = stbir_resize_float(img, img_w, img_h, 0, resized, IMG_SIZE, IMG_SIZE, 0, N_CHANNELS);
    if (status == 0) {
        printf("Error: cannot resize image\n");
        exit(1);
    }

    nnrt_Tensor *img_tensor = nnrt_tensor_alloc(4, (int[]){1, IMG_SIZE, IMG_SIZE, N_CHANNELS});
    memcpy(img_tensor->data, resized, IMG_SIZE * IMG_SIZE * N_CHANNELS * sizeof(float));

    nnrt_Tensor *img_tensor_chw = nnrt_tensor_alloc(4, (int[]){1, N_CHANNELS, IMG_SIZE, IMG_SIZE});
    nnrt_image_hwc_to_chw(img_tensor, img_tensor_chw);
    nnrt_image_standardize(img_tensor_chw,
                           (float[]){0.485, 0.456, 0.406},
                           (float[]){0.229, 0.224, 0.225},
                           img_tensor_chw);

    stbi_image_free(resized);
    stbi_image_free(img);
    nnrt_tensor_free(img_tensor);
    return img_tensor_chw;
}

nnrt_Tensor *get_feature(nnrt_Tensor *image_batch, nnrt_Conv2DLayer **conv_layers, int n_conv_layers) {
    int n = image_batch->shape[0];
    int out_h, out_w, out_c;

    /// Part 1
    nnrt_conv_2d_calc_out_size(image_batch, conv_layers[0]->w, conv_layers[0]->stride, conv_layers[0]->pad,
                               &out_h, &out_w, &out_c);
    nnrt_Tensor *h1 = nnrt_tensor_alloc(4, (int[]){n, out_c, out_h, out_w});
    nnrt_conv_2d_layer_forward(image_batch, conv_layers[0], h1);
    nnrt_relu(h1, h1);
    nnrt_maxpool_2d_calc_out_size(h1, 3, 2, 0, &out_h, &out_w, &out_c);
    nnrt_Tensor *h1_pool = nnrt_tensor_alloc(4, (int[]){n, out_c, out_h, out_w});
    nnrt_maxpool_2d(h1, 3, 2, 0, h1_pool);

    /// Part 2
    nnrt_conv_2d_calc_out_size(h1_pool, conv_layers[1]->w, conv_layers[1]->stride, conv_layers[1]->pad,
                               &out_h, &out_w, &out_c);
    nnrt_Tensor *h2 = nnrt_tensor_alloc(4, (int[]){n, out_c, out_h, out_w});
    nnrt_conv_2d_layer_forward(h1_pool, conv_layers[1], h2);
    nnrt_relu(h2, h2);
    nnrt_maxpool_2d_calc_out_size(h2, 3, 2, 0, &out_h, &out_w, &out_c);
    nnrt_Tensor *h2_pool = nnrt_tensor_alloc(4, (int[]){n, out_c, out_h, out_w});
    nnrt_tensor_free(h1);
    nnrt_tensor_free(h1_pool);
    nnrt_maxpool_2d(h2, 3, 2, 0, h2_pool);

    /// Part 3
    nnrt_conv_2d_calc_out_size(h2_pool, conv_layers[2]->w, conv_layers[2]->stride, conv_layers[2]->pad,
                               &out_h, &out_w, &out_c);
    nnrt_Tensor *h3 = nnrt_tensor_alloc(4, (int[]){n, out_c, out_h, out_w});
    nnrt_conv_2d_layer_forward(h2_pool, conv_layers[2], h3);
    nnrt_tensor_free(h2);
    nnrt_tensor_free(h2_pool);
    nnrt_relu(h3, h3);

    /// Part 4
    nnrt_conv_2d_calc_out_size(h3, conv_layers[3]->w, conv_layers[3]->stride, conv_layers[3]->pad,
                               &out_h, &out_w, &out_c);
    nnrt_Tensor *h4 = nnrt_tensor_alloc(4, (int[]){n, out_c, out_h, out_w});
    nnrt_conv_2d_layer_forward(h3, conv_layers[3], h4);
    nnrt_tensor_free(h3);
    nnrt_relu(h4, h4);

    /// Part 5
    nnrt_conv_2d_calc_out_size(h4, conv_layers[4]->w, conv_layers[4]->stride, conv_layers[4]->pad,
                               &out_h, &out_w, &out_c);
    nnrt_Tensor *h5 = nnrt_tensor_alloc(4, (int[]){n, out_c, out_h, out_w});
    nnrt_conv_2d_layer_forward(h4, conv_layers[4], h5);
    nnrt_tensor_free(h4);
    nnrt_relu(h5, h5);
    nnrt_maxpool_2d_calc_out_size(h5, 3, 2, 0, &out_h, &out_w, &out_c);
    nnrt_Tensor *h5_pool = nnrt_tensor_alloc(4, (int[]){n, out_c, out_h, out_w});
    nnrt_maxpool_2d(h5, 3, 2, 0, h5_pool);

    nnrt_Tensor *h5_pool_adaptive = nnrt_tensor_alloc(4, (int[]){n, out_c, 6, 6});
    nnrt_adaptive_avg_pool2d(h5_pool, 6, 6, h5_pool_adaptive);

    nnrt_reshape_inplace(h5_pool_adaptive, (int[]){n, out_c * 6 * 6}, 2);

    nnrt_tensor_free(h5);
    nnrt_tensor_free(h5_pool);
    return h5_pool_adaptive;
}

int get_prediction(nnrt_Tensor *feature, nnrt_LinearLayer **linear_layers, int n_linear_layers) {
    size_t n = feature->shape[0];
    nnrt_Tensor *fc1 = nnrt_tensor_alloc(2, (int[]){n, linear_layers[0]->w->shape[1]});
    nnrt_linear_layer_forward(feature, linear_layers[0], fc1);
    nnrt_relu(fc1, fc1);

    nnrt_Tensor *fc2 = nnrt_tensor_alloc(2, (int[]){n, linear_layers[1]->w->shape[1]});
    nnrt_linear_layer_forward(fc1, linear_layers[1], fc2);
    nnrt_relu(fc2, fc2);

    nnrt_Tensor *fc3 = nnrt_tensor_alloc(2, (int[]){n, linear_layers[2]->w->shape[1]});
    nnrt_linear_layer_forward(fc2, linear_layers[2], fc3);

    nnrt_Tensor *out = nnrt_tensor_alloc(2, (int[]){n, 1});
    nnrt_argmax(fc3, 1, out);

    int label_idx = out->data[0];

    nnrt_tensor_free(fc1);
    nnrt_tensor_free(fc2);
    nnrt_tensor_free(fc3);
    return label_idx;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Error: please provide the alexnet weight\n");
        exit(1);
    }
    char *weight_path = argv[1];

    if (argc < 3) {
        printf("Error: please provide the image input\n");
        exit(1);
    }
    char *image_path = argv[2];

    nnrt_Conv2DLayer *conv_layers[5];
    nnrt_LinearLayer *linear_layers[3];

    FILE *fp = fopen(weight_path, "rb");
    if (!fp) {
        printf("Error: `alexnet.dat` not found.\n");
        exit(1);
    }

    for (size_t i = 0; i < 5; i++)
        conv_layers[i] = nnrt_conv_2d_layer_fread(fp);
    for (size_t i = 0; i < 3; i++)
        linear_layers[i] = nnrt_linear_layer_fread(fp);
    fclose(fp);

#include "labels.inc"
    clock_t start = clock();
    // ======
    nnrt_Tensor *img = load_image_alexnet(image_path);
    nnrt_Tensor *feature = get_feature(img, conv_layers, 5);
    int label_index = get_prediction(feature, linear_layers, 3);
    // ======
    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Inference time: %f seconds\n\n", cpu_time_used);
    printf("Label index : %d\n", label_index);
    printf("Class name  : %s\n", labels[label_index]);

    // free layers
    for (size_t i = 0; i < 5; i++)
        nnrt_conv_2d_layer_free(conv_layers[i]);
    for (size_t i = 0; i < 3; i++)
        nnrt_linear_layer_free(linear_layers[i]);

    nnrt_tensor_free(img);
    nnrt_tensor_free(feature);

    return 0;
}
