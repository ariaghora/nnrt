#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../../nnrt.h"
#include "../../nnrt_layers.h"

#define DEBUG(label, t) (printf("%s: [%d, %d, %d, %d]\n", \
                                label,                    \
                                t->shape[0],              \
                                t->shape[1],              \
                                t->shape[2],              \
                                t->shape[3]))

#define IMG_SIZE 256
#define N_CHANNELS 3

nnrt_Tensor *get_feature(nnrt_Tensor *image_batch, nnrt_Conv2DLayer **conv_layers, int n_conv_layers) {
    /// Part 1
    nnrt_Tensor *h1 =nnrt_conv_2d_layer_forward(conv_layers[0], image_batch);
    nnrt_relu(h1, h1);
    nnrt_Tensor *h1_pool = nnrt_maxpool_2d(h1, 3, 2, 0);
    nnrt_tensor_free(h1);

    /// Part 2
    nnrt_Tensor *h2 = nnrt_conv_2d_layer_forward(conv_layers[1], h1_pool);
    nnrt_tensor_free(h1_pool);
    nnrt_relu(h2, h2);
    nnrt_Tensor *h2_pool = nnrt_maxpool_2d(h2, 3, 2, 0);
    nnrt_tensor_free(h2);

    /// Part 3
    nnrt_Tensor *h3 = nnrt_conv_2d_layer_forward(conv_layers[2], h2_pool);
    nnrt_tensor_free(h2_pool);
    nnrt_relu(h3, h3);

    /// Part 4
    nnrt_Tensor *h4 = nnrt_conv_2d_layer_forward(conv_layers[3], h3);
    nnrt_tensor_free(h3);
    nnrt_relu(h4, h4);

    /// Part 5
    nnrt_Tensor *h5 = nnrt_conv_2d_layer_forward(conv_layers[4], h4);
    nnrt_tensor_free(h4);
    nnrt_relu(h5, h5);
    nnrt_Tensor *h5_pool = nnrt_maxpool_2d(h5, 3, 2, 0);
    nnrt_tensor_free(h5);

    nnrt_Tensor *h5_pool_adaptive = nnrt_adaptive_avg_pool2d(h5_pool, 6, 6);
    nnrt_tensor_free(h5_pool);

    // Flatten
    nnrt_reshape_inplace(h5_pool_adaptive,
                         (int[]){h5_pool_adaptive->shape[0], h5_pool_adaptive->shape[1] * 6 * 6}, 2);

    return h5_pool_adaptive;
}

int get_prediction(nnrt_Tensor *feature, nnrt_LinearLayer **linear_layers, int n_linear_layers) {
    nnrt_Tensor *fc1 = nnrt_linear_layer_forward(linear_layers[0], feature);
    nnrt_relu(fc1, fc1);

    nnrt_Tensor *fc2 = nnrt_linear_layer_forward(linear_layers[1], fc1);
    nnrt_relu(fc2, fc2);

    nnrt_Tensor *fc3 = nnrt_linear_layer_forward(linear_layers[2], fc2);

    size_t n = feature->shape[0];
    nnrt_Tensor *out = nnrt_tensor_alloc(2, (int[]){n, 1});
    nnrt_argmax(fc3, 1, out);

    int label_idx = out->data[0];

    nnrt_tensor_free(fc1);
    nnrt_tensor_free(fc2);
    nnrt_tensor_free(fc3);
    nnrt_tensor_free(out);
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
    // ======
    // nnrt_Tensor *img = load_image_alexnet(image_path);
    nnrt_Tensor *img = nnrt_image_load(image_path);
    nnrt_Tensor *res = nnrt_image_resize(img, IMG_SIZE, IMG_SIZE);
    nnrt_image_standardize(res,
                           (float[]){0.485, 0.456, 0.406},
                           (float[]){0.229, 0.224, 0.225},
                           res);
    nnrt_Tensor *feature = get_feature(res, conv_layers, 5);
    int label_index = get_prediction(feature, linear_layers, 3);
    // ======
    printf("Label index : %d\n", label_index);
    printf("Class name  : %s\n", labels[label_index]);

    // free layers
    for (size_t i = 0; i < 5; i++)
        nnrt_conv_2d_layer_free(conv_layers[i]);
    for (size_t i = 0; i < 3; i++)
        nnrt_linear_layer_free(linear_layers[i]);

    nnrt_tensor_free(img);
    nnrt_tensor_free(res);
    nnrt_tensor_free(feature);

    return 0;
}
