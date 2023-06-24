#include <stdio.h>
#include <stdlib.h>

#include "../nnrt.h"

int main(void) {
    char *weight_filename = "cnn.dat";
    FILE *fp = fopen(weight_filename, "rb");
    if (fp == NULL) {
        printf("cannot load model weight\n");
        exit(1);
    }
    // ---
    nnrt_Conv2DLayer *c1 = nnrt_conv_2d_layer_fread(fp);
    nnrt_Conv2DLayer *c2 = nnrt_conv_2d_layer_fread(fp);
    nnrt_LinearLayer *li = nnrt_linear_layer_fread(fp);
    // ---
    fclose(fp);

    char *x_test_filename = "x_test.dat";
    char *y_test_filename = "y_test.dat";

    fp = fopen(x_test_filename, "rb");
    nnrt_Tensor *x_test = nnrt_tensor_fread(fp);

    printf("Test data info:\n");
    printf("ndim  : %d\n", x_test->ndim);
    printf("shape : (%d, %d, %d, %d)\n",
           x_test->shape[0], x_test->shape[1], x_test->shape[2], x_test->shape[3]);
    fclose(fp);

    fp = fopen(y_test_filename, "rb");
    nnrt_Tensor *y_test = nnrt_tensor_fread(fp);
    fclose(fp);

    int n_samples = x_test->shape[0];

    // Hidden layer 1
    nnrt_Tensor *h1 = nnrt_conv_2d_layer_forward(x_test, c1);
    nnrt_relu(h1, h1);

    // Hidden layer 2
    nnrt_Tensor *h2 = nnrt_conv_2d_layer_forward(h1, c2);
    nnrt_relu(h2, h2);

    // Flatten
    nnrt_reshape_inplace(h2, (int[]){n_samples, 128}, 2);

    // Output layer
    nnrt_Tensor *out = nnrt_linear_layer_forward(h2, li);

    // Labels
    nnrt_Tensor *y_hat = nnrt_tensor_alloc(1, (int[]){n_samples});
    nnrt_argmax(out, 1, y_hat);

    printf("\n\n");
    printf("Predicted (first 50):\n");
    for (size_t i = 0; i < 50; i++) {
        printf("%d, ", (int)y_hat->data[i]);
    }
    printf("\n\n");
    printf("Actual (first 50):\n");
    float n_match = 0;
    for (size_t i = 0; i < 50; i++) {
        printf("%d, ", (int)y_test->data[i]);
        n_match += (y_test->data[i] == y_hat->data[i]);
    }
    printf("\n\n");
    printf("Accuracy: %f\n", n_match / 50);

    nnrt_conv_2d_layer_free(c1);
    nnrt_conv_2d_layer_free(c2);
    nnrt_linear_layer_free(li);
    nnrt_tensor_free(x_test);
    nnrt_tensor_free(y_test);
    nnrt_tensor_free(h1);
    nnrt_tensor_free(h2);
    nnrt_tensor_free(out);
    nnrt_tensor_free(y_hat);
    return 0;
}
