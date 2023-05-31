#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../nnrt.h"

int main(void) {
    // Sample of min-max scaled of iris dataset
    float x_data[] = {
        0.22222222, 0.62500000, 0.06779661, 0.04166667,  // -> 0
        0.16666667, 0.41666667, 0.06779661, 0.04166667,  // -> 0
        0.61111111, 0.41666667, 0.71186441, 0.79166667,  // -> 2
        0.52777778, 0.58333333, 0.74576271, 0.91666667   // -> 2
    };
    nnrt_Tensor *x = nnrt_tensor_alloc(2, (int[]){4, 4});
    memcpy(x->data, x_data, 16 * sizeof(float));

    // Load trained parameters
    char *filename = "mlp.dat";
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Cannot load weight\n");
        exit(1);
    }
    nnrt_Tensor *w1 = nnrt_tensor_fread(fp);
    nnrt_Tensor *w2 = nnrt_tensor_fread(fp);
    nnrt_Tensor *w3 = nnrt_tensor_fread(fp);
    nnrt_Tensor *b1 = nnrt_tensor_fread(fp);
    nnrt_Tensor *b2 = nnrt_tensor_fread(fp);
    nnrt_Tensor *b3 = nnrt_tensor_fread(fp);
    fclose(fp);

    // Hidden layer 1
    nnrt_Tensor *h1 = nnrt_tensor_alloc(2, (int[]){4, 100});
    nnrt_affine(x, w1, b1, h1);
    nnrt_relu(h1, h1);

    // Hidden layer 2
    nnrt_Tensor *h2 = nnrt_tensor_alloc(2, (int[]){4, 20});
    nnrt_affine(h1, w2, b2, h2);
    nnrt_relu(h2, h2);

    // Output layer
    nnrt_Tensor *out = nnrt_tensor_alloc(2, (int[]){4, 3});
    nnrt_affine(h2, w3, b3, out);
    nnrt_relu(out, out);

    //// optionally, calculate softmax
    // nnrt_softmax(out, 1, out);

    // Get labels
    nnrt_Tensor *lbl = nnrt_tensor_alloc(2, (int[]){4, 1});
    nnrt_argmax(out, 1, lbl);

    printf("Labels:\n");
    for (size_t i = 0; i < 4; i++)
        printf("%f\n", lbl->data[i]);

    // Free params
    nnrt_tensor_free(x);
    nnrt_tensor_free(w1);
    nnrt_tensor_free(w2);
    nnrt_tensor_free(w3);
    nnrt_tensor_free(b1);
    nnrt_tensor_free(b2);
    nnrt_tensor_free(b3);

    // Free intermediary variables
    nnrt_tensor_free(h1);
    nnrt_tensor_free(h2);
    nnrt_tensor_free(out);
    nnrt_tensor_free(lbl);

    return 0;
}
