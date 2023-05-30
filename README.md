# Neural network runtime library

Load neural net weight trained in python (as NumPy array) in C, use it for inference.
The runtime library is written in pure C.

## Running examples

The python examples use PyTorch, so you need to install it (and NumPy).

- Iris:

        make iris && ./iris.out

- CNN:

        make cnn && ./cnn.out