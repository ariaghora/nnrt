PY=python3
CC=gcc
CFLAGS=-Wall -Werror -pedantic -O3 -lm -ftree-vectorize -march=armv8.4-a

# Building examples
# -----------------

iris:
	@${CC} ${CFLAGS} -o examples/iris.out examples/iris.c nnrt.c

cnn:
	@${CC} ${CFLAGS} -o examples/cnn.out examples/cnn.c nnrt.c nnrt_layers.c

# Building neural nets
# --------------------

alexnet:
	@${PY} -m nets.alexnet.utils
	@${CC} ${CFLAGS} -o nets/alexnet/alexnet.out nets/alexnet/alexnet.c nnrt.c nnrt_layers.c


# ----
# etc.
clean:
	@rm -f examples/*.dat
	@rm -f examples/*.out