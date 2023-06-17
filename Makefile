PY=python3
CC=gcc
CFLAGS=-Wall -Werror -pedantic -O3 -lm -ftree-vectorize -fno-trapping-math -mcpu=apple-m1 

libnnrt.dylib:
	@${CC} -dynamiclib -o libnnrt.dylib nnrt.c nnrt_layers.c ${CFLAGS}

install:
	@cp *.h /usr/local/include/
	@cp libnnrt.dylib /usr/local/lib/


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
	@rm -f *.dylib
