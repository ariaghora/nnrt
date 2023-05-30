iris:
	@gcc -Wall -Werror -pedantic -O3 -lm -o iris.out iris.c nnrt.c

cnn:
	@gcc -Wall -Werror -pedantic -g  -lm -o cnn.out cnn.c nnrt.c nnrt_layers.c