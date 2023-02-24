CC=gcc

CFLAGS=-Wall -g

GCC_OFLAGS=-march=native -mavx2 -Ofast -finline-functions -funroll-loops -ftree-vectorize -ftree-loop-vectorize -fopenmp -fopt-info-all=dist.gcc.optrpt 
ICX_OFLAGS=-march=native -mavx2 -Ofast -finline-functions -funroll-loops -fopenmp

all: genseq mask.g

mask.g: mask.c
	gcc $(CFLAGS) $(GCC_OFLAGS) $< -o $@
mask.i: mask.c
	icx $(CFLAGS) $(ICX_OFLAGS) $< -o $@

genseq: genseq.c
	$(CC) -march=native $(CFLAGS) -Ofast $< -o $@

clean:
	rm -Rf *~ genseq mask *.optrpt
