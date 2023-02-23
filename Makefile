CC=gcc

CFLAGS=-Wall -g

OFLAGS=-march=native -Ofast -finline-functions -fopenmp -fopt-info-all=dist.gcc.optrpt 

all: genseq mask

mask: mask.c
	$(CC) $(CFLAGS) $(OFLAGS) $< -o $@

genseq: genseq.c
	$(CC) -march=native $(CFLAGS) -Ofast $< -o $@

clean:
	rm -Rf *~ genseq mask *.optrpt
