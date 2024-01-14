.PHONY: all
all: clean conv

conv:
	mpicxx conv.cpp -O3 -o conv -fopenmp -lm -lmpi

.PHONY: run
run:
	mpiexec -n 16 --oversubscribe ./conv 10.

.PHONY: clean
clean:
	rm -fv conv
