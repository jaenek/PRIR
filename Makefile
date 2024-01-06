.PHONY: all
all: clean conv

conv:
	gcc conv.c -O3 -o conv -fopenmp -lm

.PHONY: clean
clean:
	rm -fv conv
