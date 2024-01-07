.PHONY: all
all: clean conv

conv:
	g++ conv.cpp -O3 -o conv -fopenmp -lm

.PHONY: clean
clean:
	rm -fv conv
