PATH+=:/usr/local/cuda/bin
CC=nvcc
LDLIBS=-lm
TARGETS=power power_gpu
all: ${TARGETS}
power: power.c
	cc -std=c99 -O3 -o power power.c -lm
power_gpu: power_gpu.cu
	nvcc -o power_gpu power_gpu.cu -lm
clean:
	-rm -f ${TARGETS}
