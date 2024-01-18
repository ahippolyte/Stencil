CC      = mpicc
CFLAGS += -Wall -g -O3 -DSTENCIL_SIZE=$(STENCIL_SIZE) -fopenmp
LDLIBS += -lm #-lrt

STENCIL_SIZE ?= 100

all: stencil_seq stencil_mpi stencil_openmp
	@echo "=-----------------------= STENCIL_SIZE: $(STENCIL_SIZE) =-----------------------="

clean:
	-rm stencil_seq stencil_mpi stencil_openmp

mrproper: clean
	-rm *~

archive: stencil.c Makefile
	( cd .. ; tar czf stencil.tar.gz stencil_seq/ stencil_mpi/ stencil_openmp/ )
