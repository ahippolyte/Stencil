CC      = mpicc
CFLAGS += -Wall -g -O3 -DSTENCIL_SIZE=$(STENCIL_SIZE)
LDLIBS += -lm #-lrt

STENCIL_SIZE ?= 100

all: stencil_seq stencil_mpi
	@echo "=-----------------------= STENCIL_SIZE: $(STENCIL_SIZE) =-----------------------="

clean:
	-rm stencil_seq

mrproper: clean
	-rm *~

archive: stencil.c Makefile
	( cd .. ; tar czf stencil.tar.gz stencil_seq/ )
