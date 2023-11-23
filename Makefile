CC      = mpicc
CFLAGS += -Wall -g -O4
LDLIBS += -lm -lrt

all: stencil_seq stencil_mpi

clean:
	-rm stencil_seq

mrproper: clean
	-rm *~

archive: stencil.c Makefile
	( cd .. ; tar czf stencil.tar.gz stencil_seq/ )
