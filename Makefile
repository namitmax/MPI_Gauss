WARNINGS= #-mfpmath=sse -fstack-protector-all -Wall -g -Wunused -Wcast-align -Werror -pedantic -pedantic-errors -Wfloat-equal -Wpointer-arith -Wformat-security -Wmissing-format-attribute -Wformat=1 -Wwrite-strings -Wcast-align -Wno-long-long -Woverloaded-virtual -Wnon-virtual-dtor -Wcast-qual -Wno-suggest-attribute=format
FLAGS= -O3 -ffast-math

all: a.out
a.out: func_mpi.o main.o
	mpicxx  func_mpi.o main.o
func_mpi.o: func_mpi.cpp head_MPI.h
	mpicxx $(WARNINGS) $(FLAGS) -c func_mpi.cpp
main.o: main.cpp head_MPI.h
	mpicxx $(WARNINGS) $(FLAGS) -c main.cpp
clear:
	rm *.o *.out
