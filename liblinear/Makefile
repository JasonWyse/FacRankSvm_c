CXX ?= g++
CC ?= gcc
CFLAGS = -Wall -Wconversion -O3 -fPIC
LIBS = blas/blas.a
#LIBS = -lblas

all: train predict train-fig56

train-fig56: tron-fig56.o linear-fig56.o binarytrees.o selectiontree.o train.c blas/blas.a
	$(CXX) $(CFLAGS) -D FIGURE56 -o train-fig56 train.c tron-fig56.o binarytrees.o selectiontree.o linear-fig56.o $(LIBS)

tron-fig56.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -D FIGURE56 -c -o tron-fig56.o tron.cpp

linear-fig56.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) -D FIGURE56 -c -o linear-fig56.o linear.cpp

train: tron.o binarytrees.o selectiontree.o linear.o train.c blas/blas.a
	$(CXX) $(CFLAGS) -o train train.c tron.o binarytrees.o selectiontree.o linear.o $(LIBS)

predict: tron.o binarytrees.o selectiontree.o linear.o predict.c blas/blas.a
	$(CXX) $(CFLAGS) -o predict predict.c tron.o binarytrees.o selectiontree.o linear.o $(LIBS)

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

selectiontree.o: selectiontree.cpp selectiontree.h
	$(CXX) $(CFLAGS) -c -o selectiontree.o selectiontree.cpp

binarytrees.o: binarytrees.cpp binarytrees.h
	$(CXX) $(CFLAGS) -c -o binarytrees.o binarytrees.cpp

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) -c -o linear.o linear.cpp

blas/blas.a: blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';

clean:
	make -C blas clean
	rm -f *~ selectiontree.o binarytrees.o tron*.o linear*.o train train-fig56 predict
