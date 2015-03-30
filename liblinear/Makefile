CXX ?= g++
CC ?= gcc
CFLAGS = -Wall -Wconversion -O3 -fPIC 
# -Wall: Generally we use this option，it allows gcc to output all useful warnings。use -W{warning} to mark the labeled warning
#-Wconversion: Its original purpose was helping to translate old C code to modern C standards by identifying the places where adding function prototypes may result in different behaviour. The new Wconversion option warns for any implicit conversion that may change a value
# -03:compiler's 4 level optimization choices，-O0:no optimization,-O1:default value，-O3:highest value
#-fPIC: 编译器就输出位置无关目标码.适用于动态连接(dynamic linking),即使分支需要大范围转移
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
