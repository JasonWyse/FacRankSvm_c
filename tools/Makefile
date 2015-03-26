CXX ?= g++
CFLAGS ?= -Wall -Wconversion -O3 -fPIC

all: svm-scale eval closest_pairs

eval: eval.cpp selectiontree.cpp selectiontree.h
	$(CXX) $(CFLAGS) -o eval eval.cpp selectiontree.cpp selectiontree.h

svm-scale: svm-scale.c
	$(CXX) $(CFLAGS) -o svm-scale svm-scale.c

closest_pairs: closest_pairs.c
	$(CXX) $(CFLAGS) -o closest_pairs closest_pairs.c

clean:
	rm -f eval svm-scale closest_pairs *~
