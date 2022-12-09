CXX=g++
CXXFLAGS=-fopenmp -O3 -std=c++14 -fPIC -DNDEBUG -Wall -g -lopenblas
EXTRA_INCLUDE_FLAGS=-I../../pecos/core/ -I.Eigen
ARCHFLAG=-march=native

all: go

go: go.cpp
	        ${CXX} -o go ${CXXFLAGS} go.cpp -I. ${EXTRA_INCLUDE_FLAGS} ${ARCHFLAG}
clean:
	        rm -rf *.so *.o go
