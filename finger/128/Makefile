CXX=g++
CXXFLAGS=-fopenmp -O3 -std=c++14 -fPIC -DNDEBUG -Wall -g -lopenblas 
EXTRA_INCLUDE_FLAGS=-I.Eigen
ARCHFLAG=-march=native


all: go

go: example.cpp
	${CXX} -o go ${CXXFLAGS} example.cpp -I. ${EXTRA_INCLUDE_FLAGS} ${ARCHFLAG}
clean:
	rm -rf *.so *.o go
