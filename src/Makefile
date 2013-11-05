### Compilation options.

# C++ compiler. Tested with g++ and Intel icpc.
CXX=g++
#CXX=icpc

# Compiler options. Note that -DEIGEN_NO_DEBUG is essential for good performance!
#CFLAGS=-g
CFLAGS=-O3 -DEIGEN_NO_DEBUG -DNDEBUG 

# Architecture. Set to x86_64 or i686 to override.
ARCH:=$(shell uname -m)
# Operating system. Set to override (the only option that makes any difference is Darwin).
OS:=$(shell uname -s)

# To build static binaries, uncomment the line below:
#STATIC=1

### Required libraries. You must install these prior to building.

# Set this to the root directory of Boost (should have a subdirectory named boost):
BOOST=/usr
#BOOST=/usr
#BOOST=/opt/local
# Where to find Boost header files
BOOST_INC=$(BOOST)/include

# Set this to the root directory of Eigen (should have a subdirectory named Eigen):
EIGEN=/usr/include/eigen3

### Optional libraries.

# To disable multithreading, comment out the line below:
OMP=1

# To use the MKL library, uncomment the line below and set it to the MKL root:
#MKL=/usr/usc/intel/12.1.1/mkl

# For Python bindings, set the following and run 'make python/nplm.so'.
PYTHON_VERSION=2.7
#PYTHON_ROOT=/opt/local/Library/Frameworks/Python.framework/Versions/$(PYTHON_VERSION)
PYTHON_ROOT=/home/nlg-01/chiangd/pkg64/python
CYTHON=$(PYTHON_ROOT)/bin/cython

##### End of configurable options #####

# used for profiling
#USE_CHRONO=1

TCLAP=../3rdparty/tclap/include

# Currently, this is needed only if USE_CHRONO is defined:
# Where to find Boost libraries
BOOST_LIB=$(BOOST)/lib
# On some systems, a suffix is appended for the multithreaded version.
BOOST_LIB_SUFFIX=
#BOOST_LIB_SUFFIX=-mt

BOOST_CFLAGS=-I$(BOOST_INC)
BOOST_LDFLAGS=
ifdef USE_CHRONO
  BOOST_CFLAGS+=-DUSE_CHRONO
  BOOST_LDLIBS+=-lboost_system$(BOOST_LIB_SUFFIX) -lboost_chrono$(BOOST_LIB_SUFFIX)
endif
ifdef BOOST_LDLIBS
  BOOST_LDFLAGS+=-L$(BOOST_LIB) -Wl,-rpath -Wl,$(BOOST_LIB)
endif

ifdef OMP
  ifneq (,$(findstring g++,$(CXX)))
    OMP_CFLAGS=-fopenmp
    OMP_LDFLAGS=-fopenmp
  endif
  ifneq (,$(findstring icpc,$(CXX)))
    OMP_CFLAGS=-openmp
    OMP_LDFLAGS=-openmp
  endif
endif

ifdef MKL
  MKL_CFLAGS=-I$(MKL)/include -DEIGEN_USE_MKL_ALL
  MKL_LDLIBS=-Wl,--start-group
  ifeq ($(ARCH),x86_64)
    MKL_LDFLAGS=-L$(MKL)/lib/intel64 -Wl,-rpath -Wl,$(MKL)/lib/intel64
    MKL_LDLIBS+=-lmkl_intel_lp64
  endif
  ifeq ($(ARCH),i686)
    MKL_LDFLAGS=-L$(MKL)/lib/ia32 -Wl,-rpath -Wl,$(MKL)/lib/ia32
    MKL_LDLIBS+=-lmkl_intel
  endif

  ifneq (,$(findstring g++,$(CXX)))
    MKL_LDLIBS+=-lmkl_gnu_thread
  endif
  ifneq (,$(findstring icpc,$(CXX)))
    MKL_LDLIBS+=-lmkl_intel_thread
  endif

  #MKL_LDLIBS=-lmkl_rt
  MKL_LDLIBS+=-lmkl_core -Wl,--end-group
endif

ifdef STATIC
  LDFLAGS+=-static
endif

ALL_CFLAGS=$(OMP_CFLAGS) $(MKL_CFLAGS) $(BOOST_CFLAGS) -I$(TCLAP) -I$(EIGEN) $(CFLAGS)
ALL_LDFLAGS=$(OMP_LDFLAGS) $(MKL_LDFLAGS) $(BOOST_LDFLAGS) $(LDFLAGS)
ALL_LDLIBS=$(MKL_LDLIBS) $(BOOST_LDLIBS)

PYTHON_CFLAGS+=-I$(PYTHON_ROOT)/include/python$(PYTHON_VERSION)
ifeq ($(OS),Darwin)
  # avoid having to link in libpython
  PYTHON_LDFLAGS+=-undefined dynamic_lookup
endif

# Some other programs

AR=ar
RANLIB=ranlib

# Rules

BINS=trainNeuralNetwork testNeuralNetwork prepareNeuralLM testNeuralLM prepareNeuralTM
LIBS=libneuralLM.a libneuralLM.so
OBJS=util.o model.o

all: $(BINS) $(LIBS)

clean:
	rm -f *.o shared/*.o python/*.o $(BINS) $(LIBS) python/nplm.{cpp,so}

install: all
	mkdir -p ../bin
	cp $(BINS) ../bin
	mkdir -p ../lib
	cp $(LIBS) ../lib

%.o: %.cpp
	$(CXX) -c $(ALL_CFLAGS) $< -o $@

shared/%.o: %.cpp
	$(CXX) -c -fPIC $(ALL_CFLAGS) $< -o $@

trainNeuralNetwork: trainNeuralNetwork.o $(OBJS)
	$(CXX) $(ALL_LDFLAGS) $^ $(ALL_LDLIBS) -o $@

testNeuralNetwork: testNeuralNetwork.o $(OBJS)
	$(CXX) $(ALL_LDFLAGS) $^ $(ALL_LDLIBS) -o $@

prepareNeuralLM: prepareNeuralLM.o $(OBJS)
	$(CXX) $(ALL_LDFLAGS) $^ $(ALL_LDLIBS) -o $@

testNeuralLM: testNeuralLM.o $(OBJS)
	$(CXX) $(ALL_LDFLAGS) $^ $(ALL_LDLIBS) -o $@

prepareNeuralTM: prepareNeuralTM.o $(OBJS)
	$(CXX) $(ALL_LDFLAGS) $^ $(ALL_LDLIBS) -o $@

libneuralLM.a: neuralLM.o $(OBJS)
	rm -f $@
	$(AR) rv $@ $^
	$(RANLIB) $@

libneuralLM.so: $(addprefix shared/,neuralLM.o $(OBJS))
	$(CXX) -shared $(ALL_LDFLAGS) $^ $(ALL_LDLIBS) -o $@

python/nplm.cpp: python/nplm.pyx
	$(CYTHON) --cplus $^

python/nplm.o: python/nplm.cpp
	$(CXX) -c -fPIC -I. $(ALL_CFLAGS) $(PYTHON_CFLAGS) $< -o $@

python/nplm.so: python/nplm.o $(addprefix shared/,neuralLM.o $(OBJS))
	$(CXX) -shared $(ALL_LDFLAGS) $(PYTHON_LDFLAGS) $^ $(ALL_LDLIBS) $(PYTHON_LDLIBS) -o $@ 
