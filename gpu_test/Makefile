CUDA_PATH ?= /usr/local/cuda
NVCC      ?= $(CUDA_PATH)/bin/nvcc
CXXFLAGS  := -O3 -std=c++11 -I../src -I../src/ribbon -I../src/xorfilter
TARGET    := gpu_test_ribbonfilter
SRC       := gpu_test_ribbonfilter.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)