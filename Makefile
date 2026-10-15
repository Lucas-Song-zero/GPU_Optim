# --- toolchain ----
NVCC ?= /usr/local/cuda-13.3/bin/nvcc
ARCH = sm_120
CXXSTD = c++20

# --- sources & includes ---
INCLUDES = -I src
SRC = scripts/gemm.cu src/runner.cu
HEADERS = $(wildcard src/*.cuh src/GEMM/*.cuh)
LIBS = -lcublas

# --- targets ---
TARGET = gemm
DEBUG_TARGET = gemm_dbg

# --- flags ---
COMMON = -arch=$(ARCH) -std=$(CXXSTD) $(INCLUDES)
RELEASE = $(COMMON) -O3
DEBUG = $(COMMON) -G -lineinfo

.PHONY: all debug run run-debug sanitize clean

all:
	$(NVCC) $(COMMON) -o gemm $(SRC) $(LIBS)
# $(TARGET): $(SRC) $(HEADERS)
# $(NVCC) $(RELEASE) -o $@ $(SRC) $(LIBS)

debug:
	$(NVCC) $(DEBUG) -o gemm_dbg $(SRC) $(LIBS)

KERNEL ?= 4

run: $(TARGET)
	./$(TARGET) $(KERNEL)
                                                                  
run-debug: $(DEBUG_TARGET)
	./$(DEBUG_TARGET) $(KERNEL)                               
                                                                  
sanitize: $(DEBUG_TARGET)
	/usr/local/cuda-13.3/bin/compute-sanitizer --tool memcheck ./$(DEBUG_TARGET) $(KERNEL)                                     
                  
clean:                                                          
	rm -f $(TARGET) $(DEBUG_TARGET)
