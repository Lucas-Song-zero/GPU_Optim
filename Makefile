# --- toolchain ----
NVCC ?= /usr/local/cuda-13.3/bin/nvcc
NCU ?= /usr/local/cuda-13.3/bin/ncu
ARCH = sm_120
CXXSTD = c++20

# --- sources & includes ---
INCLUDES = -I src
SRC = scripts/gemm.cu src/runner.cu
HEADERS = $(wildcard src/*.cuh src/GEMM/*.cuh)
LIBS = -lcublas

# --- targets ---
BUILD_DIR = build
TARGET = $(BUILD_DIR)/gemm
DEBUG_TARGET = $(BUILD_DIR)/gemm_dbg
NCU_TARGET = $(BUILD_DIR)/gemm_ncu

# --- flags ---
COMMON = -arch=$(ARCH) -std=$(CXXSTD) $(INCLUDES)
RELEASE = $(COMMON) -O3 -lineinfo
DEBUG = $(COMMON) -G -lineinfo

.PHONY: all debug run run-debug sanitize clean profile

all:
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(COMMON) -o $(TARGET) $(SRC) $(LIBS)
# $(TARGET): $(SRC) $(HEADERS)
# $(NVCC) $(RELEASE) -o $@ $(SRC) $(LIBS)

debug:
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(DEBUG) -o $(DEBUG_TARGET) $(SRC) $(LIBS)

KERNEL ?= 4
# support comma-separated list, e.g. KERNEL=1,2,3,4
comma := ,
empty :=
space := $(empty) $(empty)
KERNELS := $(subst $(comma),$(space),$(KERNEL))

PROFILE_M ?= 4096
PROFILE_N ?= 4096
PROFILE_K ?= 4096
PROFILE_WARMUP ?= 10
# PROFILE_FORMAT: rep (ncu-rep only) | txt (default) | csv | all (txt+csv)
PROFILE_FORMAT ?= txt
PROFILE_REP_DIR = results/profiles/ncu-rep
PROFILE_TXT_DIR = results/profiles/txt
PROFILE_CSV_DIR = results/profiles/csv

run: $(TARGET)
	@for k in $(KERNELS); do \
		echo "=== kernel $$k ==="; \
		./$(TARGET) --kernel $$k; \
	done
                                                                  
run-debug: $(DEBUG_TARGET)
	@for k in $(KERNELS); do \
		echo "=== kernel $$k ==="; \
		./$(DEBUG_TARGET) --kernel $$k; \
	done

compile_ncu: 
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(RELEASE) -o $(NCU_TARGET) $(SRC) $(LIBS)
                                                                  
profile: compile_ncu
	@mkdir -p $(PROFILE_REP_DIR) $(PROFILE_TXT_DIR) $(PROFILE_CSV_DIR)
	@for k in $(KERNELS); do \
		echo "=== profiling kernel $$k ==="; \
		$(NCU) --profile-from-start off --launch-count 1 \
			--section SpeedOfLight \
			--section ComputeWorkloadAnalysis \
			--section MemoryWorkloadAnalysis \
			--section Occupancy \
			--section SchedulerStats \
			--section WarpStateStats \
			--force-overwrite \
			-o $(PROFILE_REP_DIR)/kernel_$${k}_$(PROFILE_M) \
			./$(NCU_TARGET) --kernel $$k --M $(PROFILE_M) --N $(PROFILE_N) \
				--K $(PROFILE_K) --warmup $(PROFILE_WARMUP) --mode profile; \
		case "$(PROFILE_FORMAT)" in \
			*txt*|*all*) $(NCU) -i $(PROFILE_REP_DIR)/kernel_$${k}_$(PROFILE_M).ncu-rep \
				--page details --print-details all > $(PROFILE_TXT_DIR)/kernel_$${k}_$(PROFILE_M).txt; \
				echo "Warp Stall Breakdown (cycles/issue):" >> $(PROFILE_TXT_DIR)/kernel_$${k}_$(PROFILE_M).txt; \
				$(NCU) -i $(PROFILE_REP_DIR)/kernel_$${k}_$(PROFILE_M).ncu-rep --page raw 2>/dev/null | grep "smsp__average_warps_issue_stalled_.*_per_issue_active.ratio" >> $(PROFILE_TXT_DIR)/kernel_$${k}_$(PROFILE_M).txt ;; \
		esac; \
		case "$(PROFILE_FORMAT)" in \
			*csv*|*all*) $(NCU) -i $(PROFILE_REP_DIR)/kernel_$${k}_$(PROFILE_M).ncu-rep \
				--csv --page details > $(PROFILE_CSV_DIR)/kernel_$${k}_$(PROFILE_M).csv ;; \
		esac; \
	done

sanitize: $(DEBUG_TARGET)
	@for k in $(KERNELS); do \
		/usr/local/cuda-13.3/bin/compute-sanitizer --tool memcheck ./$(DEBUG_TARGET) --kernel $$k; \
	done                                     
                  
clean:                                                          
	rm -rf $(BUILD_DIR)
