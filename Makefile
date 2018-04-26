# Compilers

NVCC := ${CUDA_HOME}/bin/nvcc
NVCC_CFLAGS := -O3 -gencode arch=compute_35,code=sm_35 -rdc=true -I$(CUDA_HOME)/include
OPTIONS :=

ifeq ($(TRACING),1)
OPTIONS := $(OPTIONS) -D_ENABLE_TRACING_
ifeq ($(BLOCKLEVEL_TRACING),1)
OPTIONS := $(OPTIONS) -D_BLOCKLEVEL_TRACING_
endif
endif

ifeq ($(IDEALIZE),1)
OPTIONS := $(OPTIONS) -D_ENABLE_IDEALIZE_COMM_
endif

ifeq ($(OPENSHMEM),1)
OPTIONS := $(OPTIONS) -D_USE_OPENSHMEM_
NVCC_CFLAGS := $(NVCC_CFLAGS) -I$(OPENSHMEM_HOME)/include
else
NVCC_CFLAGS := $(NVCC_CFLAGS) -I$(MPI_HOME)/include
endif

# Commands
all: libnvshmem.a

libnvshmem.a: nvshmem.cu nvshmem_kernels.cu nvshmem_device.cu

	$(NVCC) $(NVCC_CFLAGS) -I$(MPI_HOME)/include $(OPTIONS) -c nvshmem.cu -o nvshmem.o
	$(NVCC) $(NVCC_CFLAGS) -I$(CUDA_HOME)/include $(OPTIONS) -c nvshmem_device.cu -o nvshmem_device.o
	$(NVCC) $(NVCC_CFLAGS) -I$(CUDA_HOME)/include $(OPTIONS) -c nvshmem_kernels.cu -o nvshmem_kernels.o
	$(NVCC) $(NVCC_CFLAGS) -I$(CUDA_HOME)/include  $(OPTIONS) -c nvshmem_cuda_constants.cu -o nvshmem_cuda_constants.o
	$(NVCC) -arch=sm_35 -lib nvshmem_device.o nvshmem.o nvshmem_kernels.o nvshmem_cuda_constants.o -o libnvshmem.a

clean:
	rm -rf *.o libnvshmem.a
