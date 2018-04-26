# Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Compilers
MPI_HOME=/ivylogin/home/spotluri/MPI/mvapich2-2.2/install
CUDA_HOME=/usr/local/cuda-9.0
NVSHMEM_HOME=/ivylogin/home/spotluri/NVSHMEM/df-nvshmem-prototype

MPICC=$(MPI_HOME)/bin/mpicc
NVCC=$(CUDA_HOME)/bin/nvcc

# Flags
CFLAGS=-O3 -Wall 
NVCCFLAGS=-O3 -arch=compute_35 -code=sm_35
NVCCFLAGS_DF=-O3 -arch=compute_35 -code=sm_35 -rdc=true
CUDA_INCLUDES=-I$(CUDA_HOME)/include -I./
MPI_INCLUDES=-I$(MPI_HOME)/include 
SHMEM_INCLUDES=-I$(NVSHMEM_HOME)/ 

LDFLAGS=-L$(CUDA_HOME)/lib64 -lcudart -lcuda 
SHMEM_LDFLAGS=-L$(NVSHMEM_HOME) -lnvshmem
MPI_LDFLAGS=-L$(MPI_HOME)/lib -lmpich

# Commands

all: shmem_sync_latency shmem_latency 

shmem_sync_latency: shmem_sync_latency.cu
	$(NVCC) $(NVCCFLAGS_DF) $(MPI_INCLUDES) $(SHMEM_INCLUDES) $(CUDA_INCLUDES) -o $@ $? $(LDFLAGS) $(MPI_LDFLAGS) $(SHMEM_LDFLAGS)

shmem_latency: shmem_latency.cu
	$(NVCC) $(NVCCFLAGS_DF) $(MPI_INCLUDES) $(SHMEM_INCLUDES) $(CUDA_INCLUDES) -o $@ $? $(LDFLAGS) $(MPI_LDFLAGS) $(SHMEM_LDFLAGS)

clean:
	rm -rf shmem_sync_latency shmem_latency *.o
