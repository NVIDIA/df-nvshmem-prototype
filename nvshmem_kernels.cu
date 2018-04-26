/****
 * Copyright (c) 2014, NVIDIA Corporation.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright notice,
 *      this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the NVIDIA Corporation Lawrence Berkeley National 
 *      Laboratory, the U.S. Department of Energy, nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * The U.S. Department of Energy funded the development of this software 
 * under subcontract 7078610 with Lawrence Berkeley National Laboratory.
 * 
 ****/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "nvshmem.h"
#include "nvshmem_constants.h"

extern int _nvshmem_npes;
extern int _nvshmem_mype;

#define CUDA_CHECK(stmt)                                \
do {                                                    \
    cudaError_t result = (stmt);                        \
    if (cudaSuccess != result) {                        \
        fprintf(stderr, "[%s:%d] [%d] cuda failed with %s \n",   \
         __FILE__, __LINE__, _nvshmem_mype, cudaGetErrorString(result));\
        exit(-1);                                       \
    }                                                   \
    assert(cudaSuccess == result);                      \
} while (0)

__global__ void launch_gpu_sync (int bvalue, int **baddr_array, int mype) {
    int tid = threadIdx.x;
    volatile long long int start, stop;

    if (mype == 0) {
        /*there are npes number of threads*/
        start = clock64();
        while (*((volatile int *)(baddr_array[0] + tid + 1)) != bvalue) {
            stop = clock64();
            if ((stop - start) > 1000000) {
                printf("[0<-%d] timeout, breaking at root actual: %d expected: %d \n", (tid+1), *((volatile int *)(baddr_array[0] + tid + 1)), bvalue);
                break;
            }
        }
        _nvshmem_global_time = stop;

        __syncthreads();

        *((volatile int *)(baddr_array[tid + 1])) = bvalue;
        __threadfence_system();
    } else {
        /*there is just one thread*/
        *((volatile int *) (baddr_array[0] + mype)) = bvalue;
        __threadfence_system();

        start = clock64();
        while (*((volatile int *) (baddr_array[mype])) != bvalue) {
            stop = clock64();
            if ((stop - start) > 1000000) {
                printf("timeout, breaking at rank %d expected: %d actual: %d \n", mype, bvalue, *((volatile int *) (baddr_array[mype])));
                break;
            }
        }
        _nvshmem_global_time = stop;
    }

    __syncthreads();
}

void launch_gpu_sync_wrapper(int bvalue, int **baddr_array, int mype)
{
   int threads = (_nvshmem_mype == 0) ? (_nvshmem_npes - 1) : 1;

   launch_gpu_sync<<<1, threads>>>(bvalue, baddr_array, _nvshmem_mype);

   CUDA_CHECK(cudaGetLastError());
}
