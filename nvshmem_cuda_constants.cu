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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "nvshmem_constants.h"

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

void initialize_cuda_constants () { 
    CUDA_CHECK(cudaMemcpyToSymbol(_nvshmem_mype_d, &_nvshmem_mype, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(_nvshmem_npes_d, &_nvshmem_npes, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(_nvshmem_heap_pointers_d, _nvshmem_heap_pointers, sizeof(void *)*_nvshmem_npes));
    CUDA_CHECK(cudaMemcpyToSymbol(_nvshmem_my_heap_pointer_d, &_nvshmem_my_heap_pointer, sizeof(void *)));
}

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif
