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

#ifdef __cplusplus
extern "C" {
#endif

__device__ void nvshmem_trace_begin ();
__device__ void nvshmem_trace_end ();
__device__ void nvshmem_quiet ();

#define DECL_NVSHMEM_TYPE_P(Name, Type)                      \
  __device__ void nvshmem_##Name##_p (Type *target, const Type value, int pe)

DECL_NVSHMEM_TYPE_P (int, int);
DECL_NVSHMEM_TYPE_P (float, float);
DECL_NVSHMEM_TYPE_P (double, double);

#define DECL_NVSHMEM_TYPE_G(Name, Type)                      \
  __device__ Type nvshmem_##Name##_g (Type *source, int pe)

DECL_NVSHMEM_TYPE_G (int, int);
DECL_NVSHMEM_TYPE_G (float, float);
DECL_NVSHMEM_TYPE_G (double, double);

#define DECL_NVSHMEM_TYPE_PUT(Name, Type)                    \
  __device__ void nvshmem_##Name##_put (Type *target, Type *source, int len, int pe)

DECL_NVSHMEM_TYPE_PUT (int, int);
DECL_NVSHMEM_TYPE_PUT (float, float);
DECL_NVSHMEM_TYPE_PUT (double, double);

#define DECL_NVSHMEM_TYPE_GET(Name, Type)                    \
  __device__ void nvshmem_##Name##_get (Type *target, Type *source, int len, int pe)

DECL_NVSHMEM_TYPE_GET (int, int);
DECL_NVSHMEM_TYPE_GET (float, float);
DECL_NVSHMEM_TYPE_GET (double, double);

__device__ void nvshmem_int_wait(int *ivar, int cmp_value);
__device__ void nvshmem_int_wait_until(int *ivar, int cmp, int cmp_value);

#ifdef __cplusplus
}
#endif
