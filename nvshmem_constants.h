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

#ifndef SHMEM_MAIN
#define SHMEM_MAIN
#endif

#define _NVSHMEM_MAX_NPES 16
extern int _nvshmem_npes;
extern int _nvshmem_mype;
extern CUdeviceptr _nvshmem_heap_pointers[_NVSHMEM_MAX_NPES];
extern CUdeviceptr _nvshmem_my_heap_pointer;

extern int _nvshmem_enable_tracing;

extern __constant__ void *_nvshmem_heap_pointers_d[16];
extern __constant__ void *_nvshmem_my_heap_pointer_d;
extern __constant__ int _nvshmem_mype_d;
extern __constant__ int _nvshmem_npes_d;
extern __constant__ int _nvshmem_enable_tracing_d;

extern __constant__ void *_nvshmem_log_buffer_d; 
extern __constant__ int _nvshmem_log_block_size_d;

extern __device__ clock_t _nvshmem_global_time;

#if defined(_ENABLE_TRACING_)
typedef struct log_record
{
    long long int tstamp;
#if defined (_BLOCKLEVEL_TRACING_)
    int tid;
#endif
    int type;
    void *local_addr;
    void *remote_addr;
    int size;
    int remote_pe;
    long long int tspent;
} log_record_t;

typedef struct log_data
{
    int mype;
#if !defined (_BLOCKLEVEL_TRACING_)
    int tid;
#endif
    int blockid;
    unsigned long long int gridid;
    int filled_records;
    int log_index;
    log_record_t buffer[];
} log_data_t;

typedef struct log_buffer
{
    volatile int status;
    log_data_t *data;
} log_buffer_t;

extern const char *op_type[10];

extern volatile unsigned int *full_head;
extern volatile unsigned int *full_tail;
extern volatile unsigned int *free_head;
extern volatile unsigned int *free_tail;
extern volatile unsigned int *overflow_flag;
extern volatile unsigned int *trace_call_conflict;
extern volatile int backup_overflow; 

extern __device__ int log_buffer_count_d;
extern __device__ int log_record_limit_d;
extern __device__ log_buffer_t *log_full_d;
extern __device__ log_buffer_t *log_free_d;
extern __device__ log_data_t **log_active_d;
extern __device__ volatile unsigned  int *free_tail_d;
extern __device__ volatile unsigned  int *free_head_d;
extern __device__ volatile unsigned  int *full_head_d;
extern __device__ volatile unsigned  int *full_tail_d;
extern __device__ volatile unsigned  int *overflow_flag_d;
extern __device__ volatile unsigned  int *trace_call_conflict_d;
extern __device__ unsigned long long int log_blockid_d;
extern __device__ int ngrids_d;
extern __device__ int lngrids_d;
extern __device__ int nblocks_d;
extern __device__ int lnblocks_d;
extern __device__ int lnthreads_d;
#endif
