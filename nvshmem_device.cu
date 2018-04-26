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

#include "stdio.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "nvshmem_device.h"

__constant__ void *_nvshmem_heap_pointers_d[16]; 
__constant__ void *_nvshmem_my_heap_pointer_d; 
__constant__ int _nvshmem_mype_d;
__constant__ int _nvshmem_npes_d;
__constant__ int _nvshmem_enable_tracing_d;

__constant__ void *_nvshmem_log_buffer_d;
__constant__ int _nvshmem_log_block_size_d;
__device__ long long int _nvshmem_global_time;

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

__device__ int log_buffer_count_d;
__device__ int log_record_limit_d;
__device__ log_buffer_t *log_full_d;
__device__ log_buffer_t *log_free_d;
__device__ log_data_t **log_active_d;
__device__ volatile unsigned int *free_tail_d;
__device__ volatile unsigned int *free_head_d;
__device__ volatile unsigned int *full_head_d;
__device__ volatile unsigned int *full_tail_d;
__device__ volatile unsigned int *overflow_flag_d;
__device__ volatile unsigned int *trace_call_conflict_d;
__device__ unsigned long long int log_blockid_d;
__device__ int ngrids_d;
__device__ int lngrids_d;
__device__ int nblocks_d;
__device__ int lnblocks_d;
__device__ int lnthreads_d;
#endif

#if defined(_ENABLE_TRACING_)
enum {
    PUT_int = 0,
    PUT_float,
    PUT_double,
    GET_int,
    GET_float,
    GET_double,
    WAIT_int,
    WAIT_UNTIL_int
};
#endif 

#ifndef NONVOLATILE_UPDATES
#define VOLATILE_POINTER(x) ((volatile typeof(*x) *) x)
#else
#define VOLATILE_POINTER(x) (x)
#endif

#if defined(_ENABLE_TRACING_)

#if defined(_BLOCKLEVEL_TRACING_)
#define FLUSH_LOG(index, mygridid)                                                      \
  do {                                                                                  \
        log_data_t *logdata;                                                            \
        unsigned int log_id;                                                            \
        logdata = log_active_d[index];                                                  \
        if (((void *)logdata != (void *) 0x1) 						\
	    && ((void *)logdata != NULL)						\
	    && (logdata->gridid == mygridid))						\
        {										\
            log_id = logdata->log_index;	                                        \
            log_full_d[log_id].status = logdata->filled_records;                        \
            log_active_d[index] = NULL;                                                 \
            __threadfence_system();                                                     \
        }										\
  } while(0)

#define COMPUTE_LOG_BLOCKINDEX(gridid, blockid, index)                                  \
  do {											\
     unsigned long long int gridid_mod, blockid_mod;				 	\
     gridid_mod = gridid&(ngrids_d - 1);                                                \
     blockid_mod = blockid&(nblocks_d - 1);                                             \
     *index = (gridid_mod << lnblocks_d)                               			\
                + blockid_mod;                                   			\
  } while (0)

#define GET_TRACE_BUF(index, mygridid, myrecord)                      			\
  do {                                                                                  \
     unsigned int log_id;                                                               \
     log_data_t *logdata;                                                               \
											\
     *myrecord = NULL;									\
     logdata = log_active_d[index];                                                     \
     if ( logdata != NULL) {                                                            \
         /*check if another block with same index is using the slot*/                   \
         if (logdata->gridid != mygridid) {            					\
             *trace_call_conflict_d = 1;                         			\
         } else {                                                                       \
             if (logdata->filled_records >= log_record_limit_d) {             		\
	         *overflow_flag_d = 1;							\
	     } else {									\
                 log_id = atomicAdd(&logdata->filled_records, 1);        		\
             	 if (log_id >= log_record_limit_d) {                       		\
                    *overflow_flag_d = 1;						\
		 } else { 								\
             	    *myrecord = logdata->buffer + log_id;				\
		 }									\
             }										\
         }                                                                              \
      }											\
  } while (0)


#define NVSHMEM_TRACE(op_type, local_ptr, remote_ptr, datasize, target, timespent)      \
  do {                                                                                  \
     int blockid, tid;                                   				\
     unsigned long long int gridid, myindex;                                    	\
     log_record_t *record;                                                              \
                                                                                        \
     if (_nvshmem_enable_tracing_d) {							\
          asm volatile("mov.u64  %0, %gridid;" : "=l"(gridid));                              \
          blockid = blockIdx.z*(gridDim.y*gridDim.x) + blockIdx.y*gridDim.x + blockIdx.x;    \
          tid = threadIdx.z*(blockDim.y*blockDim.x) + threadIdx.y*blockDim.x + threadIdx.x;  \
                                                                                             \
          COMPUTE_LOG_BLOCKINDEX(gridid, blockid, &myindex);					\
                                                                                             \
          GET_TRACE_BUF(myindex, gridid, &record);                                  		\
             										\
          if (record != NULL) {                                                              \
                long long int tstamp;                                                        \
                asm volatile("mov.u64  %0, %globaltimer;" : "=l"(tstamp));                   \
                record->tstamp = tstamp;                                               \
                record->tid = tid;                                                	  \
                record->type = op_type;                                                \
                record->size = datasize;                                               \
                record->remote_pe = target;						  \
                record->tspent = timespent;						  \
                record->local_addr = local_ptr;                                        \
                record->remote_addr = remote_ptr;                                      \
          }                                                                                  \
     }											\
  } while(0)


#else /*threadlevel tracing*/

#define TRACE (myrecord, mytype, local_ptr, remote_ptr, mysize, mytarget, timespent)    \
  do {                                                                                  \
    long long int tstamp;                                                               \
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(tstamp));                          \
    myrecord->tstamp = tstamp;                                                          \
    myrecord->type = mytype;								\
    myrecord->local_addr = local_ptr;                                                   \
    myrecord->remote_addr = remote_ptr;                                                 \
    myrecord->size = mysize;                                                            \
    myrecord->remote_pe = target;                                                       \
    myrecord->tspent = timespent; 							\
  } while (0)

#define GET_TRACE_BUF(index, mygridid, myblockid, mytid, myrecord)                      \
  do {                                                                                  \
     unsigned int log_id;                                                               \
     log_data_t *logdata;                                                               \
     int status;                                                                        \
											\
     *myrecord = NULL;									\
     logdata = log_active_d[index];                                                     \
     /*0x1 is set on log_begin at start of the kernel or a buffer got filled up*/	\
     if ((void *)logdata == (void *)0x1) {                                              \
         log_id = atomicInc((unsigned int *)free_tail_d, log_buffer_count_d-1);         \
                                                                                        \
         status = atomicCAS((unsigned int *)&log_free_d[log_id].status, 1, 0);          \
         if (status == 0) {                                                             \
            *overflow_flag_d = 1;                          				\
         } else {                                                                       \
            logdata = log_free_d[log_id].data;                                          \
            logdata->mype = _nvshmem_mype_d;                                                    \
            logdata->tid = mytid;                                                       \
            logdata->gridid = mygridid;                                                 \
            logdata->blockid = myblockid;                                               \
            logdata->filled_records = 0;                                                \
	    logdata->log_index = log_id; 						\
											\
            /*there could be a race among threads with same index, considering it unlikely */ \
            log_active_d[index] = logdata;                                              \
                                                                                        \
            *myrecord = (log_record_t *) logdata->buffer;                               \
            logdata->filled_records++;							\
         }                                                                              \
      } else if ( logdata != NULL) {                                                    \
         /*check if another thread with same index is using the slot*/                  \
         if (logdata->blockid != myblockid || logdata->gridid != mygridid) {            \
             *trace_call_conflict_d = 1;                         			\
         } else {                                                                       \
             *myrecord = logdata->buffer + logdata->filled_records;			\
             logdata->filled_records++;							\
	     /*printf("[%d] after inserting a new record, idx: %lld gid: %lld blck: %d thrd: %d buffr: %p filled_records_ptr: %p filled records: %d \n", _nvshmem_mype_d, index, mygridid, myblockid, mytid, logdata, &logdata->filled_records, logdata->filled_records); */\
         }                                                                              \
      }											\
  } while (0)

#define FLUSH_LOG(index)                                                         	\
  do {                                                                                  \
        log_data_t *logdata;                                                            \
        unsigned int log_id;                                                            \
        logdata = log_active_d[index];                                                  \
        if (((void *)logdata != (void *) 0x1) 						\
	    && ((void *)logdata != NULL)) {						\
            log_id = logdata->log_index;	                                        \
            log_full_d[log_id].status = logdata->filled_records;                        \
            log_active_d[index] = NULL;                                                 \
            __threadfence_system();                                                     \
       }										\
  } while(0)

#define CHECK_FLUSH_FULL(index)                                                         \
  do {                                                                                  \
        log_data_t *logdata;                                                            \
        unsigned int log_id;                                                            \
        logdata = log_active_d[index];                                                  \
        if (logdata->filled_records >= log_record_limit_d) {                            \
            log_id = logdata->log_index; 						\
            log_full_d[log_id].status = logdata->filled_records;                        \
            log_active_d[index] = (log_data_t *)0x1;                                    \
            __threadfence_system();                                                     \
	}										\
  } while(0)

#define COMPUTE_LOG_INDEX(gridid, blockid, tid, index)                                  \
  do {											\
     unsigned long long int gridid_mod, blockid_mod;				 	\
     gridid_mod = gridid&(ngrids_d - 1);                                                \
     blockid_mod = blockid&(nblocks_d - 1);                                             \
     *index = ((gridid_mod << lnblocks_d) << lnthreads_d)                               \
                + (blockid_mod << lnthreads_d) + tid;                                   \
  } while (0)

#define NVSHMEM_TRACE(op_type, local_ptr, remote_ptr, datasize, target, timespent)      \
  do {                                                                                  \
     int blockid, tid;                                   				\
     unsigned long long int gridid, myindex;                                    	\
     log_record_t *record;                                                              \
                                                                                        \
     if (_nvshmem_enable_tracing_d) {							\
          asm volatile("mov.u64  %0, %gridid;" : "=l"(gridid));                              \
          blockid = blockIdx.z*(gridDim.y*gridDim.x) + blockIdx.y*gridDim.x + blockIdx.x;    \
          tid = threadIdx.z*(blockDim.y*blockDim.x) + threadIdx.y*blockDim.x + threadIdx.x;  \
                                                                                             \
          COMPUTE_LOG_INDEX(gridid, blockid, tid, &myindex);					\
                                                                                             \
          GET_TRACE_BUF(myindex, gridid, blockid, tid, &record);                             \
          if (record != NULL) {                                                              \
                long long int tstamp;                                                        \
                asm volatile("mov.u64  %0, %globaltimer;" : "=l"(tstamp));                   \
                record->tstamp = tstamp;                                               \
                record->type = op_type;                                                \
                record->size = datasize;                                               \
                record->remote_pe = target;						  \
                record->tspent = timespent;						  \
                record->local_addr = local_ptr;                                        \
                record->remote_addr = remote_ptr;                                      \
                CHECK_FLUSH_FULL(myindex);                                                   \
          }                                                                                  \
     }											\
  } while(0)

#endif /*_BLOCKLEVEL_TRACING_*/

__device__ void nvshmem_trace_begin ()
{
     int blockid, tid;
     unsigned long long int gridid, myindex;                                                      

     if (!_nvshmem_enable_tracing_d) { 
	return;
     }

     asm volatile("mov.u64  %0, %gridid;" : "=l"(gridid));
     blockid = blockIdx.z*(gridDim.y*gridDim.x) + blockIdx.y*gridDim.x + blockIdx.x;
     tid = threadIdx.z*(blockDim.y*blockDim.x) + threadIdx.y*blockDim.x + threadIdx.x;

#if defined (_BLOCKLEVEL_TRACING_)
     if (tid == 0) {
	 int status, log_id;
         log_data_t *logdata;

	 COMPUTE_LOG_BLOCKINDEX(gridid, blockid, &myindex);

	 if ((void *)log_active_d[myindex] != NULL) {
             *trace_call_conflict_d = 1;
     	 } else {
             log_id = atomicInc((unsigned int *)free_tail_d, log_buffer_count_d-1);

             status = atomicCAS((unsigned int *)&log_free_d[log_id].status, 1, 0);
             if (status == 0) {
                 *overflow_flag_d = 1;
             } else {                                                                       
                 logdata = log_free_d[log_id].data;
                 logdata->mype = _nvshmem_mype_d;
                 logdata->gridid = gridid;
                 logdata->blockid = blockid;
                 logdata->filled_records = 0;
	         logdata->log_index = log_id;
											
                 /*there could be a race among blocks with same index, considering it unlikely */ 
                 log_active_d[myindex] = logdata;
	     }
         }                                                                              
	 __threadfence_block();
     }

     __syncthreads();
#else
                                                                                        
     COMPUTE_LOG_INDEX(gridid, blockid, tid, &myindex); 

     if ((void *)log_active_d[myindex] == (void *) 0x1) {
         *trace_call_conflict_d = 1;
     } else {                                    
         log_active_d[myindex] = (log_data_t *)0x1;                                   
     }
#endif
}

__device__ void nvshmem_trace_end ()
{
     int blockid, tid;
     unsigned long long int gridid, myindex;                                                      

     if (!_nvshmem_enable_tracing_d) {
        return;
     }

     asm volatile("mov.u64  %0, %gridid;" : "=l"(gridid));
     blockid = blockIdx.z*(gridDim.y*gridDim.x) + blockIdx.y*gridDim.x + blockIdx.x;
     tid = threadIdx.z*(blockDim.y*blockDim.x) + threadIdx.y*blockDim.x + threadIdx.x;
                     
#if defined (_BLOCKLEVEL_TRACING_)
     __syncthreads();

     if (tid == 0) {
         COMPUTE_LOG_BLOCKINDEX(gridid, blockid, &myindex);

         FLUSH_LOG(myindex, gridid); 

         log_active_d[myindex] = NULL;

         __threadfence_block();
     }

     __syncthreads();
#else
     COMPUTE_LOG_INDEX(gridid, blockid, tid, &myindex);

     FLUSH_LOG(myindex); 

     log_active_d[myindex] = NULL;
#endif 
}

#else

#define NVSHMEM_TRACE(type, local_ptr, remote_ptr, size, target, timespent) 		
__device__ void nvshmem_trace_begin() {}
__device__ void nvshmem_trace_end() {}

#endif /*_ENABLE_TRACING_*/

__device__ void nvshmem_quiet() 
{
#ifndef _ENABLE_IDEALIZE_COMM_          
    __threadfence_system();
#endif
}

#ifndef _ENABLE_IDEALIZE_COMM_                                          
#define NVSHMEM_TYPE_P(Name, Type)                                      \
  __device__ void                                            		\
  nvshmem_##Name##_p (Type *target, const Type value, int pe)           \
  {                                                                     \
     NVSHMEM_TRACE(PUT_##Name, NULL, target, 1, pe, 0);		        \
									\
     volatile Type *target_actual = (volatile Type *)((char *)_nvshmem_heap_pointers_d[pe] + ((char *)target \
           -  (char *)_nvshmem_my_heap_pointer_d));			\
									\
     *(target_actual) = value;						\
  }
#else
#define NVSHMEM_TYPE_P(Name, Type)                                      \
  __device__ void                                            		\
  nvshmem_##Name##_p (Type *target, const Type value, int pe)           \
  {                                                                     \
  }
#endif									

NVSHMEM_TYPE_P (int, int)
NVSHMEM_TYPE_P (float, float)
NVSHMEM_TYPE_P (double, double)

#ifndef _ENABLE_IDEALIZE_COMM_                                          
#define NVSHMEM_TYPE_G(Name, Type)                                      \
  __device__ Type                                            \
  nvshmem_##Name##_g (Type *source, int pe)             		\
  {                                                                     \
     NVSHMEM_TRACE(GET_##Name, NULL, source, 1, pe, 0);                 \
                                                                        \
     volatile Type *source_actual = (volatile Type *)((char *)_nvshmem_heap_pointers_d[pe] + ((char *)source \
           -  (char *)_nvshmem_my_heap_pointer_d));                              \
                                                                        \
     return *source_actual;						\
  }
#else
#define NVSHMEM_TYPE_G(Name, Type)                                      \
  __device__ Type                                            \
  nvshmem_##Name##_g (Type *source, int pe)                             \
  {                                                                     \
     return 0;                                                          \
  }
#endif

NVSHMEM_TYPE_G (int, int)
NVSHMEM_TYPE_G (float, float)
NVSHMEM_TYPE_G (double, double)

#ifndef _ENABLE_IDEALIZE_COMM_						
#define NVSHMEM_TYPE_PUT(Name, Type)                                    \
  __device__ void                                            \
  nvshmem_##Name##_put (Type *target, Type *source, int len, int pe)    \
  {                                                                     \
     NVSHMEM_TRACE(PUT_##Name, source, target, len, pe, 0);             \
                                                                        \
     volatile Type *target_actual = (volatile Type *)((char *)_nvshmem_heap_pointers_d[pe] + ((char *)target \
                             -  (char *)_nvshmem_my_heap_pointer_d));		\
									\
     int i;								\
     for (i = 0; i < len; i++) {					\
        *(target_actual + i) = *(source + i);				\
     }									\
  }
#else
#define NVSHMEM_TYPE_PUT(Name, Type)                                    \
  __device__ void                                            \
  nvshmem_##Name##_put (Type *target, Type *source, int len, int pe)    \
  {                                                                     \
  }
#endif									

NVSHMEM_TYPE_PUT (int, int);
NVSHMEM_TYPE_PUT (float, float);
NVSHMEM_TYPE_PUT (double, double);

#ifndef _ENABLE_IDEALIZE_COMM_						
#define NVSHMEM_TYPE_GET(Name, Type)                                    \
  __device__ void                                            \
  nvshmem_##Name##_get (Type *target, Type *source, int len, int pe)    \
  {                                                                     \
     NVSHMEM_TRACE(GET_##Name, target, source, len, pe, 0);             \
                                                                        \
     volatile Type *source_actual = (volatile Type *)((char *)_nvshmem_heap_pointers_d[pe] + ((char *)source \
                             -  (char *)_nvshmem_my_heap_pointer_d));		\
									\
     int i;								\
     for (i = 0; i < len; i++) {					\
        *(target + i) = *(source_actual + i);				\
     }									\
  }
#else
#define NVSHMEM_TYPE_GET(Name, Type)                                    \
  __device__ void                                            \
  nvshmem_##Name##_get (Type *target, Type *source, int len, int pe)    \
  {                                                                     \
  }
#endif


NVSHMEM_TYPE_GET (int, int);
NVSHMEM_TYPE_GET (float, float);
NVSHMEM_TYPE_GET (double, double);

__device__ void nvshmem_int_wait(int *ivar, int cmp_value) 
{
#ifndef _ENABLE_IDEALIZE_COMM_
    long long int time = 0;
#if defined(_ENABLE_TRACING_)
    long long int time_start; 
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(time_start));                   
#endif

    while (*((volatile int *)ivar) == cmp_value) { 
    	asm volatile("mov.u64  %0, %globaltimer;" : "=l"(time));
    }
    _nvshmem_global_time = time;
#if defined(_ENABLE_TRACING_)
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(time));
    time = time - time_start;
#endif

    NVSHMEM_TRACE(WAIT_int, ivar, NULL, 1, _nvshmem_mype_d, time); 
#endif
}

__device__ void nvshmem_int_wait_until(int *ivar, int cmp, int cmp_value)
{
#ifndef _ENABLE_IDEALIZE_COMM_
    long long int time = 0;
#if defined(_ENABLE_TRACING_) 
    long long int time_start; 
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(time_start));                   
#endif

    while (*((volatile int *)ivar) != cmp_value) { 
    	asm volatile("mov.u64  %0, %globaltimer;" : "=l"(time));
    }
    _nvshmem_global_time = time;
#if defined(_ENABLE_TRACING_)
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(time));
    time = time - time_start;
#endif

    NVSHMEM_TRACE(WAIT_UNTIL_int, ivar, NULL, 1, _nvshmem_mype_d, time); 
#endif
}
