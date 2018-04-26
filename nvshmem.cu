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
#include <stdint.h>
#include <assert.h>
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#ifndef _USE_OPENSHMEM_
#include "mpi.h"
#else 
#include "shmem.h"
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include "nvshmem_constants.h"
#include "nvshmem.h"

#ifdef _USE_OPENSHMEM_
long pSync_bcast[_SHMEM_BCAST_SYNC_SIZE];
long pSync_collect[_SHMEM_COLLECT_SYNC_SIZE];
#endif

#define CU_CHECK(stmt)                                  \
do {                                                    \
    CUresult result = (stmt);                           \
    if (CUDA_SUCCESS != result) {                       \
        fprintf(stderr, "[%s:%d] [%d] cu failed with %d \n", \
         __FILE__, __LINE__, _nvshmem_mype, result);		\
        exit(-1);                                       \
    }                                                   \
    assert(CUDA_SUCCESS == result);                     \
} while (0)


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

#ifndef _USE_OPENSHMEM_
#define MPI_CHECK(stmt)                                 \
do {                                                    \
    int mpi_errno = (stmt);                             \
    if (MPI_SUCCESS != mpi_errno) {                     \
        fprintf(stderr, "[%s:%d] mpi call failed with error: %d \n",   \
         __FILE__, __LINE__, mpi_errno);                \
        exit(-1); 					\
    }                                                   \
    assert(MPI_SUCCESS == mpi_errno);                   \
} while (0)
#endif

#ifdef _ENABLE_DEBUG_
#define DEBUG_PRINT(...) 				\
do {							\
   fprintf(stderr, __VA_ARGS__); 			\
} while (0)
#else
#define DEBUG_PRINT(...) do { } while (0)
#endif

#define ERROR_PRINT(...) 				\
do {							\
   fprintf(stderr, __VA_ARGS__); 			\
} while (0)

#define HEAP_SIZE 1*1024*1024*1024;

typedef struct shmem_region
{
   void *buffer;
   void **pointers;
   int size;
   struct shmem_region *next;
} shmem_region_t;

const char *op_type[10] = {"put_int", "put_float", "put_double", "get_int", "get_float", "get_double", "wait_int", "wait_until_int"};

volatile unsigned int *full_head;
volatile unsigned int *full_tail;
volatile unsigned int *free_head;
volatile unsigned int *free_tail;
volatile unsigned int *overflow_flag;
volatile unsigned int *trace_call_conflict;
volatile int backup_overflow;

int _nvshmem_npes;
int _nvshmem_mype;
shmem_region_t *_nvshmem_region_list; 
shmem_region_t *_nvshmem_region_list_tail; 
CUdeviceptr _nvshmem_heap; 
CUdeviceptr _nvshmem_heap_pointers[_NVSHMEM_MAX_NPES]; 
CUdeviceptr _nvshmem_heap_head;
CUdeviceptr _nvshmem_heap_tail; 
CUdeviceptr _nvshmem_my_heap_pointer;
CUdeviceptr _nvshmem_barrier_sync;
CUdeviceptr *_nvshmem_barrier_sync_addr;
CUdeviceptr *_nvshmem_barrier_sync_addr_d;
int barrier_value = 1;

int _nvshmem_enable_tracing = 1;

#if defined(_ENABLE_TRACING_)
#if defined (_BLOCKLEVEL_TRACING_)
#define LOG_BUFFER_SIZE 2*1024*1024
#define LOG_BUFFER_COUNT 32
#else
#define LOG_BUFFER_SIZE 1024
#define LOG_BUFFER_COUNT 1024
#endif
#define MAX_SM_COUNT 16
#define MAX_ACTIVE_GRIDS 128
#define MAX_ACTIVE_BLOCKS_PER_GRID 256
#define MAX_ACTIVE_THREADS_PER_BLOCK 1024

int active_grids = MAX_ACTIVE_GRIDS;
int active_blocks = MAX_ACTIVE_BLOCKS_PER_GRID;
int threads_per_block = MAX_ACTIVE_THREADS_PER_BLOCK;

int log_data_size;
int log_buffer_count;
int log_storage_size = 16;
int log_storage_bytes;
unsigned long long int log_storage_counter = 0;

log_buffer_t *log_full;
log_buffer_t *log_free;
log_buffer_t *log_full_devptr;
log_buffer_t *log_free_devptr;
log_data_t **log_active;

void *log_data;
void *log_data_host;

unsigned int *ring_counters;
unsigned int *ring_counters_devptr;

#define COUNTER_ALIGN_SIZE 64

pthread_t log_thread;
volatile int log_thread_exit_flag = 0;
#endif

long long int _nvshmem_heap_size = HEAP_SIZE;

typedef struct {
    CUipcMemHandle heap_handle;
    CUipcMemHandle _nvshmem_barrier_sync_handle;
} start_info_t;

CUcontext device_context;

int nvshmem_my_pe () 
{
    return _nvshmem_mype;
}

int nvshmem_n_pes () 
{
    return _nvshmem_npes;
}

extern void launch_gpu_sync_wrapper(int bvalue, int **baddr_array, int mype);
extern void initialize_cuda_constants();

#if defined(_ENABLE_TRACING_)
void *log_backup (void *ptr) {
   log_data_t *data;
   log_data_t *buf;
   int bytes;
   char filename[50]; 
   FILE *fd = NULL; 
   size_t flushed; 
   CUstream copystream; 

   CU_CHECK(cuCtxSetCurrent(device_context));

   CU_CHECK(cuStreamCreate(&copystream, CU_STREAM_NON_BLOCKING));          
 
   memset(filename, 0, 50);
   sprintf(filename, "nvshmem_tracefile_%d", getpid());
   fprintf(stdout, "[%d] Tracing enabled . . . \n", _nvshmem_mype);
   fflush(stdout);

   do {
       if (!backup_overflow) {
          if (*((volatile int *) &log_full[*full_tail].status) != 0) {
	
              data = log_full[*full_tail].data;

              buf = (log_data_t *)((char *)log_data_host + log_storage_counter);
              bytes = sizeof(log_data_t) + log_full[*full_tail].status*(sizeof(log_record_t));
  
              CU_CHECK(cuMemcpyDtoHAsync((void *)buf, (CUdeviceptr)data, bytes, copystream));
   	      CU_CHECK(cuStreamSynchronize(copystream));
 
	      //fprintf(stderr, "[%d] log device -> host size: %d bytes %d records (sizeofheader: %d sizeofrecord: %d ) \n", _nvshmem_mype, bytes, log_full[*full_tail].status, sizeof(log_data_t), sizeof(log_record_t));

              log_storage_counter += bytes;
              if (log_storage_counter >= (log_storage_bytes - (sizeof(log_data_t) + log_data_size))) {
	      	  if (fd == NULL) 
		  { 
		      fd = fopen (filename, "w+");
		      if (fd == NULL) { 
               	          ERROR_PRINT("log file creation failed!!\n");
                          exit(-1); 
                      }
                  }  

		  flushed = fwrite (log_data_host, 1, log_storage_counter, fd);
		  DEBUG_PRINT("flushed to file, bytes: %d \n", flushed);
		  if (flushed < log_storage_counter) {
		      backup_overflow = 1;
		  }
                  log_storage_counter = 0;  
              }

              log_full[*full_tail].status = 0;
              log_free[*full_tail].status = 1;

              *full_tail = (*full_tail + 1)%log_buffer_count;	      
          }
      }

      if (log_thread_exit_flag == 1) { 
	  break;
      }
   } while (1);

   if (!backup_overflow) {
      int i;
      for (i = 0; i<log_buffer_count; i++) { 
         if (log_full[i].status != 0) {
             data = log_full[i].data;
 
             buf = (log_data_t *)((char *)log_data_host + log_storage_counter);
             bytes = sizeof(log_data_t) + log_full[i].status*(sizeof(log_record_t));

             CU_CHECK(cuMemcpyDtoHAsync((void *)buf, (CUdeviceptr)data, bytes, copystream));
             CU_CHECK(cuStreamSynchronize(copystream));
 
             log_storage_counter += bytes;
             if (log_storage_counter >= (log_storage_bytes - (sizeof(log_data_t) + log_data_size))) {
	      	  if (fd == NULL) 
		  { 
		      fd = fopen (filename, "w+");
		      if (fd == NULL) { 
               	          ERROR_PRINT("log file creation failed!!\n");
                          exit(-1); 
                      }
                  }  

		  flushed = fwrite (log_data_host, 1, log_storage_counter, fd);
                  if (flushed < log_storage_counter) {
                      backup_overflow = 1;
                      break;
                  }
                  log_storage_counter = 0;
             }

             log_full[i].status = 0;
         }
      }
   } 

   if (log_storage_counter > 0) {
       if (fd == NULL) 
       { 
           fd = fopen (filename, "w+");
           if (fd == NULL) { 
               ERROR_PRINT("log file creation failed!!\n");
               exit(-1); 
           }
       }  

       flushed = fwrite (log_data_host, 1, log_storage_counter, fd);
       if (flushed < log_storage_counter) {
           backup_overflow = 1;
       }
   } 

   if (fd != NULL) {
      fclose(fd); 
   } else { 
      DEBUG_PRINT("[%d] No traces logged... \n", _nvshmem_mype);
      fflush(stdout);
   }

   return NULL;
}

void process_logs () 
{
   log_data_t *data;
   log_record_t *log_record; 
   char print_record[1024];  
   int expected_bytes, actual_bytes;
   char rfilename[50]; 
   char wfilename[50]; 
   FILE *rfd, *wfd; 
   int i;

   memset(rfilename, 0, 50);
   memset(wfilename, 0, 50);
   sprintf(rfilename, "nvshmem_tracefile_%d", getpid());

#ifndef _USE_OPENSHMEM_
   pid_t root_pid;
#else
   static pid_t root_pid;
#endif
   if (_nvshmem_mype == 0) { 
      root_pid = getpid(); 
   }

   fprintf(stdout, "[%d] LOG STATUS: overflow: %d conflict: %d backup overflow: %d \n", _nvshmem_mype, *overflow_flag, *trace_call_conflict, backup_overflow);
   if (_nvshmem_mype == 0) { 
       if (*overflow_flag != 0) {
           fprintf(stderr, "Note!! overflow occured due to limited number/size of log buffers, can increase by setting NVSHMEM_LOG_BUFFER_COUNT, NVSHMEM_LOG_BUFFER_SIZE in environment (current values: %d, %d) \n", log_buffer_count, log_data_size);
       }
       if (*trace_call_conflict != 0) {
           fprintf(stderr, "Note!! call conflict happened as two blocks picked up the same log buffer slot (max active blocks: 256 max active grids: 128), can be set using NVSHMEM_LOG_ACTIVE_BLOCKS, NVSHMEM_LOG_ACTIVE_GRIDS in environment \n");
       }
       if (backup_overflow) {
           fprintf(stderr, "Note!! backup buffer overflow happened \n"); 
       }

   }
   fflush(stdout);

#ifndef _USE_OPENSHMEM_
   MPI_CHECK(MPI_Bcast (&root_pid, sizeof(pid_t), MPI_BYTE, 0, MPI_COMM_WORLD));
#else
   shmem_broadcast32(&root_pid, &root_pid, 1, 0, 0, 0, _nvshmem_npes, pSync_bcast); 
#endif

   sprintf(wfilename, "nvshmem_processed_tracefile_%d_%d", root_pid, _nvshmem_mype);
  
   rfd = fopen (rfilename, "r");
   if (rfd == NULL) { 
       goto nolog; 
   }

   wfd = fopen (wfilename, "w+");
   if (wfd == NULL) { 
       ERROR_PRINT("procesed log file creation failed!!\n");
       exit(-1); 
   }

   sprintf(print_record, "# timestamp, gridid, blockid, tid, type, local_address, remote_address, size, remote_pe, timespent\n");
   expected_bytes = strlen(print_record); 
   actual_bytes = fwrite (print_record, 1, expected_bytes, wfd);
   assert(expected_bytes == actual_bytes);

   data = (log_data_t *) malloc (sizeof(log_data_t) + log_data_size);
   
   expected_bytes = 1; 
   actual_bytes = fread (data, sizeof(log_data_t), expected_bytes, rfd); 
   assert(expected_bytes == actual_bytes);

   while (actual_bytes != 0) {
        expected_bytes = data->filled_records;
        actual_bytes = fread ((void *)(data + 1), sizeof(log_record_t), expected_bytes, rfd); 
        assert(expected_bytes == actual_bytes);
 
        for (i=0; i<data->filled_records; i++) {
	    log_record = data->buffer + i;

            memset(print_record, 0, 1024);
#if defined (_BLOCKLEVEL_TRACING_)
	    sprintf(print_record, "%lld, %llu, %d, %d, %s, %p, %p, %d, %d, %lld\n", 
		     log_record->tstamp, 
		     data->gridid,
		     data->blockid, 
		     log_record->tid, 
		     op_type[log_record->type],
	             log_record->local_addr, 
		     log_record->remote_addr, 
		     log_record->size, 
		     log_record->remote_pe, 
		     log_record->tspent);
#else
	    sprintf(print_record, "%lld, %llu, %d, %d, %s, %p, %p, %d, %d, %lld\n", 
		     log_record->tstamp, 
		     data->gridid,
		     data->blockid, 
		     data->tid, 
		     op_type[log_record->type],
	             log_record->local_addr, 
		     log_record->remote_addr, 
		     log_record->size, 
		     log_record->remote_pe, 
		     log_record->tspent);
#endif
            expected_bytes = strlen(print_record);
            actual_bytes = fwrite (print_record, 1, expected_bytes, wfd);
	    assert(actual_bytes == expected_bytes);
	} 

   	actual_bytes = fread (data, sizeof(log_data_t), 1, rfd); 
   }

   fclose(rfd); 
   fclose(wfd); 

   if (remove(rfilename) != 0) { 
	ERROR_PRINT("error cleaning up temp log file \n");
   }

   fprintf(stdout, "[%d] LOGGED TO FILE : %s \n", _nvshmem_mype, wfilename);
   fflush(stdout);

end: 
   return;   

nolog:
   goto end;
}


/*asssumes power of 2*/
static inline int intlog2(int x) {
    int result = 0, temp = x;
    while (temp >>= 1) result++;
    return result;
}
#endif

CUdevice my_dev;
start_info_t start_info;

void nvstart_pes () 
{
    int i, dev_id, can_access = 0;
    char *value = NULL;
    start_info_t *share_info; 
    CUdevice *dev_list;
    
#ifndef _USE_OPENSHMEM_
    MPI_Comm_rank (MPI_COMM_WORLD, (int *) &_nvshmem_mype); 
    MPI_Comm_size (MPI_COMM_WORLD, (int *) &_nvshmem_npes);
#else
    _nvshmem_mype = shmem_my_pe(); 
    _nvshmem_npes = shmem_n_pes();
#endif

#ifdef _USE_OPENSHMEM_
    for (i=0; i<_SHMEM_BCAST_SYNC_SIZE; i++)   
       pSync_bcast[i] = _SHMEM_SYNC_VALUE;
    for (i=0; i<_SHMEM_COLLECT_SYNC_SIZE; i++)   
       pSync_collect[i] = _SHMEM_SYNC_VALUE;
#endif

    CUresult result = cuCtxGetCurrent(&device_context);
    if (result != CUDA_SUCCESS) {
	CUDA_CHECK(cudaGetDevice (&dev_id));
        DEBUG_PRINT("a cuda context is not active, initializing context of the current selected device: %d.\n", dev_id);
    }
    /*make sure context is properly initialized*/
    cudaFree(0);

    CU_CHECK(cuCtxGetDevice (&my_dev));
    CU_CHECK(cuCtxGetCurrent(&device_context));
    CU_CHECK(cuMemAlloc(&_nvshmem_heap, _nvshmem_heap_size));

#ifndef _USE_OPENSHMEM_
    dev_list = (int *) malloc (_nvshmem_npes*sizeof(CUdevice)); 

    MPI_CHECK(MPI_Allgather (&my_dev, sizeof(CUdevice), MPI_BYTE, dev_list, 
		sizeof(CUdevice), MPI_BYTE, MPI_COMM_WORLD));
#else 
    dev_list = (CUdevice *)shmalloc (_nvshmem_npes*sizeof(CUdevice)); 

    shmem_fcollect64((void *)dev_list, (const void *)&my_dev, 1, 0, 0, _nvshmem_npes, pSync_collect); 
#endif 

    /*check peer-to-peer access among the GPUs*/ 
    for (i = 0; i < _nvshmem_npes; i++) { 
        if (i == _nvshmem_mype)
	    continue;

        if (my_dev == dev_list[i]) 
	    continue;

        CU_CHECK(cuDeviceCanAccessPeer(&can_access, my_dev, dev_list[i]));
        if (can_access != 1) { 
	    DEBUG_PRINT("nvshmem requires peer-to-peer access between all the GPUs used \n"); 
	    exit(-1);
	}
    }
    
    value = getenv("NVSHMEM_HEAP_SIZE");
    if (value != NULL) { 
	_nvshmem_heap_size = atol(value);
    }

    _nvshmem_heap_head = _nvshmem_heap_tail = _nvshmem_heap;

    CU_CHECK(cuMemAlloc(&_nvshmem_barrier_sync, 2*_nvshmem_npes*sizeof(int)));
    CU_CHECK(cuMemsetD32(_nvshmem_barrier_sync, 0, 2*_nvshmem_npes));

    CU_CHECK(cuCtxSynchronize());

#ifndef _USE_OPENSHMEM_
    start_info_t start_info;
    share_info = (start_info_t *) malloc (sizeof(start_info_t)*_nvshmem_npes);

    CU_CHECK(cuIpcGetMemHandle(&start_info.heap_handle, _nvshmem_heap));
    CU_CHECK(cuIpcGetMemHandle(&start_info._nvshmem_barrier_sync_handle, _nvshmem_barrier_sync));

    MPI_CHECK(MPI_Allgather((void *)&start_info, sizeof(start_info_t), MPI_BYTE, 
		(void *)share_info, sizeof(start_info_t), MPI_BYTE,
                MPI_COMM_WORLD));
#else
    share_info = (start_info_t *) shmalloc(sizeof(start_info_t)*_nvshmem_npes);

    CU_CHECK(cuIpcGetMemHandle(&start_info.heap_handle, _nvshmem_heap));
    CU_CHECK(cuIpcGetMemHandle(&start_info._nvshmem_barrier_sync_handle, _nvshmem_barrier_sync));

    shmem_fcollect32((void *)share_info, (const void *)&start_info, sizeof(start_info_t)/4, 0, 0, _nvshmem_npes, pSync_collect);
#endif

    _nvshmem_barrier_sync_addr = (CUdeviceptr *) malloc (2*sizeof(CUdeviceptr)*_nvshmem_npes);

    CU_CHECK(cuMemAlloc ((CUdeviceptr *) &_nvshmem_barrier_sync_addr_d, 2*sizeof(CUdeviceptr)*_nvshmem_npes));

    for (i = 0; i < _nvshmem_npes; i++) { 
        if (i == _nvshmem_mype) {
            _nvshmem_heap_pointers[i] = _nvshmem_heap;
	    DEBUG_PRINT("[%d] my heap pointer %p \n", _nvshmem_mype, _nvshmem_heap_pointers[i]);
            _nvshmem_my_heap_pointer = _nvshmem_heap;

            _nvshmem_barrier_sync_addr[i] = _nvshmem_barrier_sync;
            _nvshmem_barrier_sync_addr[i + _nvshmem_npes] = _nvshmem_barrier_sync + _nvshmem_npes*sizeof(int);
       
            continue; 
        }
       
        CU_CHECK(cuIpcOpenMemHandle(&(_nvshmem_heap_pointers[i]), 
          		share_info[i].heap_handle, 
          		CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS)); 
	
	DEBUG_PRINT("[%d] heap pointer for pe: %d is %p \n", _nvshmem_mype, i, _nvshmem_heap_pointers[i]);

        CU_CHECK(cuIpcOpenMemHandle(&(_nvshmem_barrier_sync_addr[i]),
                          share_info[i]._nvshmem_barrier_sync_handle,
                          cudaIpcMemLazyEnablePeerAccess));
        _nvshmem_barrier_sync_addr[i + _nvshmem_npes] = _nvshmem_barrier_sync_addr[i] + _nvshmem_npes*sizeof(int);  
    }

    CU_CHECK(cuMemcpyHtoD((CUdeviceptr) _nvshmem_barrier_sync_addr_d, (void *) _nvshmem_barrier_sync_addr, 2*sizeof(CUdeviceptr)*_nvshmem_npes));

    if (_nvshmem_npes > _NVSHMEM_MAX_NPES) { 
	DEBUG_PRINT("problem size limited to %d processes, constant arrays declared statically \n", _NVSHMEM_MAX_NPES); 
        exit(-1);
    }

    initialize_cuda_constants();

    _nvshmem_region_list = _nvshmem_region_list_tail = NULL;

#ifndef _USE_OPENSHMEM_
    free(share_info);
    free(dev_list);
#else
    shfree(share_info);
    shfree(dev_list);
#endif

#if defined(_ENABLE_TRACING_)
    value = getenv("NVSHMEM_ENABLE_TRACING");
    if (value != NULL) {
        _nvshmem_enable_tracing = atol(value);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(_nvshmem_enable_tracing_d, &_nvshmem_enable_tracing, sizeof(int)));

    if (_nvshmem_enable_tracing) { 
        log_data_size = LOG_BUFFER_SIZE;
        value = getenv("NVSHMEM_LOG_BUFFER_SIZE");
        if (value != NULL) {
            log_data_size = atoi(value);
        }
 
        log_buffer_count = LOG_BUFFER_COUNT;
        value = getenv("NVSHMEM_LOG_BUFFER_COUNT");
        if (value != NULL) {
            log_buffer_count = atoi(value);
        }
 
        value = getenv("NVSHMEM_LOG_ACTIVE_GRIDS");
        if (value != NULL) {
            active_grids = atoi(value);
        }
 
        value = getenv("NVSHMEM_LOG_ACTIVE_BLOCKS");
        if (value != NULL) {
            active_blocks = atoi(value);
        }
 
        /*allocate host and device log regions*/
        CUDA_CHECK(cudaMallocHost((void **)&log_full, log_buffer_count*sizeof(log_buffer_t)));
        CUDA_CHECK(cudaMallocHost((void **)&log_free, log_buffer_count*sizeof(log_buffer_t)));
        memset(log_full, 0, log_buffer_count*sizeof(log_buffer_t));
        memset(log_free, 0, log_buffer_count*sizeof(log_buffer_t));
        CUDA_CHECK(cudaHostGetDevicePointer((void **) &log_full_devptr, (void *)log_full, 0));
        CUDA_CHECK(cudaHostGetDevicePointer((void **) &log_free_devptr, (void *)log_free, 0));

#if defined (_BLOCKLEVEL_TRACING_)
        int active_buffer_size = active_grids*active_blocks*sizeof(log_data_t *);
#else
        int active_buffer_size = active_grids*active_blocks*threads_per_block*sizeof(log_data_t *);
#endif

        CUDA_CHECK(cudaMalloc(&log_active, active_buffer_size));
        CUDA_CHECK(cudaMemset(log_active, 0, active_buffer_size));
 
        CUDA_CHECK(cudaMalloc(&log_data, log_buffer_count*(sizeof(log_data_t) + log_data_size)));
        CUDA_CHECK(cudaMemset(log_data, 0, log_buffer_count*(sizeof(log_data_t) + log_data_size)));
 
        log_storage_bytes = (log_buffer_count*(sizeof(log_data_t) + log_data_size)*log_storage_size);
        CUDA_CHECK(cudaMallocHost(&log_data_host, log_storage_bytes));
 
        /*allocate the ring_counters on the host and get the device pointers for mapped memory*/
        CUDA_CHECK(cudaMallocHost((void **)&ring_counters, 6*COUNTER_ALIGN_SIZE));
        memset(ring_counters, 0, 6*COUNTER_ALIGN_SIZE);
        CUDA_CHECK(cudaHostGetDevicePointer((void **) &ring_counters_devptr, ring_counters, 0));
 
        /*retrieve from head, add to tail*/
        full_head = (volatile unsigned int *) ring_counters;
        full_tail = (volatile unsigned int *) ((char *)ring_counters + 1*COUNTER_ALIGN_SIZE); 
        free_head = (volatile unsigned int *) ((char *)ring_counters + 2*COUNTER_ALIGN_SIZE);
        free_tail = (volatile unsigned int *) ((char *)ring_counters + 3*COUNTER_ALIGN_SIZE);
        overflow_flag = (volatile unsigned int *) ((char *)ring_counters + 4*COUNTER_ALIGN_SIZE);
        trace_call_conflict = (volatile unsigned int *) ((char *)ring_counters + 5*COUNTER_ALIGN_SIZE);
 
        /*free_head points to the next buffer to be freed 
          free_tail points to the next free buffer in the list 
          full_head points to the next buffer to be filled 
          full_tail points to the next full buffer in the list*/
        *overflow_flag = 0;
        *trace_call_conflict = 0;
        *full_head = 0;
        *full_tail = 0;
        *free_head = 0; //log_buffer_count - 1;
        *free_tail = 0;
 
        /*populate the free buffer region*/
        for (i=0; i<log_buffer_count; i++) {
            log_data_t *data = NULL; 
 
            data = (log_data_t *)((char *)log_data + i*(sizeof(log_data_t) + log_data_size));
            (log_free + i)->data = data; 
            (log_free + i)->status = 1;
            (log_full + i)->data = data; 
            (log_full + i)->status = 0;
        }
 
        unsigned int *temp_ptr;
        int temp_int;
 
        temp_ptr = ring_counters_devptr;
        CUDA_CHECK(cudaMemcpyToSymbol(full_head_d, &temp_ptr, sizeof(unsigned int *)));
        temp_ptr = (unsigned int *) ((char *)ring_counters_devptr + COUNTER_ALIGN_SIZE);
        CUDA_CHECK(cudaMemcpyToSymbol(full_tail_d, &temp_ptr, sizeof(unsigned int *)));
        temp_ptr = (unsigned int *) ((char *)ring_counters_devptr + 2*COUNTER_ALIGN_SIZE);
        CUDA_CHECK(cudaMemcpyToSymbol(free_head_d, &temp_ptr, sizeof(unsigned int *)));
        temp_ptr = (unsigned int *) ((char *)ring_counters_devptr + 3*COUNTER_ALIGN_SIZE);
        CUDA_CHECK(cudaMemcpyToSymbol(free_tail_d, &temp_ptr, sizeof(unsigned int *)));
        temp_ptr = (unsigned int *) ((char *)ring_counters_devptr + 4*COUNTER_ALIGN_SIZE);
        CUDA_CHECK(cudaMemcpyToSymbol(overflow_flag_d, &temp_ptr, sizeof(unsigned  int *)));
        temp_ptr = (unsigned int *) ((char *)ring_counters_devptr + 5*COUNTER_ALIGN_SIZE);
        CUDA_CHECK(cudaMemcpyToSymbol(trace_call_conflict_d, &temp_ptr, sizeof(unsigned int *)));
 
        CUDA_CHECK(cudaMemcpyToSymbol(log_buffer_count_d, &log_buffer_count, sizeof(int)));
 
        /*to simply check before inserting an entry*/
        temp_int = log_data_size/sizeof(log_record_t);
        CUDA_CHECK(cudaMemcpyToSymbol(log_record_limit_d, &temp_int, sizeof(int)));
        CUDA_CHECK(cudaMemcpyToSymbol(log_full_d, &log_full_devptr, sizeof(log_buffer_t *)));
        CUDA_CHECK(cudaMemcpyToSymbol(log_free_d, &log_free_devptr, sizeof(log_buffer_t *)));
        CUDA_CHECK(cudaMemcpyToSymbol(log_active_d, &log_active, sizeof(log_buffer_t **)));
 
        CUDA_CHECK(cudaMemcpyToSymbol(ngrids_d, &active_grids, sizeof(int)));
        temp_int = intlog2(active_grids);
        CUDA_CHECK(cudaMemcpyToSymbol(lngrids_d, &temp_int, sizeof(int)));
        CUDA_CHECK(cudaMemcpyToSymbol(nblocks_d, &active_blocks, sizeof(int)));
        temp_int = intlog2(active_blocks);
        CUDA_CHECK(cudaMemcpyToSymbol(lnblocks_d, &temp_int, sizeof(int)));
        temp_int = intlog2(threads_per_block);
        CUDA_CHECK(cudaMemcpyToSymbol(lnthreads_d, &temp_int, sizeof(int)));
 
        /*launch the logging thread*/
        pthread_attr_t attr;
 
        pthread_attr_init(&attr);
        pthread_create(&log_thread, &attr,
                              &log_backup, NULL);
        pthread_attr_destroy(&attr);
 
        /*explicitly relax affinity of the log thread to not conflict with the main thread (MPI library might have set its affinity*/
        int num_cores, num_cores_set, status;
        cpu_set_t cpuset;
 
        CPU_ZERO(&cpuset);
        num_cores = sysconf(_SC_NPROCESSORS_ONLN);
        for (i=0; i<num_cores; i++) {
            CPU_SET(i, &cpuset);
        }
 
        status = pthread_setaffinity_np(log_thread, sizeof(cpu_set_t), &cpuset);
        if (status != 0) {
            ERROR_PRINT("error setting affinity of log thread \n");
        }
 
        status = pthread_getaffinity_np(log_thread, sizeof(cpu_set_t), &cpuset);
        if (status != 0) {
            ERROR_PRINT("error getting affinity of log thread \n");
        }
 
        num_cores_set = 0;
        for (i = 0; i < num_cores; i++) {
            if (CPU_ISSET(i, &cpuset))
                num_cores_set++;
        }
        if (num_cores_set != num_cores) {
            ERROR_PRINT("the affinity of progress thread restricted to %d cores \n", num_cores_set);
        }
 
    }
#endif

    CUDA_CHECK(cudaDeviceSynchronize());
}

void *nvshmalloc (size_t size) 
{
    void *buf = NULL; 
    shmem_region_t *region_entry;

    if (((uint64_t)_nvshmem_heap_tail + size) > ((uint64_t)_nvshmem_heap + _nvshmem_heap_size)) { 
	ERROR_PRINT("Not enough memory on the heap (set to %d, can be increased by setting NVSHMEM_HEAP_SIZE), exiting . . .  \n", _nvshmem_heap_size); 
        exit(-1);
    }

    buf = (void *)_nvshmem_heap_tail; 
    _nvshmem_heap_tail = (_nvshmem_heap_tail + size); 

    region_entry = (shmem_region_t *) malloc (sizeof(shmem_region_t)); 

    region_entry->buffer = buf; 
    region_entry->size = size;
    region_entry->next = NULL;

    if (_nvshmem_region_list_tail == NULL) { 
         _nvshmem_region_list = _nvshmem_region_list_tail = region_entry;
    } else {
	 _nvshmem_region_list_tail->next = region_entry;
         _nvshmem_region_list_tail = region_entry; 
    }

    return buf;
}

/*deallocates all regions and resets the heap*/
void nvshmcleanup () 
{
    shmem_region_t *region_entry, *tmp;
    region_entry = _nvshmem_region_list;

    /*free regions*/
    while (region_entry != NULL) { 
        tmp = region_entry;
	region_entry = region_entry->next;
	free((void *)tmp); 
    }

    /*reset heap pointers*/
    _nvshmem_heap_head = _nvshmem_heap_tail = _nvshmem_heap; 	
}

void nvstop_pes () 
{
    int i; 

    for (i = 0; i < _nvshmem_npes; i++) {
      if (i == _nvshmem_mype) {
          continue;
      } 
      DEBUG_PRINT("[%d] closing heap pointer for pe: %d is %p \n", _nvshmem_mype, i, _nvshmem_heap_pointers[i]);
      CU_CHECK(cuIpcCloseMemHandle(_nvshmem_heap_pointers[i]));
      CU_CHECK(cuIpcCloseMemHandle(_nvshmem_barrier_sync_addr[i]));
    }
	
    free(_nvshmem_barrier_sync_addr);

    nvshmem_barrier_all();
 
    CU_CHECK(cuMemFree(_nvshmem_heap));
    CU_CHECK(cuMemFree(_nvshmem_barrier_sync));

#if defined(_ENABLE_TRACING_)
    if (_nvshmem_enable_tracing) {
        log_thread_exit_flag = 1; 
 
        pthread_join(log_thread, NULL);
 
        process_logs();
 
        CUDA_CHECK(cudaFreeHost (log_full));
        CUDA_CHECK(cudaFreeHost (log_free));
        CUDA_CHECK(cudaFreeHost (log_data_host));
        CUDA_CHECK(cudaFree (log_data));
        CUDA_CHECK(cudaFree (log_active));
    }
#endif 
}

void nvshmem_barrier_all () 
{
    CU_CHECK(cuCtxSynchronize());

#ifndef _USE_OPENSHMEM_
    MPI_Barrier(MPI_COMM_WORLD);
#else 
    shmem_barrier_all();
#endif 
}

void nvshmem_barrier_all_offload ()
{
   int **barrier_addr; 

   barrier_addr = (int **) (_nvshmem_barrier_sync_addr_d + (barrier_value%2)*_nvshmem_npes);
 
   launch_gpu_sync_wrapper(barrier_value, barrier_addr, _nvshmem_mype);

   barrier_value = (barrier_value + 1)%(64*1024);
}

void *nvshmem_ptr(void *local, int pe)
{
   return ((void *)(((char *)_nvshmem_heap_pointers[pe] + ((char *)local - (char *)_nvshmem_my_heap_pointer))));
}
