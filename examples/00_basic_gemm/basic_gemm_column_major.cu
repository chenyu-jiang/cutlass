/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*
  This example demonstrates how to call a CUTLASS GEMM kernel and provides a naive reference
  matrix multiply kernel to verify its correctness.

  The CUTLASS Gemm template is instantiated in the function CutlassSgemmNN. This is kernel computes
  the general matrix product (GEMM) using single-precision floating-point arithmetic and assumes
  all matrices have column-major layout.

  The threadblock tile size is chosen as 128x128x8 which offers good performance for large matrices.
  See the CUTLASS Parallel for All blog post for more exposition on the tunable parameters available
  in CUTLASS.

  https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

  Aside from defining and launching the SGEMM kernel, this example does not use any other components
  or utilities within CUTLASS. Such utilities are demonstrated elsewhere in other examples and are
  prevalent in the CUTLASS unit tests.

  This example has delibrately been kept similar to the basic_gemm example from cutass-1.3 to 
  highlight the minimum amount of differences needed to transition to cutlass-2.0.

  Cutlass-1.3 sgemm: https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu
*/

// Standard Library includes
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <chrono>

#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <cmath>
#include <cstdlib>

#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"

// Helper methods to check for errors
#include "helper.h"

//
// CUTLASS includes needed for single-precision GEMM kernel
//

// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/device/gemm.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

#define THREAD_BLOCK_SIZE 32
#define BLOCK_REPEAT 4
#define BLOCK_DIM ((THREAD_BLOCK_SIZE)*(BLOCK_REPEAT))

#define COMM_UNIT_LANES 4

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

// Matrix serialization for debugging purpose
std::string SerializeMatrixData(float* data, int size) {
  std::string str;
  for (int i=0;i<size;i++) {
    str += std::to_string(*(data + i));
    str += "\n";
  }
  return str;
}

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc,
  volatile int* progress) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                  ColumnMajor,  // Layout of A matrix
                                                  float,        // Data-type of B matrix
                                                  ColumnMajor,  // Layout of B matrix
                                                  float,        // Data-type of C matrix
                                                  ColumnMajor>; // Layout of C matrix

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //
  CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  //
  // Launch the CUTLASS GEMM kernel.
  //
  
  cutlass::Status status = gemm_operator(args, progress);

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// The source code after this point in the file is generic CUDA using the CUDA Runtime API
// and simple CUDA kernels to initialize matrices and compute the general matrix product.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a matrix with small integers.
__global__ void InitializeMatrix_kernel(
  float *matrix,
  int ldm,
  int rows,
  int columns,
  int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * ldm;

    // Generate arbitrary elements.
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);

    matrix[offset] = value;
  }
}

/// Simple function to initialize a matrix to arbitrary small integers.
cudaError_t InitializeMatrix(float *matrix, int ldm, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<<< grid, block >>>(matrix, ldm, rows, columns, seed);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device memory for a matrix then fills with arbitrary small integers.
cudaError_t AllocateMatrix(float **matrix, int ldm, int rows, int columns, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(float) * ldm * columns;

  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Initialize matrix elements to arbitrary small integers.
  result = InitializeMatrix(*matrix, ldm, rows, columns, seed);

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference GEMM computation.
__global__ void ReferenceGemm_kernel(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  for (int bi=0; bi < BLOCK_REPEAT; bi ++) {
    for (int bj=0; bj < BLOCK_REPEAT; bj++) {
      int i = threadIdx.x + blockIdx.x * BLOCK_DIM + bi * THREAD_BLOCK_SIZE;
      int j = threadIdx.y + blockIdx.y * BLOCK_DIM + bj * THREAD_BLOCK_SIZE;
      if (i < M && j < N) {
        float accumulator = 0;

        for (int k = 0; k < K; ++k) {
          accumulator += A[i + k * lda] * B[k + j * ldb];
        }

        C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
      }
    }
  }
}

__global__ void ReferenceGemm_kernel_with_progress(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc,
  volatile int* progress,
  int blocks_per_dim) {

  for (int bi=0; bi < BLOCK_REPEAT; bi ++) {
    for (int bj=0; bj< BLOCK_REPEAT; bj++) {
      int i = threadIdx.x + blockIdx.x * BLOCK_DIM + bi * THREAD_BLOCK_SIZE;
      int j = threadIdx.y + blockIdx.y * BLOCK_DIM + bj * THREAD_BLOCK_SIZE;
      if (i < M && j < N) {
        float accumulator = 0;

        for (int k = 0; k < K; ++k) {
          accumulator += A[i + k * lda] * B[k + j * ldb];
        }

        C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
      }
    }
  }

  __syncthreads();
  __threadfence_system();
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    *(progress + blockIdx.y * blocks_per_dim + blockIdx.x) = 1;
  }
  __threadfence_system();
}

/// Reference GEMM computation.
cudaError_t ReferenceGemm(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  dim3 block(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
  dim3 grid(
    (M + BLOCK_DIM - 1) / BLOCK_DIM,
    (N + BLOCK_DIM - 1) / BLOCK_DIM
  );

  ReferenceGemm_kernel<<< grid, block >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  return cudaGetLastError();
}

cudaError_t ReferenceGemm_with_progress(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc,
  volatile int* progress) {

  dim3 block(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
  dim3 grid(
    (M + BLOCK_DIM - 1) / BLOCK_DIM,
    (N + BLOCK_DIM - 1) / BLOCK_DIM
  );

  ReferenceGemm_kernel_with_progress<<< grid, block >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, progress, M / BLOCK_DIM);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
cudaError_t TestCutlassGemm(int M, int N, int K, float alpha, float beta, 
                            ncclComm_t& comm, cudaStream_t& s, int rank) {
  cudaError_t result;

  //
  // Define several matrices to be used as operands to GEMM kernels.
  //

  // Compute leading dimensions for each matrix.
  int lda = M;
  int ldb = K;
  int ldc = M;

  // Compute size in bytes of the C matrix.
  size_t sizeof_C = sizeof(float) * ldc * N;

  // Define pointers to matrices in GPU device memory.
  float *A;
  float *B;
  float *C_cutlass;
  float *C_reference;

  volatile int *device_progress, *host_progress;

  cudaSetDeviceFlags(cudaDeviceMapHost);
  cudaCheckErrors("cudaSetDeviceFlags error");
  cudaHostAlloc((void **)&host_progress, sizeof(int)*(M/BLOCK_DIM)*(N/BLOCK_DIM), cudaHostAllocMapped);
  cudaCheckErrors("cudaHostAlloc error");
  cudaHostGetDevicePointer((int **)&device_progress, (int *)host_progress, 0);
  cudaCheckErrors("cudaHostGetDevicePointer error");

  for(int i=0; i < (M/BLOCK_DIM)*(N/BLOCK_DIM); i++) {
    *(host_progress+i) = 0;
  }
  //
  // Allocate matrices in GPU device memory with arbitrary seeds.
  //

  result = AllocateMatrix(&A, lda, M, K, 0);

  if (result !=  cudaSuccess) {
    return result;
  }

  result = AllocateMatrix(&B, ldb, K, N, 17);

  if (result !=  cudaSuccess) {
    cudaFree(A);
    return result;
  }

  result = AllocateMatrix(&C_cutlass, ldc, M, N, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    return result;
  }

  result = AllocateMatrix(&C_reference, ldc, M, N, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    cudaFree(C_cutlass);
    return result;
  }

  result = cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  std::unordered_set<int> unfinished_block_ids;
  for(int i=0;i<(M/BLOCK_DIM)*(N/BLOCK_DIM);i++) {
    unfinished_block_ids.insert(i);
  }

  std::vector<std::unordered_set<int>> comm_lanes_tracker;

  for (int n=0; n<(N / BLOCK_DIM); n++) {
    std::unordered_set<int> comm_lane_blocks;
    for (int m=0; m < (M / BLOCK_DIM); m++) {
      int block_id = (M / BLOCK_DIM) * n + m;
      comm_lane_blocks.insert(block_id);
    }
    comm_lanes_tracker.emplace_back(std::move(comm_lane_blocks));
  }
  // serialize original arrays
  // std::ofstream Af;
  // std::ofstream Bf;
  
  // float* Af_host = (float*)malloc(M*K*sizeof(float));
  // cudaMemcpy(Af_host, A, M*K*sizeof(float), cudaMemcpyDeviceToHost);

  // Af.open(std::to_string(rank) + "_A.txt");
  // Af << SerializeMatrixData(Af_host, M*K);
  // Af.close();

  // free(Af_host);

  // float* Bf_host = (float*)malloc(N*K*sizeof(float));
  // cudaMemcpy(Bf_host, B, N*K*sizeof(float), cudaMemcpyDeviceToHost);

  // Bf.open(std::to_string(rank) + "_B.txt");
  // Bf << SerializeMatrixData(Bf_host, K*N);
  // Bf.close();

  // free(Bf_host);

  // std::cout << "Elements in set: ";
  // for (const auto& elem: unfinished_block_ids) {
    // std::cout << elem << ", ";
  // }
  // std::cout << std::endl;

  //
  // Launch CUTLASS GEMM.
  //
  std::cout << "Launching CUTLASS GEMM kernel." << std::endl;

  std::cout << "M: " << M << ", N: " << N << ", K: " << K << std::endl;

  result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc, device_progress);
  // for (int i=0; i<50; i++)
    // result = ReferenceGemm_with_progress(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc, device_progress);

  std::vector<std::pair<std::chrono::_V2::system_clock::time_point, const int>> timestamps;

  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  // spin check for progress
  int current_lane = 0;
  do{
    int idx_to_remove = -1;
    for (const auto& elem: unfinished_block_ids) {
      if (*(host_progress+elem) == 1) {
        idx_to_remove = elem;
        timestamps.push_back(std::make_pair(std::chrono::high_resolution_clock::now(), elem));
        // std::cout << "Block " << elem << " finished." << std::endl;
        comm_lanes_tracker[elem / (M/BLOCK_DIM)].erase(elem);
        while (comm_lanes_tracker[current_lane].empty() && current_lane < N/BLOCK_DIM) {
          current_lane ++;
          if (current_lane % COMM_UNIT_LANES == 0) {
            float* buffer = C_cutlass + (current_lane-COMM_UNIT_LANES) * M * BLOCK_DIM;
            // entire block lane finished, call NCCL
            std::cout << "Launching NCCL on offset " << current_lane * M * BLOCK_DIM << "." << std::endl;
            NCCLCHECK(ncclAllReduce((const void*)buffer, (void*)buffer, M * BLOCK_DIM * COMM_UNIT_LANES, ncclFloat, ncclSum,
            comm, s));
          }
        }
        break;
      }
    }
    if (idx_to_remove != -1) {
      unfinished_block_ids.erase(idx_to_remove);
    }
  }
  while (!unfinished_block_ids.empty());

  printf("CUTLASS kernel finished.\n");

  CUDACHECK(cudaDeviceSynchronize());

  std::cout << "Timestamps: " << std::endl;
  for (const auto &st: timestamps) {
    auto timestamp = st.first;
    int block_id = st.second;
    std::cout << timestamp.time_since_epoch().count() << ", Block " << block_id << std::endl;
  }
 
  std::cout << "Launching reference GEMM." << std::endl;

  //
  // Verify.
  //

  // Launch reference GEMM
  // for (int i=0;i<50;i++) 
  result = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

  if (result != cudaSuccess) {
    std::cerr << "Reference GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  CUDACHECK(cudaDeviceSynchronize());

  // we average reference GEMM as well
  NCCLCHECK(ncclAllReduce((const void*)C_reference, (void*)C_reference, M*N, ncclFloat, ncclSum,
  comm, s));

  CUDACHECK(cudaDeviceSynchronize());

  // Copy to host and verify equivalence.
  std::vector<float> host_cutlass(ldc * N, 0);
  std::vector<float> host_reference(ldc * N, 0);

  result = cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy CUTLASS GEMM results: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  result = cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy Reference GEMM results: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  CUDACHECK(cudaDeviceSynchronize());

  // std::ofstream C_cutlassf;
  // std::ofstream C_referencef;

  // C_cutlassf.open(std::to_string(rank) + "_C_cutlass.txt");
  // C_cutlassf << SerializeMatrixData(host_cutlass.data(), M*N);
  // C_cutlassf.close();

  // C_referencef.open(std::to_string(rank) + "_C_reference.txt");
  // C_referencef << SerializeMatrixData(host_reference.data(), M*N);
  // C_referencef.close();

  //
  // Free device memory allocations.
  //

  cudaFree(C_reference);
  cudaFree(C_cutlass);
  cudaFree(B);
  cudaFree(A);

  //
  // Test for bit equivalence of results.
  //
  double sum_of_difference = 0;
  for (int i=0;i<ldc * N; i++) {
    sum_of_difference += std::abs(host_cutlass[i] - host_reference[i]);
  }
  if (sum_of_difference > 1e-5) {
    std::cerr << "CUTLASS results incorrect. Sum of difference: " << sum_of_difference << std::endl;

    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to basic_gemm example.
//
// usage:
//
//   00_basic_gemm <M> <N> <K> <alpha> <beta>
//
int main(int argc, char *argv[]) {

  //
  // Parse the command line to obtain GEMM dimensions and scalar values.
  //

  // GEMM problem dimensions.
  int problem[3] = { 128, 128, 128 };

  for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(argv[i]);
    ss >> problem[i - 1];
  }

  // Scalars used for linear scaling the result of the matrix product.
  float scalars[2] = { 1, 0 };

  for (int i = 4; i < argc && i < 6; ++i) {
    std::stringstream ss(argv[i]);
    ss >> scalars[i - 4];
  }

  // initialize MPI and NCCL
  int myRank, nRanks, localRank = 0;

  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));


  //calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }

  ncclUniqueId id;
  ncclComm_t comm;
  cudaStream_t s;

  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));

  int priority;

  CUDACHECK(cudaDeviceGetStreamPriorityRange(NULL, &priority));

  CUDACHECK(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, priority));

  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  //
  // Run the CUTLASS GEMM test.
  //

  cudaError_t result = TestCutlassGemm(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    scalars[0],     // alpha
    scalars[1],      // beta
    comm,
    s,
    myRank
  );

  if (result == cudaSuccess) {
    std::cout << "Passed." << std::endl;
  }

  ncclCommDestroy(comm);
  MPICHECK(MPI_Finalize());

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
