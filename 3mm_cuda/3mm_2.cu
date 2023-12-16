#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
extern "C" {
#include <polybench.h>
}

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "3mm.h"

/* Array initialization. */
static void init_array(int ni, int nj, int nk, int nl, int nm,
                       DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
                       DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
                       DATA_TYPE POLYBENCH_2D(C, NJ, NM, nj, nm),
                       DATA_TYPE POLYBENCH_2D(D, NM, NL, nm, nl))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = ((DATA_TYPE)i * j) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = ((DATA_TYPE)i * (j + 1)) / nj;
  for (i = 0; i < nj; i++)
    for (j = 0; j < nm; j++)
      C[i][j] = ((DATA_TYPE)i * (j + 3)) / nl;
  for (i = 0; i < nm; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = ((DATA_TYPE)i * (j + 2)) / nk;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nl,
                        DATA_TYPE POLYBENCH_2D(G, NI, NL, ni, nl))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, G[i][j]);
      if ((i * ni + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

/* Parameters */
#define BLOCK_SIZE 32
#define BLOCKS(n) (n + (BLOCK_SIZE)-1) / (BLOCK_SIZE)

/* GPU CODE*/

#define __devinline__ __device__ __forceinline__

// Intended for use as a zero-cost abstraction by the GPU
class GpuMatrix {
  DATA_TYPE* ptr_;
  int pitch_;

public:
  __devinline__ GpuMatrix(DATA_TYPE* ptr, int pitch)
    : ptr_(ptr), pitch_(pitch) {}

  __devinline__ DATA_TYPE& operator() (int row, int col) const
  {
    return ptr()[row * pitch() + col];
  }
  __devinline__ DATA_TYPE* ptr() const { return ptr_; }
  __devinline__ int pitch() const { return pitch_; }

};

class GpuMatrixSpan {
  GpuMatrix mat_;
  int row_;
  int col_;

public:
  __devinline__ GpuMatrixSpan(GpuMatrix mat, int row, int col)
    : mat_(mat), row_(row), col_(col) { }

  __devinline__ DATA_TYPE& operator() (int row, int col) const
  {
    return mat_(row_ + row, col_ + col);
  }
};
#undef __devinline__

__global__ void gemm_gpu(DATA_TYPE* __restrict__ ans, DATA_TYPE* __restrict__ a, DATA_TYPE* __restrict__ b, int ni, int nj, int nk)
{
  // One thread computes a single element of ans
  // The coordinates of the element that this thread should compute
  int ans_row = threadIdx.y + blockIdx.y * blockDim.y;
  int ans_col = threadIdx.x + blockIdx.x * blockDim.x;
  if (ans_row >= ni || ans_col >= nj)
    return;

  // Abstractions over the whole matrix in global memory
  const GpuMatrix ANS_global(ans, nj);
  const GpuMatrix A_global(a, nk);
  const GpuMatrix B_global(b, nj);

  // Shared tile
  __shared__ DATA_TYPE A_shared_tile[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ DATA_TYPE B_shared_tile[BLOCK_SIZE][BLOCK_SIZE];

  // The value of the result element
  float sum = 0.0f;

  // For each tile
  for (int tile_start_k = 0; tile_start_k < nk; tile_start_k += BLOCK_SIZE) {
    // Abstractions over the tile in global memory
    const GpuMatrixSpan A_global_tile(A_global, blockIdx.y * blockDim.y, tile_start_k);
    const GpuMatrixSpan B_global_tile(B_global, tile_start_k, blockIdx.x * blockDim.x);

    // Wait until all block threads are ready to load data (i.e. they have finished the previous iteration)
    __syncthreads();

    // Each thread in the block should copy one element from global memory to shared memory
    A_shared_tile[threadIdx.y][threadIdx.x] = A_global_tile(threadIdx.y, threadIdx.x);
    B_shared_tile[threadIdx.y][threadIdx.x] = B_global_tile(threadIdx.y, threadIdx.x);

    // Wait until the whole tile has been loaded to shared memory
    __syncthreads();

    // Accumulate the partial sum
    for (int k = 0; k < BLOCK_SIZE; ++k)
      sum += A_shared_tile[threadIdx.y][k] * B_shared_tile[k][threadIdx.x];
  }

  // Finally, store the result
  ANS_global(ans_row, ans_col) = sum;
}



/* HOST CODE */

#include <cuda_runtime.h>
#define gpuErrchk(ans)                  \
{                                       \
  gpuAssert((ans), __FILE__, __LINE__); \
}
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s (%d) %s %d\n", cudaGetErrorString(code), code, file, line);
    if (abort)
      exit(code);
  }
}


#include <cassert>

class HostMatrix {
  void* hostPtr_;
  void* devPtr_;
  int rows_;
  int cols_;

public:
  // Use RAII to manage device memory (avoid repeating malloc and free)
  HostMatrix(void* hostPtr, int rows, int cols) 
      : hostPtr_(hostPtr), devPtr_(nullptr), rows_(rows), cols_(cols) {
    gpuErrchk(cudaMalloc(&devPtr_, size()));
    // gpuErrchk(cudaMemset(devPtr_, 0, size()));
  }
  ~HostMatrix() {
    cudaFree(devPtr_);
  }

  int rows() const { return rows_; }
  int cols() const { return cols_; }
  void* devPtr() const { return devPtr_; }
  void* hostPtr() const { return hostPtr_; }
  size_t size() const { return rows() * cols() * sizeof(DATA_TYPE); }

  void copy_h2d() const { gpuErrchk(cudaMemcpy(devPtr(), hostPtr(), size(), cudaMemcpyHostToDevice)); }
  void copy_d2h() const { gpuErrchk(cudaMemcpy(hostPtr(), devPtr(), size(), cudaMemcpyDeviceToHost)); }

  void gemm(const HostMatrix& a, const HostMatrix& b) {
    assert(rows() == a.rows());
    assert(cols() == b.cols());
    assert(a.cols() == b.rows());

    int ni = rows();
    int nj = cols();
    int nk = a.cols();

    dim3 gridSize(BLOCKS(nj), BLOCKS(ni));
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    gemm_gpu<<<gridSize, blockSize>>>((DATA_TYPE*)devPtr(), (DATA_TYPE*)a.devPtr(), (DATA_TYPE*)b.devPtr(), ni, nj, nk);
  }
};


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_3mm(int ni, int nj, int nk, int nl, int nm,
                       DATA_TYPE POLYBENCH_2D(E, NI, NJ, ni, nj),
                       DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
                       DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
                       DATA_TYPE POLYBENCH_2D(F, NJ, NL, nj, nl),
                       DATA_TYPE POLYBENCH_2D(C, NJ, NM, nj, nm),
                       DATA_TYPE POLYBENCH_2D(D, NM, NL, nm, nl),
                       DATA_TYPE POLYBENCH_2D(G, NI, NL, ni, nl))
{
  HostMatrix a(A, NI, NK), b(B, NK, NJ), c(C, NJ, NM), d(D, NM, NL), e(E, NI, NJ), f(F, NJ, NL), g(G, NI, NL);
  
  a.copy_h2d();
  b.copy_h2d();
  c.copy_h2d();
  d.copy_h2d();

  e.gemm(a, b); // E = A*B
  f.gemm(c, d); // F = C*D
  g.gemm(e, f); // G = E*F
  cudaDeviceSynchronize();

  e.copy_d2h();
  f.copy_d2h();
  g.copy_d2h();
}

int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;
  int nm = NM;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(E, DATA_TYPE, NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
  POLYBENCH_2D_ARRAY_DECL(F, DATA_TYPE, NJ, NL, nj, nl);
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NM, nj, nm);
  POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NM, NL, nm, nl);
  POLYBENCH_2D_ARRAY_DECL(G, DATA_TYPE, NI, NL, ni, nl);

  /* Initialize array(s). */
  init_array(ni, nj, nk, nl, nm,
             POLYBENCH_ARRAY(A),
             POLYBENCH_ARRAY(B),
             POLYBENCH_ARRAY(C),
             POLYBENCH_ARRAY(D));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_3mm(ni, nj, nk, nl, nm,
             POLYBENCH_ARRAY(E),
             POLYBENCH_ARRAY(A),
             POLYBENCH_ARRAY(B),
             POLYBENCH_ARRAY(F),
             POLYBENCH_ARRAY(C),
             POLYBENCH_ARRAY(D),
             POLYBENCH_ARRAY(G));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nl, POLYBENCH_ARRAY(G)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(E);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(F);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(D);
  POLYBENCH_FREE_ARRAY(G);

  return 0;
}
