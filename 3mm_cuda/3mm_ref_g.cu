#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
extern "C" {
#include <polybench.h>
}

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#ifndef N
#define N (1 << 10)
#endif


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
  /* E := A*B */
  for (int i = 0; i < _PB_NI; i++)
  {
    for (int j = 0; j < _PB_NJ; j++)
    {
      E[i][j] = 0;
      for (int k = 0; k < _PB_NK; ++k)
      {
        E[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  /* F := C*D */
  for (int i = 0; i < _PB_NJ; i++)
  {
    for (int j = 0; j < _PB_NL; j++)
    {
      F[i][j] = 0;
      for (int k = 0; k < _PB_NM; ++k)
      {
        F[i][j] += C[i][k] * D[k][j];
      }
    }
  }
  /* G := E*F */
  for (int i = 0; i < _PB_NI; i++)
  {
    for (int j = 0; j < _PB_NL; j++)
    {
      G[i][j] = 0;
      for (int k = 0; k < _PB_NJ; ++k)
      {
        G[i][j] += E[i][k] * F[k][j];
      }
    }
  }
}

__global__ void mm3_kernel1 (int ni, int nj, int nk, int nl, int nm, 
				DATA_TYPE *A, 
				DATA_TYPE *B, 
				DATA_TYPE *E) {

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	DATA_TYPE sum = 0;
	
	if ((i < _PB_NI) && (j < _PB_NJ))
	{
		E[i * NJ + j] = 0;
		int k;
		for(k=0; k < _PB_NK; k++)
		{
			sum += A[i * NK + k] * B[k * NJ + j];
		}
		E[i * NJ + j] = sum;
	}
}

__global__ void mm3_kernel2 (int ni, int nj, int nk, int nl, int nm, 
				DATA_TYPE *C, 
				DATA_TYPE *D, 
				DATA_TYPE *F) {
	
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	DATA_TYPE sum = 0;
	
	if ((i < _PB_NJ) && (j < _PB_NL))
	{
		F[i * NL + j] = 0;
		int k;
		for(k=0; k < _PB_NM; k++)
		{
			sum += C[i * NM + k] * D[k * NL +j];
		}
		F[i * NL + j] = sum;
	}
}

__global__ void mm3_kernel3 (int ni, int nj, int nk, int nl, int nm, 
				DATA_TYPE *E, 
				DATA_TYPE *F, 
				DATA_TYPE *G) {
	
	// index is a unique identifier of each GPU thread
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	DATA_TYPE sum = 0;

	if ((i < _PB_NI) && (j < _PB_NL))
	{
		G[i * NL + j] = 0;
		int k;
		for(k=0; k < _PB_NJ; k++)
		{
			sum += E[i * NJ + k] * F[k * NL + j];
		}
		G[i * NL + j] = sum;
	}
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

  /* Run kernel. */
  polybench_start_instruments;
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


  /* Start timer. */
  polybench_start_instruments;


	DATA_TYPE *d_a, *d_b, *d_c, *d_d, *d_e, *d_f, *d_g;
		
	cudaMalloc((void **)&d_a, sizeof(DATA_TYPE) * ni * nk);
	cudaMalloc((void **)&d_b, sizeof(DATA_TYPE) * nk * nj);
	cudaMalloc((void **)&d_c, sizeof(DATA_TYPE) * nj * nm);
	cudaMalloc((void **)&d_d, sizeof(DATA_TYPE) * nm * nl);
	cudaMalloc((void **)&d_e, sizeof(DATA_TYPE) * ni * nj);
	cudaMalloc((void **)&d_f, sizeof(DATA_TYPE) * nj * nl);
	cudaMalloc((void **)&d_g, sizeof(DATA_TYPE) * ni * nl);

	cudaMemcpy(d_a, POLYBENCH_ARRAY(A), sizeof(DATA_TYPE) * ni * nk, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, POLYBENCH_ARRAY(B), sizeof(DATA_TYPE) * nk * nj, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, POLYBENCH_ARRAY(C), sizeof(DATA_TYPE) * nj * nm, cudaMemcpyHostToDevice);
	cudaMemcpy(d_d, POLYBENCH_ARRAY(D), sizeof(DATA_TYPE) * nm * nl, cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	//dim3 dimGrid((ni + (BLOCK_SIZE)-1) / (BLOCK_SIZE), (n + (BLOCK_SIZE)-1) / (BLOCK_SIZE));
	//dim3 dimGrid((ni + (dimBlock.x)-1) / (dimBlock.x), (nl + (dimBlock.y)-1) / (dimBlocl.y));
	//3dim grid

	dim3 dimGrid1((ni+BLOCK_SIZE-1)/BLOCK_SIZE,(nj+BLOCK_SIZE-1)/BLOCK_SIZE);
	dim3 dimGrid2((nj+BLOCK_SIZE-1)/BLOCK_SIZE,(nl+BLOCK_SIZE-1)/BLOCK_SIZE);
	dim3 dimGrid3((ni+BLOCK_SIZE-1)/BLOCK_SIZE,(nl+BLOCK_SIZE-1)/BLOCK_SIZE);
   
	
	mm3_kernel1<<<dimGrid1, dimBlock>>>(ni, nj, nk, nl, nm, d_a, d_b, d_e);
	mm3_kernel2<<<dimGrid2, dimBlock>>>(ni, nj, nk, nl, nm, d_c, d_d, d_f);
	mm3_kernel3<<<dimGrid3, dimBlock>>>(ni, nj, nk, nl, nm, d_e, d_f, d_g);
	
	// Synchronize to make sure all kernels are finished
    	cudaDeviceSynchronize();

	cudaMemcpy(POLYBENCH_ARRAY(E), d_e, sizeof(DATA_TYPE) * ni * nj, cudaMemcpyDeviceToHost);
	cudaMemcpy(POLYBENCH_ARRAY(F), d_f, sizeof(DATA_TYPE) * nj * nl, cudaMemcpyDeviceToHost);
	cudaMemcpy(POLYBENCH_ARRAY(G), d_g, sizeof(DATA_TYPE) * ni * nl, cudaMemcpyDeviceToHost);

	cudaFree(d_e);
	cudaFree(d_f);
	cudaFree(d_g);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_d);

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
