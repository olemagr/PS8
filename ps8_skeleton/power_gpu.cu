#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int 
n = 512;
float
a, beta_old, beta = 0.0,
  sqydot, ydot, thr = 1e-5,
  *x, *dx, *y, *dy;

#define I (i+1)
#define J (j+1)
#define BLOCKSIZE 256 // Recommended blocksize from cuda ducumentation.

// y = A * x_old on GPU
__global__ void 
cudaDy(float * dx, float * dy, int n)
{
  int i = blockIdx.x*BLOCKSIZE+threadIdx.x;
  float ytemp = 0.0;
  for (int j = 0; j<n; j++)
    {
      ytemp += dx[j]/(0.5*(I+J-1)*(I+J-2)+I);
    }
  dy[i]=ytemp;
}

// x = y / sqrt(y dot y) on GPU. 2-norm precalculated on CPU.
__global__ void 
cudaDx(float* dx, float* dy, float sqydot)
{
  int i = blockIdx.x*BLOCKSIZE+threadIdx.x;
  dx[i] = dy[i]/sqydot;
}

int
main ( int argc, char **argv )
{
  if ( argc > 1 )
    n = (1 << strtol ( argv[1], NULL, 10 ));
  x = (float *) malloc ( n*sizeof(float) );
  y = (float *) malloc ( n*sizeof(float) );
  memset ( x, 0, n*sizeof(float) );
  x[0] = 1.0;
  // Allocate similar arrays on device.
  cudaMalloc(&dx, n*sizeof(float));
  cudaMalloc(&dy, n*sizeof(float));
  // Copy initial contents of x one time.
  cudaMemcpy(dx, x, n*sizeof(float), cudaMemcpyHostToDevice);
  // Set size of thread block
  dim3 threadBlock (BLOCKSIZE);
  // Set number of thread blocks
  dim3 gridBlock (n/BLOCKSIZE);
  
  do
    {
      // Calculate y vector
      cudaDy <<< gridBlock, threadBlock >>> (dx, dy, n);
      // Copy result to host
      cudaMemcpy(y, dy, n*sizeof(float), cudaMemcpyDeviceToHost);
      
      // Calculate new beta and y dot product on host.
      beta_old = beta;
      beta = 0.0;
      ydot = 0.0;
      for ( int j=0; j<n; j++ ) 
	{
	  beta += y[j] * x[j];
	  ydot += y[j] * y[j];
	  }
      if ( fabs(beta_old-beta) < thr )
	break;
      // Precalculate square root on host and send to x vector calculation.
      sqydot = sqrt(ydot);
      cudaDx <<< gridBlock, threadBlock >>> (dx, dy, sqydot);
      // Copy result to host.
      cudaMemcpy(x, dx, n*sizeof(float), cudaMemcpyDeviceToHost);
      
    } while ( 1 );
  printf ( "%e\n", beta );
  free ( x ), free ( y );
  cudaFree(dx), cudaFree(dy);
}
