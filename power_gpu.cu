#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


int n = 512;
float
    a,
    beta_old = 1.0,
    beta = 0.0,
    *x, *y,
    ydot,
    thr = 1e-5;

float
    * dx, * dy;

#define I (i+1)
#define J (j+1)

#define BLOCKSIZE 256
/* Notice that for n < BLOCKSIZE the program will fail
 *
 * Can be solved by choosing bigger n
 * or by having BLOCKSIZE as variable and add a test
 */

__global__ void cu_mult (float * dx, float * dy, int n)
{
        int i = blockIdx.x * BLOCKSIZE + threadIdx.x;
        dy[i] = 0;
	for (int j = 0; j < n; ++j)
	{
                float a = 1.0 / (0.5*(I+J-1)*(I+J-2)+I);
                dy[i] += a * dx[j];
	}
}

__global__ void cu_divide_ydot (float * dx, float * dy, float ydot)
{
        int i = blockIdx.x * BLOCKSIZE + threadIdx.x;
        dx[i] = dy[i] / sqrt(ydot);
}

int
main ( int argc, char **argv )
{
    // Setting up n
    if ( argc > 1 )
        n = (1 << strtol ( argv[1], NULL, 10 ));
	

    // Allocating space on CPU
    x = (float *) malloc ( n*sizeof(float) );
    y = (float *) malloc ( n*sizeof(float) );

    // Allocating space on GPU
    cudaMalloc ( (void**) &dx,    n*sizeof(float) );
    cudaMalloc ( (void**) &dy,    n*sizeof(float) );
    cudaMemset ( dy , 0, n*sizeof(float));	

    // Setting up initial values
    memset ( x, 0, n*sizeof(float) );
    x[0] = 1.0;
    cudaMemcpy ( dx, x, n*sizeof(float), cudaMemcpyHostToDevice );

    // Setting up dimentions on GPU
    dim3 gridBlock ( n/BLOCKSIZE );
    dim3 threadBlock ( BLOCKSIZE );

	do
	{
    	  cu_mult <<< gridBlock, threadBlock >>> ( dx, dy, n );
	  cudaMemcpy ( y , dy, n*sizeof(float), cudaMemcpyDeviceToHost );

        if ( fabs(beta_old-beta) < thr )
            break;

        cudaMemcpy ( x , dx, n*sizeof(float), cudaMemcpyDeviceToHost );
        
        beta_old = beta;
        beta = 0.0;
        ydot = 0.0;
        for ( int j=0; j<n; j++ )
	{
            beta += y[j] * x[j];
            ydot += y[j] * y[j];
	}

        cu_divide_ydot <<< gridBlock, threadBlock >>> ( dx, dy, ydot );
    }
    while ( 1 );

    printf ( "%e\n", beta );
    free ( x ), free ( y );
    cudaFree ( &dx ), cudaFree ( &dy );
}
