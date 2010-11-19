#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int n = 512;
float
    a,
    beta_old,
    beta = 0.0,
  *x, *dx, *y, *dy,
  sqydot,ydot,
    thr = 1e-5;

#define I (i+1)
#define J (j+1)
#define BLOCKSIZE 256

__global__ void cudaDy(float * dx, float * dy, int n)
{
	int i = blockIdx.x*BLOCKSIZE+threadIdx.x;
	float ytemp = 0.0;
	for (int j = 0; j<n; j++)
	{
		ytemp += dx[j]/(0.5*(I+J-1)*(I+J-2)+I);
	}
	dy[i]=ytemp;
}
__global__ void cudaDx(float* dx, float* dy, float sqydot)
{
        int i = blockIdx.x*BLOCKSIZE+threadIdx.x;
        dx[i] = dy[i] / sqydot;
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
printf("lillolab");
fflush(stdout);
    cudaMalloc(&dx, n*sizeof(float));
    cudaMalloc(&dy, n*sizeof(float));
printf("lolcat");
    cudaMemcpy(dx, x, n*sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridBlock (n/BLOCKSIZE);
    dim3 threadBlock (BLOCKSIZE);

    do
    {
	printf("daeeeeooo");

      cudaDy <<< gridBlock, threadBlock >>> (dx, dy, n);
      cudaMemcpy(y, dy, n*sizeof(float), cudaMemcpyDeviceToHost);

        
        /* beta = dot(y,x) */
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
	sqydot = sqrt(ydot);
      cudaDx <<< gridBlock, threadBlock >>> (dx, dy, sqydot);
      cudaMemcpy(x, dx, n*sizeof(float), cudaMemcpyDeviceToHost);

    } while ( 1 );
    printf ( "%e\n", beta );
    free ( x ), free ( y );
    cudaFree(dx), cudaFree(dy);
}
