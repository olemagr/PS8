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

#define I (i+1)
#define J (j+1)

int
main ( int argc, char **argv )
{
    if ( argc > 1 )
        n = (1 << strtol ( argv[1], NULL, 10 ));
    x = (float *) malloc ( n*sizeof(float) );
    y = (float *) malloc ( n*sizeof(float) );
    memset ( x, 0, n*sizeof(float) );
    x[0] = 1.0;

    do
    {
        // y = A * x_old
        for ( int i=0; i<n; i++ )
        {
            y[i] = 0.0;
            for ( int j=0; j<n; j++ )
            {
                a = 1.0 / (0.5*(I+J-1)*(I+J-2)+I);
                y[i] += a * x[j];
            }
        }
        if ( fabs(beta_old-beta) < thr )
            break;
        
        /* beta = dot(y,x) */
        beta_old = beta;
        beta = 0.0;
        for ( int j=0; j<n; j++ ) beta += y[j] * x[j];

        ydot = 0.0;
        for ( int j=0; j<n; j++ )
            ydot += y[j] * y[j];
        for ( int j=0; j<n; j++ )
            x[j] = y[j] / sqrt(ydot);
    } while ( 1 );
    printf ( "%e\n", beta );
    free ( x ), free ( y );
}
