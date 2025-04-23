#include "MATMUL.h"

#include <stdio.h>
#include <stdlib.h>

extern void do_compute(struct parameters *p)
{
    int r_M = p->r_M;
    int c_M = p->c_M;
    int r_N = p->r_N;
    int c_N = p->c_N;

    int len = snprintf(NULL, 0, "output/MATMUL_%d_%d_%d_%d_seq.out", r_M, c_M, r_N, c_N);
    p->output_filename = (char *)malloc(len + 1);

    if (p->output_filename != NULL)
    {
        snprintf(p->output_filename, len + 1, "output/MATMUL_%d_%d_%d_%d_seq.out", r_M, c_M, r_N, c_N);
    }
    else
    {
        fprintf(stderr, "Error generating output file name!\n");
        fflush(stderr);
        exit(-1);
    }

    float *P = p->P;
    float *M = p->M;
    float *N = p->N;

    // Perform matrix multiplication
    for (int i = 0; i < r_M; i++)
    {
        for (int j = 0; j < c_N; j++)
        {
            P[i * c_M + j] = 0;

            for (int k = 0; k < c_M; k++)
            {
                P[i * c_M + j] += M[i * r_M + k] * N[k * c_N + j];
            }
        }
    }
}
