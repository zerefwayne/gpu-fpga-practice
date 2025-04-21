#include "MATMUL.h"

#include <stdio.h>

extern void do_compute(struct parameters *p)
{
    p->output_filename = "output/MATMUL_seq.out";

    float *P = p->P;
    float *M = p->M;
    float *N = p->N;

    int r_M = p->r_M;
    int c_M = p->c_M;
    int c_N = p->c_N;

    // Perform matrix multiplication
    for (int i = 0; i < r_M; i++)
    {
        for (int j = 0; j < c_N; j++)
        {
            P[i * c_N + j] = 0;
            for (int k = 0; k < c_M; k++)
            {
                P[i * c_N + j] += M[i * c_M + k] * N[k * c_N + j];
            }
        }
    }
}
