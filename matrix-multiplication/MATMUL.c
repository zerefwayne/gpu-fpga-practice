#include "MATMUL.h"

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

void do_compute(const struct parameters *p);

void initMatrix(float **X, int *r_X, int *c_X)
{
    long size_X = *r_X * *c_X;

    *X = (float *)calloc(size_X, sizeof(float));
    if (*X == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        exit(-1);
    }
}

void readMatrix(FILE *fp, float **X, int *r_X, int *c_X)
{
    long size_X = *r_X * *c_X;

    float *ptr;
    ptr = *X;
    for (int i = 0; i < size_X; i++)
    {
        if (fscanf(fp, "%f", ptr) <= 0)
        {
            fprintf(stderr, "Error reading data!\n");
            exit(-2);
        }
        ptr++;
    }
}

// If you wish to print the entire matrix, keep head = -1
void printMatrix(float *X, int r_X, int c_X, int head)
{
    if (head == -1)
    {
        head = INT_MAX;
    }

    float *ptr = X;
    for (int i = 0; i < MIN(head, r_X); i++)
    {
        for (int j = 0; j < MIN(head, c_X); j++)
        {
            printf("%f ", *(ptr++));
        }
        printf("\n");
    }
    printf("\n");
}

void readInput(char *filename, float **M, float **N, float **P, int *r_M, int *c_M, int *r_N, int *c_N, int *r_P, int *c_P)
{
    printf("Reading file: %s\n\n", filename);

    // Open file in read-only mode
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "File open error: %s.\n", filename);
        exit(-2);
    }

    // Read matrix M and N dimensions
    if (fscanf(fp, "%d%d%d%d", r_M, c_M, r_N, c_N) < 4)
    {
        fprintf(stderr, "Error reading file: %s.\n", filename);
        exit(-2);
    }

    if (*c_M != *r_N)
    {
        fprintf(stderr, "Matrix dimensions are not compatible for multiplication, c_M (%d) != r_N (%d).\n", *c_M, *r_N);
        exit(-1);
    }

    // Assign matrix dimensions for P
    *r_P = *r_M;
    *c_P = *c_N;

    initMatrix(M, r_M, c_M);
    initMatrix(N, r_N, c_N);
    initMatrix(P, r_P, c_P);

    readMatrix(fp, M, r_M, c_M);
    readMatrix(fp, N, r_N, c_N);
}

void outputMatrix(char *filename, float *P, int r_P, int c_P)
{
    // Open file in write-only mode
    FILE *fp = fopen(filename, "w");
    if (fp == NULL)
    {
        fprintf(stderr, "File open error: %s.\n", filename);
        exit(-2);
    }

    // Write matrix P dimensions
    fprintf(fp, "%d %d\n", r_P, c_P);

    // Write matrix P data
    float *ptr = P;
    for (int i = 0; i < r_P; i++)
    {
        for (int j = 0; j < c_P; j++)
        {
            fprintf(fp, "%f ", *(ptr++));
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

void sanityCheck(struct parameters *p)
{
    printf("Matrix M: %dx%d, Matrix N: %dx%d, Result P: %dx%d\n\n", p->r_M, p->c_M, p->r_N, p->c_N, p->r_P, p->c_P);
    printf("Matrix M (head):\n");
    printMatrix(p->M, p->r_M, p->c_M, p->head);
    printf("Matrix N (head):\n");
    printMatrix(p->N, p->r_N, p->c_N, p->head);
}

int main(int argc, char **argv)
{
    // Verify arguments
    if (argc != 2)
    {
        fprintf(stderr, "EXECUTION ERROR MATMUL: Parameters are not correct.\n");
        fprintf(stderr, "./MATMUL <input_file>");
        fflush(stderr);
        exit(-1);
    }

    // Initialise matrix pointers
    int r_M = 0, c_M = 0, r_N = 0, c_N = 0, r_P = 0, c_P = 0;
    float *M, *N, *P;

    readInput(argv[1], &M, &N, &P, &r_M, &c_M, &r_N, &c_N, &r_P, &c_P);

    struct parameters p = {
        .M = M,
        .N = N,
        .P = P,
        .r_M = r_M,
        .c_M = c_M,
        .r_N = r_N,
        .c_N = c_N,
        .r_P = r_P,
        .c_P = c_P,
        .head = 5,
        .output_filename = "output/MATMUL.out"};

    // Sanity check: Print matrix dimensions and data
    // Note: This is a simple check to ensure the matrices are read correctly.
    sanityCheck(&p);

    printf("Starting computation!\n");

    // Entry point of computation
    do_compute(&p);

    // Print result to output file
    printf("Successfully completed!\n");

    outputMatrix(p.output_filename, p.P, p.r_P, p.c_P);
    printf("Output written to: %s\n", p.output_filename);

    // Free pointers
    free(M);
    free(N);
    free(P);

    return 0;
}