#include "MATMUL.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to calculate elapsed time in milliseconds
static inline double elapsedMS(const struct timespec *start, const struct timespec *end)
{
    return (end->tv_sec - start->tv_sec) * 1e3 + (end->tv_nsec - start->tv_nsec) / 1e6;
}

void initMatrix(float **X, int r_X, int c_X)
{
    long size_X = r_X * c_X;

    *X = (float *)calloc(size_X, sizeof(float));
    if (*X == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        exit(-1);
    }
}

void fillMatrix(float *X, int r_X, int c_X)
{
    long int size_X = r_X * c_X;

    for (int i = 0; i < size_X; i++)
    {
        X[i] = (float)rand() / RAND_MAX;
    }
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <N_BENCHMARKS> <N_ITERATIONS>\n", argv[0]);
        return -1;
    }

    int N_BENCHMARKS = atoi(argv[1]);
    int N_ITERATIONS = atoi(argv[2]);

    if (N_BENCHMARKS <= 0 || N_ITERATIONS <= 0)
    {
        fprintf(stderr, "Both N_BENCHMARKS and N_ITERATIONS must be positive integers.\n");
        return -1;
    }

    printf("N_BENCHMARKS: %d\n", N_BENCHMARKS);
    printf("N_ITERATIONS: %d\n", N_ITERATIONS);
    printf("\n----------------------------------\n\n");
    fflush(stdout);

    // Array of N (matrix size)
    long int *benchmarks = (long int *)malloc(N_BENCHMARKS * sizeof(long int));

    // Generate benchmarks
    for (int i = 0; i < N_BENCHMARKS; i++)
    {
        benchmarks[i] = 1 << (i + 1); // Starts from 2x2
    }

    double computeTime;
    double validationTime;
    long int matrix_size;
    // [...(stage_1_time_it_1, stage_2_time_it_1) * N_ITERATIONS]
    float *timings = (float *)calloc(N_ITERATIONS * 2, sizeof(float));
    // Initialise timing
    struct timespec doComputeStart, doComputeEnd;
    struct timespec doValidateStart, doValidateEnd;
    double avgComputeTime;
    double avgValidationTime;

    // Loop over all benchmark cases
    for (int i = 0; i < N_BENCHMARKS; i++)
    {
        matrix_size = benchmarks[i];

        memset(timings, 0.0, N_ITERATIONS * 2 * sizeof(float));

        printf("Benchmark %d: %ldx%ld\n\n", i + 1, matrix_size, matrix_size);

        float *M, *N, *P;

        int r_M = matrix_size, c_M = matrix_size, r_N = matrix_size, c_N = matrix_size, r_P = matrix_size, c_P = matrix_size;

        initMatrix(&M, r_M, c_M);
        initMatrix(&N, r_N, c_N);
        initMatrix(&P, r_P, c_P);

        fillMatrix(M, r_M, c_M);
        fillMatrix(N, r_N, c_N);

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
        };

        for (int it = 0; it < N_ITERATIONS; it++)
        {
            clock_gettime(CLOCK_MONOTONIC, &doComputeStart);

            // Entry point of computation
            do_compute(&p);

            clock_gettime(CLOCK_MONOTONIC, &doComputeEnd);

            computeTime = elapsedMS(&doComputeStart, &doComputeEnd);
            timings[it * 2 + 0] = computeTime;

            clock_gettime(CLOCK_MONOTONIC, &doValidateStart);

            if (do_validate(&p) == 0)
            {
                printf("Iteration %d\n", it);
                printf("Stage 2: End validation\n");
                fflush(stdout);
                fprintf(stderr, "Stage 2: Incorrect output. Exiting.");
                fflush(stderr);
                exit(-1);
            }

            clock_gettime(CLOCK_MONOTONIC, &doValidateEnd);
            validationTime = elapsedMS(&doValidateStart, &doValidateEnd);
            timings[it * 2 + 1] = validationTime;
        }

        avgComputeTime = 0.0;
        avgValidationTime = 0.0;

        for (int it = 0; it < N_ITERATIONS; it++)
        {
            avgComputeTime += timings[it * 2 + 0];
            avgValidationTime += timings[it * 2 + 1];
        }

        avgComputeTime /= N_ITERATIONS;
        avgValidationTime /= N_ITERATIONS;

        printf("Compute Time: %.3f ms\n", avgComputeTime);
        printf("Validation Time: %.3f ms\n", avgValidationTime);

        printf("\n----------------------------------\n\n");
        fflush(stdout);

        free(M);
        free(N);
        free(P);
    }

    printf("Benchmarking complete!\n");

    free(timings);
    free(benchmarks);

    return 0;
}
