#ifndef MATMUL_H
#define MATMUL_H

#ifdef __cplusplus
extern "C" {
#endif

struct parameters {
    float *M;
    float *N;
    float *P;
    int r_M;
    int c_M;
    int r_N;
    int c_N;
    int r_P;
    int c_P;
    int head;
    char *output_filename;
    int blockSize;
};

struct gpu_parameters {
    int block_size;
};

void do_compute(struct parameters *p);
int do_validate(const struct parameters *p);

#ifdef __cplusplus
}
#endif

#endif
