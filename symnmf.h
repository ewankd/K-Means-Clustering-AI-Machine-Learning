#ifndef SYMNMF_H
#define SYMNMF_H

double** alloc_matrix(int rows, int cols);
void free_matrix(double** mat, int rows);

double** read_points_file(const char* filename, int* out_n, int* out_d);

double** calc_similarity(double** X, int n, int d);
double** calc_ddg(double** A, int n);
double** calc_norm(double** A, double** D, int n);

double** symnmf_optimize(double** W, double** Hinit, int n, int k, int max_iter, double eps);

void print_matrix(double** M, int rows, int cols);

#endif
