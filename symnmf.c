#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "symnmf.h"

#ifndef PYTHON_EXT
static void print_error(void){
    printf("An Error Has Occurred\n");
}
#endif

double** alloc_matrix(int rows, int cols){
    int i;
    double** m = (double**)malloc((size_t)rows * sizeof(double*));
    if(!m) return NULL;

    for(i=0;i<rows;i++){
        m[i] = (double*)calloc((size_t)cols, sizeof(double));
        if(!m[i]){
            while(i-- > 0) free(m[i]);
            free(m);
            return NULL;
        }
    }
    return m;
}

void free_matrix(double** mat, int rows){
    int i;
    if(!mat) return;
    for(i=0;i<rows;i++) free(mat[i]);
    free(mat);
}

static int count_cols(const char* line){
    int c = 1;
    const char* p = line;
    while(*p){
        if(*p == ',') c++;
        p++;
    }
    return c;
}

static int scan_shape(FILE* fp, int* out_n, int* out_d){
    char buf[8192];
    int n = 0, d = 0;

    while(fgets(buf, (int)sizeof(buf), fp)){
        /* If a line does not fit in the buffer, fgets will not include \n and we are mid-line. */
        if(strchr(buf, '\n') == NULL && !feof(fp)) return 0;
        if(n == 0) d = count_cols(buf);
        n++;
    }
    if(n <= 0 || d <= 0) return 0;
    *out_n = n;
    *out_d = d;
    return 1;
}

static int parse_row(char* buf, double* row, int d){
    int j = 0;
    char* token = strtok(buf, ",\n\r");

    while(token){
        char* endptr;
        double v;
        if(j >= d) return 0;
        v = strtod(token, &endptr);
        if(endptr == token) return 0; /* no conversion */
        while(*endptr == ' ' || *endptr == '\t') endptr++;
        if(*endptr != '\0') return 0;
        row[j] = v;
        j++;
        token = strtok(NULL, ",\n\r");
    }
    return j == d;
}

double** read_points_file(const char* filename, int* out_n, int* out_d){
    FILE* fp;
    double** X = NULL;
    int n, d, i;
    char buf[8192];

    fp = fopen(filename, "r");
    if(!fp) return NULL;

    if(!scan_shape(fp, &n, &d)){
        fclose(fp);
        return NULL;
    }
    rewind(fp);

    X = alloc_matrix(n, d);
    if(!X){ fclose(fp); return NULL; }

    for(i=0;i<n;i++){
        if(!fgets(buf, (int)sizeof(buf), fp)){
            free_matrix(X, n);
            fclose(fp);
            return NULL;
        }
        if(strchr(buf, '\n') == NULL && !feof(fp)){
            /* line too long -> reject */
            free_matrix(X, n);
            fclose(fp);
            return NULL;
        }
        if(!parse_row(buf, X[i], d)){
            free_matrix(X, n);
            fclose(fp);
            return NULL;
        }
    }
    fclose(fp);

    *out_n = n;
    *out_d = d;
    return X;
}

double** calc_similarity(double** X, int n, int d){
    int i, j, l;
    double** A = alloc_matrix(n, n);
    if(!A) return NULL;

    for(i=0;i<n;i++){
        A[i][i] = 0.0;
        for(j=i+1;j<n;j++){
            double dist = 0.0;

            for(l=0;l<d;l++){
                double diff = X[i][l] - X[j][l];
                dist += diff * diff;

                /* Guard against overflow in accumulation */
                if(dist > DBL_MAX){
                    dist = DBL_MAX * 2.0;
                    break;
                }
            }

            /* If dist is NaN/Inf/overflow, similarity underflows to 0 anyway */
            if(!(dist >= 0.0) || dist > DBL_MAX){
                A[i][j] = 0.0;
            }else{
                A[i][j] = exp(-dist / 2.0);
            }
            A[j][i] = A[i][j];
        }
    }
    return A;
}


double** calc_ddg(double** A, int n){
    int i, j;
    double** D = alloc_matrix(n, n);
    if(!D) return NULL;

    for(i=0;i<n;i++){
        double sum = 0.0;
        for(j=0;j<n;j++) sum += A[i][j];
        D[i][i] = sum;
    }
    return D;
}

double** calc_norm(double** A, double** D, int n){
    int i, j;
    double** W;
    double* inv_sqrt_d;

    W = alloc_matrix(n, n);
    if(!W) return NULL;

    inv_sqrt_d = (double*)malloc((size_t)n * sizeof(double));
    if(!inv_sqrt_d){ free_matrix(W, n); return NULL; }

    for(i=0;i<n;i++){
        double di = D[i][i];

        /* Numerical-stability guard: if degree is 0 (e.g., all affinities underflow),
           define D^{-1/2} as 0 on that entry to avoid 1/sqrt(0) -> Inf. */
        if(di <= 0.0 || !(di == di)){
            inv_sqrt_d[i] = 0.0;
        }else{
            inv_sqrt_d[i] = 1.0 / sqrt(di);
        }
    }

    for(i=0;i<n;i++){
        W[i][i] = 0.0;
        for(j=i+1;j<n;j++){
            W[i][j] = A[i][j] * inv_sqrt_d[i] * inv_sqrt_d[j];
            W[j][i] = W[i][j];
        }
    }

    free(inv_sqrt_d);
    return W;
}


static double** matmul(double** A, int rA, int cA, double** B, int rB, int cB){
    int i, j, k;
    double** C;

    if(cA != rB) return NULL;
    C = alloc_matrix(rA, cB);
    if(!C) return NULL;

    for(i=0;i<rA;i++){
        for(k=0;k<cA;k++){
            double aik = A[i][k];
            for(j=0;j<cB;j++) C[i][j] += aik * B[k][j];
        }
    }
    return C;
}

static double** compute_HHT(double** H, int n, int k){
    int i, j, l;
    double** HHT = alloc_matrix(n, n);
    if(!HHT) return NULL;

    for(i=0;i<n;i++){
        for(j=i;j<n;j++){
            double sum = 0.0;
            for(l=0;l<k;l++) sum += H[i][l] * H[j][l];
            HHT[i][j] = sum;
            HHT[j][i] = sum;
        }
    }
    return HHT;
}

static double** compute_HHTH(double** HHT, double** H, int n, int k){
    int i, j, t;
    double** out = alloc_matrix(n, k);
    if(!out) return NULL;

    for(i=0;i<n;i++){
        for(t=0;t<n;t++){
            double a = HHT[i][t];
            for(j=0;j<k;j++) out[i][j] += a * H[t][j];
        }
    }
    return out;
}

static double update_and_diff(double** Hcur, double** WH, double** HHTH,
                              double** Hnext, int n, int k){
    int i, j;
    const double beta = 0.5;
    double diff = 0.0;

    for(i=0;i<n;i++){
        for(j=0;j<k;j++){
            double denom = HHTH[i][j];
            double num = WH[i][j];
            double ratio;
            double mult;
            double val;
            double d;
            const double denom_eps = 1e-6;

            if(!(num == num) || !(denom == denom)){
                ratio = 0.0;
            }else{
                ratio = num / (denom + denom_eps);
            }

            mult = (1.0 - beta) + beta * ratio;
            val = Hcur[i][j] * mult;

            if(val < 0.0) val = 0.0;
            Hnext[i][j] = val;

            d = val - Hcur[i][j];
            diff += d * d;
        }
    }
    return diff;
}


double** symnmf_optimize(double** W, double** Hinit, int n, int k, int max_iter, double eps){
    int iter;
    double** Hcur = Hinit;
    double** Hnext = NULL;
    double** WH = NULL;
    double** HHT = NULL;
    double** HHTH = NULL;

    if(!W || !Hcur) return NULL;

    Hnext = alloc_matrix(n, k);
    if(!Hnext){ free_matrix(Hcur, n); return NULL; }

    for(iter=0; iter<max_iter; iter++){
        double diff;

        WH = matmul(W, n, n, Hcur, n, k);
        HHT = compute_HHT(Hcur, n, k);
        if(!WH || !HHT) goto fail;

        HHTH = compute_HHTH(HHT, Hcur, n, k);
        if(!HHTH) goto fail;

        diff = update_and_diff(Hcur, WH, HHTH, Hnext, n, k);

        free_matrix(WH, n); WH = NULL;
        free_matrix(HHT, n); HHT = NULL;
        free_matrix(HHTH, n); HHTH = NULL;

        if(diff < eps){
            free_matrix(Hcur, n);
            return Hnext;
        }

        /* swap Hcur and Hnext */
        {
            double** tmp = Hcur;
            Hcur = Hnext;
            Hnext = tmp;
        }
    }

    free_matrix(Hnext, n);
    return Hcur;

fail:
    if(WH) free_matrix(WH, n);
    if(HHT) free_matrix(HHT, n);
    if(HHTH) free_matrix(HHTH, n);
    free_matrix(Hcur, n);
    free_matrix(Hnext, n);
    return NULL;
}

void print_matrix(double** M, int rows, int cols){
    int i, j;
    for(i=0;i<rows;i++){
        for(j=0;j<cols;j++){
            if(j > 0) printf(",");
            printf("%.4f", M[i][j]);
        }
        printf("\n");
    }
}

#ifndef PYTHON_EXT
int main(int argc, char** argv){
    const char* goal;
    const char* filename;
    int n = 0, d = 0;
    int had_error = 0;
    double** X = NULL;
    double** A = NULL;
    double** D = NULL;
    double** W = NULL;

    if(argc != 3){
        print_error();
        return 0;
    }
    goal = argv[1];
    filename = argv[2];

    X = read_points_file(filename, &n, &d);
    if(!X){
        print_error();
        return 0;
    }

    if(strcmp(goal, "sym") == 0){
        A = calc_similarity(X, n, d);
        if(!A){ had_error = 1; goto cleanup; }
        print_matrix(A, n, n);
    }else if(strcmp(goal, "ddg") == 0){
        A = calc_similarity(X, n, d);
        if(!A){ had_error = 1; goto cleanup; }
        D = calc_ddg(A, n);
        if(!D){ had_error = 1; goto cleanup; }
        print_matrix(D, n, n);
    }else if(strcmp(goal, "norm") == 0){
        A = calc_similarity(X, n, d);
        if(!A){ had_error = 1; goto cleanup; }
        D = calc_ddg(A, n);
        if(!D){ had_error = 1; goto cleanup; }
        W = calc_norm(A, D, n);
        if(!W){ had_error = 1; goto cleanup; }
        print_matrix(W, n, n);
    }else{
        had_error = 1;
        goto cleanup;
    }

cleanup:
    if(W) free_matrix(W, n);
    if(D) free_matrix(D, n);
    if(A) free_matrix(A, n);
    if(X) free_matrix(X, n);
    if(had_error) print_error();
    return 0;
}
#endif
