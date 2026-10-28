#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include "symnmf.h"

static void set_error(void){
    PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
}

static double** pylist_to_matrix(PyObject* obj, int* out_r, int* out_c){
    Py_ssize_t r, c, i, j;
    double** M = NULL;

    if(!PyList_Check(obj)) return NULL;
    r = PyList_Size(obj);
    if(r <= 0) return NULL;

    if(!PyList_Check(PyList_GetItem(obj, 0))) return NULL;
    c = PyList_Size(PyList_GetItem(obj, 0));
    if(c <= 0) return NULL;

    M = alloc_matrix((int)r, (int)c);
    if(!M) return NULL;

    for(i=0;i<r;i++){
        PyObject* row = PyList_GetItem(obj, i);
        if(!PyList_Check(row) || PyList_Size(row) != c){
            free_matrix(M, (int)r);
            return NULL;
        }
        for(j=0;j<c;j++){
            PyObject* val = PyList_GetItem(row, j);
            double x = PyFloat_AsDouble(val);
            if(PyErr_Occurred()){
                free_matrix(M, (int)r);
                return NULL;
            }
            M[(int)i][(int)j] = x;
        }
    }

    *out_r = (int)r;
    *out_c = (int)c;
    return M;
}

static PyObject* matrix_to_pylist(double** M, int r, int c){
    int i, j;
    PyObject* outer = PyList_New(r);
    if(!outer) return NULL;

    for(i=0;i<r;i++){
        PyObject* row = PyList_New(c);
        if(!row){ Py_DECREF(outer); return NULL; }
        for(j=0;j<c;j++){
            PyObject* v = PyFloat_FromDouble(M[i][j]);
            if(!v){ Py_DECREF(row); Py_DECREF(outer); return NULL; }
            PyList_SetItem(row, j, v);
        }
        PyList_SetItem(outer, i, row);
    }
    return outer;
}

static double** compute_W_from_X(double** X, int n, int d){
    double** A = calc_similarity(X, n, d);
    double** D;
    double** W;

    if(!A) return NULL;
    D = calc_ddg(A, n);
    if(!D){ free_matrix(A, n); return NULL; }
    W = calc_norm(A, D, n);
    free_matrix(A, n);
    free_matrix(D, n);
    return W;
}

static PyObject* py_sym(PyObject* self, PyObject* args){
    PyObject* Xobj;
    double** X;
    double** A;
    int n, d;
    PyObject* out;

    (void)self;
    if(!PyArg_ParseTuple(args, "O", &Xobj)) return NULL;

    X = pylist_to_matrix(Xobj, &n, &d);
    if(!X){ set_error(); return NULL; }

    A = calc_similarity(X, n, d);
    free_matrix(X, n);
    if(!A){ set_error(); return NULL; }

    out = matrix_to_pylist(A, n, n);
    free_matrix(A, n);
    return out;
}

static PyObject* py_ddg(PyObject* self, PyObject* args){
    PyObject* Xobj;
    double** X;
    double** A;
    double** D;
    int n, d;
    PyObject* out;

    (void)self;
    if(!PyArg_ParseTuple(args, "O", &Xobj)) return NULL;

    X = pylist_to_matrix(Xobj, &n, &d);
    if(!X){ set_error(); return NULL; }

    A = calc_similarity(X, n, d);
    free_matrix(X, n);
    if(!A){ set_error(); return NULL; }

    D = calc_ddg(A, n);
    free_matrix(A, n);
    if(!D){ set_error(); return NULL; }

    out = matrix_to_pylist(D, n, n);
    free_matrix(D, n);
    return out;
}

static PyObject* py_norm(PyObject* self, PyObject* args){
    PyObject* Xobj;
    double** X;
    double** W;
    int n, d;
    PyObject* out;

    (void)self;
    if(!PyArg_ParseTuple(args, "O", &Xobj)) return NULL;

    X = pylist_to_matrix(Xobj, &n, &d);
    if(!X){ set_error(); return NULL; }

    W = compute_W_from_X(X, n, d);
    free_matrix(X, n);
    if(!W){ set_error(); return NULL; }

    out = matrix_to_pylist(W, n, n);
    free_matrix(W, n);
    return out;
}

static PyObject* py_symnmf(PyObject* self, PyObject* args){
    PyObject* Wobj;
    PyObject* Hobj;
    int max_iter;
    double eps;
    double** W;
    double** H;
    int nW, cW, nH, k;
    double** Hfinal;
    PyObject* out;

    (void)self;
    if(!PyArg_ParseTuple(args, "OOid", &Wobj, &Hobj, &max_iter, &eps)) return NULL;

    W = pylist_to_matrix(Wobj, &nW, &cW);
    if(!W){ set_error(); return NULL; }

    H = pylist_to_matrix(Hobj, &nH, &k);
    if(!H){ free_matrix(W, nW); set_error(); return NULL; }

    if(nW != cW || nH != nW){
        free_matrix(W, nW);
        free_matrix(H, nH);
        set_error();
        return NULL;
    }

    Hfinal = symnmf_optimize(W, H, nW, k, max_iter, eps);
    free_matrix(W, nW);
    if(!Hfinal){ set_error(); return NULL; }

    out = matrix_to_pylist(Hfinal, nW, k);
    free_matrix(Hfinal, nW);
    return out;
}

static PyMethodDef SymNMFMethods[] = {
    {"sym",    (PyCFunction)py_sym,    METH_VARARGS, "Compute similarity matrix A from X"},
    {"ddg",    (PyCFunction)py_ddg,    METH_VARARGS, "Compute diagonal degree matrix D from X"},
    {"norm",   (PyCFunction)py_norm,   METH_VARARGS, "Compute normalized similarity matrix W from X"},
    {"symnmf", (PyCFunction)py_symnmf, METH_VARARGS, "Optimize H for SymNMF given W and initial H"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    NULL,
    -1,
    SymNMFMethods
};

PyMODINIT_FUNC PyInit_symnmfmodule(void){
    return PyModule_Create(&symnmfmodule);
}
