/********************************************************************************
 *
 * Copyright (c) 2018 ROCm Developer Tools
 *
 * MIT LICENSE:
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "include/rvs_blas.h"

#include <time.h>
#include <limits>

#define RANDOM_CT               320000
#define RANDOM_DIV_CT           0.1234

rocblas_operation transa = rocblas_operation_none;
rocblas_operation transb = rocblas_operation_transpose;

/**
 * @brief class constructor
 * @param _gpu_device_index the gpu that will run the GEMM
 * @param _m matrix size
 * @param _n matrix size
 * @param _k matrix size
 */
rvs_blas::rvs_blas(int _gpu_device_index, int _m, int _n, int _k) :
                             gpu_device_index(_gpu_device_index),
                             m(_m),
                             n(_n),
                             k(_k) {
    is_handle_init = false;
    is_error = false;
    da = db = dc = NULL;
    ha = hb = hc = nullptr;

    size_a = k * m;
    size_b = k * n;
    size_c = n * m;

    if (alocate_host_matrix_mem()) {
        if (!init_gpu_device())
            is_error = true;
    } else {
        is_error = true;
    }
}

/**
 * @brief class destructor
 */
rvs_blas::~rvs_blas() {
    release_host_matrix_mem();
    release_gpu_matrix_mem();
}

/**
 * @brief selects GPU device, allocates GPU memory, creates a rocBlas
 * handle and get a reference to the rocBlas's stream
 * @return true if everything went fine, otherwise false
 */
bool rvs_blas::init_gpu_device(void) {
    // select GPU device & allocate memory
    if (hipSetDevice(gpu_device_index) != hipSuccess) {
        // cannot select the given GPU device
        return false;
    } else {
        if (!allocate_gpu_matrix_mem())
            return false;
        if (rocblas_create_handle(&blas_handle) == rocblas_status_success) {
            is_handle_init = true;
            if (rocblas_get_stream(blas_handle, &hip_stream)
                 != rocblas_status_success)
                return false;
        } else {
            return false;
        }
    }
    return true;
}

/**
 * @brief copy data matrix from host to gpu
 * @return true if everything went fine, otherwise false
 */
bool rvs_blas::copy_hlf_data_to_gpu(void) {
    if (!is_error) {
        if (dhlfa) {
            if (hipMemcpy(dhlfa, hlfa, sizeof(rocblas_half) * size_a, hipMemcpyHostToDevice)
                    != hipSuccess) {
                is_error = true;
                return false;
            }
        }

        if (ddblb) {
            if (hipMemcpy(dhlfb, hlfb, sizeof(rocblas_half) * size_b, hipMemcpyHostToDevice)
                     != hipSuccess) {
                is_error = true;
                return false;
            }
        }

        if (ddblc) {
            if (hipMemcpy(dhlfc, hlfc, sizeof(rocblas_half) * size_c, hipMemcpyHostToDevice)
                     != hipSuccess) {
                is_error = true;
                return false;
            }
        }

        return true;
    } else {
        return false;
    }
}


/**
 * @brief copy data matrix from host to gpu
 * @return true if everything went fine, otherwise false
 */
bool rvs_blas::copy_dbl_data_to_gpu(void) {
    if (!is_error) {
        if (ddbla) {
            if (hipMemcpy(ddbla, hdbla, sizeof(double) * size_a, hipMemcpyHostToDevice)
                    != hipSuccess) {
                is_error = true;
                return false;
            }
        }

        if (ddblb) {
            if (hipMemcpy(ddblb, hdblb, sizeof(double) * size_b, hipMemcpyHostToDevice)
                     != hipSuccess) {
                is_error = true;
                return false;
            }
        }

        if (ddblc) {
            if (hipMemcpy(ddblc, hdblc, sizeof(double) * size_c, hipMemcpyHostToDevice)
                     != hipSuccess) {
                is_error = true;
                return false;
            }
        }

        return true;
    } else {
        return false;
    }
}


/**
 * @brief copy data matrix from host to gpu
 * @return true if everything went fine, otherwise false
 */
bool rvs_blas::copy_data_to_gpu(void) {
    if (!is_error) {
        if (da) {
            if (hipMemcpy(da, ha, sizeof(float) * size_a, hipMemcpyHostToDevice)
                    != hipSuccess) {
                is_error = true;
                return false;
            }
        }

        if (db) {
            if (hipMemcpy(db, hb, sizeof(float) * size_b, hipMemcpyHostToDevice)
                     != hipSuccess) {
                is_error = true;
                return false;
            }
        }

        if (dc) {
            if (hipMemcpy(dc, hc, sizeof(float) * size_c, hipMemcpyHostToDevice)
                     != hipSuccess) {
                is_error = true;
                return false;
            }
        }

        return true;
    } else {
        return false;
    }
}

/**
 * @brief allocates memory (for matrix multiplication) on the selected GPU
 * @return true if everything went fine, otherwise false
 */
bool rvs_blas::allocate_gpu_matrix_mem(void) {
    //SGEMM
    if (hipMalloc(&da, size_a * sizeof(float)) != hipSuccess)
        return false;
    if (hipMalloc(&db, size_b * sizeof(float)) != hipSuccess)
        return false;
    if (hipMalloc(&dc, size_c * sizeof(float)) != hipSuccess)
        return false;

    //DGEMM
    if (hipMalloc(&ddbla, size_a * sizeof(double)) != hipSuccess)
        return false;
    if (hipMalloc(&ddblb, size_b * sizeof(double)) != hipSuccess)
        return false;
    if (hipMalloc(&ddblc, size_c * sizeof(double)) != hipSuccess)
        return false;

    //HGEMM
    if (hipMalloc(&dhlfa, size_a * sizeof(rocblas_half)) != hipSuccess)
        return false;
    if (hipMalloc(&dhlfb, size_b * sizeof(rocblas_half)) != hipSuccess)
        return false;
    if (hipMalloc(&dhlfc, size_c * sizeof(rocblas_half)) != hipSuccess)
        return false;

    return true;
}

/**
 * @brief releases GPU mem & destroys the rocBlas handle
 */
void rvs_blas::release_gpu_matrix_mem(void) {

    //SGEMM
    if (da)
        hipFree(da);
    if (db)
        hipFree(db);
    if (dc)
        hipFree(dc);

    //DGEMM
    if (ddbla)
        hipFree(ddbla);
    if (ddblb)
        hipFree(ddblb);
    if (ddblc)
        hipFree(ddblc);

    //HGEMM
    if (dhlfa)
        hipFree(dhlfa);
    if (dhlfb)
        hipFree(dhlfb);
    if (dhlfc)
        hipFree(dhlfc);

    if (is_handle_init)
        rocblas_destroy_handle(blas_handle);
}

/**
 * @brief allocate host matrix memory
 * @return true if everything went fine, otherwise false
 */
bool rvs_blas::alocate_host_matrix_mem(void) {
    try {
        //SGEMM
        ha = new float[size_a];
        hb = new float[size_b];
        hc = new float[size_c];

        //HGEMM
        hdbla = new double[size_a];
        hdblb = new double[size_b];
        hdblc = new double[size_c];

        //Half GEMM
        hlfa = new rocblas_half[size_a];
        hlfb = new rocblas_half[size_b];
        hlfc = new rocblas_half[size_c];

        return true;
    } catch (std::bad_alloc&) {
        return false;
    }
}

/**
 * @brief releases the host matrix memory
 */
void rvs_blas::release_host_matrix_mem(void) {
  //SGEMM
    if (ha)
        delete []ha;
    if (hb)
        delete []hb;
    if (hc)
        delete []hc;
   //DGEMM
   if (hdbla)
        delete []hdbla;
    if (hdblb)
        delete []hdblb;
    if (hdblc)
        delete []hdblc;
   //HGEMM
   if (hlfa)
        delete []hlfa;
    if (hlfb)
        delete []hlfb;
    if (hlfc)
        delete []hlfc;

}

/**
 * @brief checks whether the matrix multiplication completed
 * @return true if GPU finished with matrix multiplication, otherwise false
 */
bool rvs_blas::is_gemm_op_complete(void) {
    if (is_error)
        return true;  // avoid blocking the calling thread
    if (hipStreamQuery(hip_stream) != hipSuccess)
        return false;
    return true;
}

/**
 * @brief performs the SGEMM matrix multiplication
 * @return true if GPU was able to enqueue the GEMM operation, otherwise false
 */
bool rvs_blas::run_blass_dgemm(void) {
    if (!is_error) {
        double alpha = 1.1, beta = 0.9;
        hipDeviceSynchronize();
        if (rocblas_dgemm(blas_handle, transa, transb,
                    rvs_blas::m, rvs_blas::n, rvs_blas::k,
                    &alpha, ddbla, rvs_blas::m,
                    ddblb, rvs_blas::n, &beta,
                    ddblc, rvs_blas::m) != rocblas_status_success) {
            is_error = true;  // GPU cannot enqueue the gemm
            return false;
        } else {
            return true;
        }
        hipDeviceSynchronize();
    } else {
        return false;
    }
}

/**
 * @brief performs the SGEMM matrix multiplication
 * @return true if GPU was able to enqueue the GEMM operation, otherwise false
 */
bool rvs_blas::run_blass_hgemm(void) {
    hipDeviceSynchronize();
    if (!is_error) {
        #if __clang__ && (__STDC_VERSION__ >= 201112L || __cplusplus >= 201103L)
        rocblas_half alpha = (rocblas_half)1.1, beta = (rocblas_half)0.9;
        #else
        rocblas_half alpha, beta;
        alpha.data = 1;
        beta.data = 1;
        #endif
        hipDeviceSynchronize();
        if (rocblas_hgemm(blas_handle, transa, transb,
                    rvs_blas::m, rvs_blas::n, rvs_blas::k,
                    &alpha, dhlfa, rvs_blas::m,
                    dhlfb, rvs_blas::n, &beta,
                    dhlfc, rvs_blas::k) != rocblas_status_success) {
            is_error = true;  // GPU cannot enqueue the gemm
            return false;
        } else {
            return true;
        }
    } else {
        return false;
    }
    hipDeviceSynchronize();
}


/**
 * @brief performs the SGEMM matrix multiplication
 * @return true if GPU was able to enqueue the GEMM operation, otherwise false
 */
bool rvs_blas::run_blass_gemm(void) {
    hipDeviceSynchronize();
    if (!is_error) {
        float alpha = 1.1, beta = 0.9;
        if (rocblas_sgemm(blas_handle, transa, transb,
                    rvs_blas::m, rvs_blas::n, rvs_blas::k,
                    &alpha, da, rvs_blas::m,
                    db, rvs_blas::n, &beta,
                    dc, rvs_blas::k) != rocblas_status_success) {
            is_error = true;  // GPU cannot enqueue the gemm
            return false;
        } else {
            return true;
        }
    } else {
        return false;
    }
    hipDeviceSynchronize();
}

/**
 * @brief generate matrix random data
 * it should be called before rocBlas GEMM
 */
void rvs_blas::generate_random_half_matrix_data(void) {
    int i;
    if (!is_error) {
        srand (time(NULL));

        #if __clang__ && (__STDC_VERSION__ >= 201112L || __cplusplus >= 201103L)
        for (i = 0; i < size_a; ++i)
            hlfa[i] = genHalfRand();

        for (i = 0; i < size_b; ++i)
            hlfb[i] = genHalfRand();

        for (i = 0; i < size_c; ++i)
            hlfc[i] = genHalfRand();
        #else
        for (i = 0; i < size_a; ++i)
            hlfa[i].data = genHalfRand();

        for (i = 0; i < size_b; ++i)
            hlfb[i].data = genHalfRand();

        for (i = 0; i < size_c; ++i)
            hlfc[i].data = genHalfRand();
        #endif
    }
}

/**
 * @brief generate matrix random data
 * it should be called before rocBlas GEMM
 */
void rvs_blas::generate_random_dbl_matrix_data(void) {
    int i;
    if (!is_error) {
        srand (time(NULL));

        for (i = 0; i < size_a; ++i)
            hdbla[i] = genDoubleRand();

        for (i = 0; i < size_b; ++i)
            hdblb[i] = genDoubleRand();

        for (int i = 0; i < size_c; ++i)
            hdblc[i] = genDoubleRand();
    }
}



/**
 * @brief generate matrix random data
 * it should be called before rocBlas GEMM
 */
void rvs_blas::generate_random_matrix_data(void) {
    int i;
    if (!is_error) {
        srand (time(NULL));

        for (i = 0; i < size_a; ++i)
            ha[i] = genFloatRand();

        for (i = 0; i < size_b; ++i)
            hb[i] = genFloatRand();

        for (int i = 0; i < size_c; ++i)
            hc[i] = genFloatRand();
    }
}

double rvs_blas::genDoubleRand()
{
    double d = (double)rand() / RAND_MAX;
    double dMin =  std::numeric_limits<double>::min();
    double dMax =  std::numeric_limits<double>::max();
    return dMin + d * (dMax - dMin);
}

float rvs_blas::genFloatRand()
{
    float f = (float)rand() / RAND_MAX;
    float fMin =  std::numeric_limits<float>::min();
    float fMax =  std::numeric_limits<float>::max();
    return fMin + f * (fMax - fMin);
}

#if __clang__ && (__STDC_VERSION__ >= 201112L || __cplusplus >= 201103L)
rocblas_half rvs_blas::genHalfRand()
{
    rocblas_half h = (rocblas_half)rand() / RAND_MAX;
    rocblas_half hMin =  std::numeric_limits<rocblas_half>::min();
    rocblas_half hMax =  std::numeric_limits<rocblas_half>::max();
    return hMin + h * (hMax - hMin);
}
#else
uint16_t rvs_blas::genHalfRand()
{
    uint16_t h = (uint16_t)rand() / RAND_MAX;
    uint16_t hMin =  std::numeric_limits<uint16_t>::min();
    uint16_t hMax =  std::numeric_limits<uint16_t>::max();
    return hMin + h * (hMax - hMin);
}
#endif

