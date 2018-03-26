from knn_search.utils import compute_distances, check_pairwise_arrays


def euclidean_distances(X, Y=None):
    """
    Расчет расстояния Евклида для всех пар векторов (строк) 
    матриц X и Y (Y=X, если Y=None)

    :param X: array-like, shape=(n_samples_1, n_features)
    :param Y: array-like, shape=(n_samples_2, n_features)

    """
    X, Y = check_pairwise_arrays(X, Y)

    kernel_code_template = """
        #include <math.h>
        
        __global__ void euclidean(float *x, float *y, float *solution) {
            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            int idy = threadIdx.y + blockDim.y * blockIdx.y;
            
            if ( ( idx < %(NCOLS)s ) && ( idy < %(NROWS)s ) ) {
            
                float result = 0.0;
                
                for(int iter = 0; iter < %(NDIM)s; iter++) {
                    float x_e = x[%(NDIM)s * idx + iter];
                    float y_e = y[%(NDIM)s * idy + iter];
                    result += pow((x_e - y_e), 2);
                }
                
                int pos = idy + %(NROWS)s * idx;
                 
                solution[pos] = sqrt(result);
                
            }
        }
    """
    
    kernel_code = kernel_code_template % {
        "NCOLS": str(X.shape[0]),
        "NROWS": str(Y.shape[0]),
        "NDIM": str(X.shape[1])
    }
    
    return compute_distances("euclidean", kernel_code, X, Y)


def cosine_distances(X, Y=None):
    """
    Расчет косинуса для всех пар векторов (строк) 
    матриц X и Y (Y=X, если Y=None)

    :param X: array-like, shape=(n_samples_1, n_features)
    :param Y: array-like, shape=(n_samples_2, n_features)

    """

    X, Y = check_pairwise_arrays(X, Y)

    kernel_code_template = """
        #include <math.h>
        
        __global__ void cosine(float *x, float *y, float *solution) {
            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            int idy = threadIdx.y + blockDim.y * blockIdx.y;
            
            if ( ( idx < %(NCOLS)s ) && ( idy < %(NROWS)s ) ) {
            

                float sum_ab = 0.0;
                float mag_a = 0.0;
                float mag_b = 0.0;
                
                for(int iter = 0; iter < %(NDIM)s; iter++) {
                    float x_e = x[%(NDIM)s * idx + iter];
                    float y_e = y[%(NDIM)s * idy + iter];
                    
                    sum_ab += x_e * y_e;
                    mag_a += pow(x_e, 2);
                    mag_b += pow(y_e, 2);
                }
                
                int pos = idy + %(NROWS)s * idx;
                 
                solution[pos] = sum_ab / (sqrt(mag_a) * sqrt(mag_b));
                
            }
        }
    """
    
    kernel_code = kernel_code_template % {
        "NCOLS": str(X.shape[0]),
        "NROWS": str(Y.shape[0]),
        "NDIM": str(X.shape[1])
    }
    
    return compute_distances("cosine", kernel_code, X, Y)


def pearson_correlation(X, Y=None):
    """
    Расчет коэффициента Пирсона для всех пар векторов (строк) 
    матриц X и Y (Y=X, если Y=None)

    :param X: array-like, shape=(n_samples_1, n_features)
    :param Y: array-like, shape=(n_samples_2, n_features)

    """

    X, Y = check_pairwise_arrays(X, Y)
    
    kernel_code_template = """
        #include <math.h>
        
        __global__ void pearson(float *x, float *y, float *solution) {
            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            int idy = threadIdx.y + blockDim.y * blockIdx.y;
            
            if ( ( idx < %(NCOLS)s ) && ( idy < %(NROWS)s ) ) {

                float sum_xy, sum_x, sum_y, sum_square_x, sum_square_y;
                sum_x = sum_y = sum_xy = sum_square_x = sum_square_y = 0.0f;

                for(int iter = 0; iter < %(NDIM)s; iter++) {
                    float x_e = x[%(NDIM)s * idx + iter];
                    float y_e = y[%(NDIM)s * idy + iter];

                    sum_x += x_e;
                    sum_y += y_e;
                    sum_xy += x_e * y_e;
                    sum_square_x += pow(x_e, 2);
                    sum_square_y += pow(y_e, 2);
                }
                
                int pos = idy + %(NROWS)s * idx;
                float denom = sqrt(sum_square_x - (pow(sum_x, 2) / %(NDIM)s)) * sqrt(sum_square_y - (pow(sum_y, 2) / %(NDIM)s));
                if (denom == 0) {
                    solution[pos] = 0;
                } else {
                    float quot = sum_xy - ((sum_x * sum_y) / %(NDIM)s);
                    solution[pos] = quot / denom;
                }
            }
        }
    """
    
    kernel_code = kernel_code_template % {
        "NCOLS": str(X.shape[0]),
        "NROWS": str(Y.shape[0]),
        "NDIM": str(X.shape[1])
    }
    return compute_distances("pearson", kernel_code, X, Y)



