#include "StdAfx.h"
#include "matrix.h"
#include <curand_kernel.h>
/*
Notice:
For GTX 1060, max thread block is 1024 = 32 x 32

*/
 
/*
Assumiing only 1 GPU
==================================
Device 0: "GeForce GTX 1060"
CUDA Driver Version / Runtime Version          9.2 / 9.2
CUDA Capability Major/Minor version number:    6.1
Total amount of global memory:                 6144 MBytes (6442450944 bytes)
(10) Multiprocessors, (128) CUDA Cores/MP:     1280 CUDA Cores
GPU Max Clock rate:                            1671 MHz (1.67 GHz)
Memory Clock rate:                             4004 Mhz
Memory Bus Width:                              192-bit
L2 Cache Size:                                 1572864 bytes
Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
Total amount of constant memory:               65536 bytes
Total amount of shared memory per block:       49152 bytes
Total number of registers available per block: 65536
Warp size:                                     32
Maximum number of threads per multiprocessor:  2048
Maximum number of threads per block:           1024
Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
Maximum memory pitch:                          2147483647 bytes
Texture alignment:                             512 bytes
Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
Run time limit on kernels:                     Yes
Integrated GPU sharing Host Memory:            No
Support host page-locked memory mapping:       Yes
Alignment requirement for Surfaces:            Yes
Device has ECC support:                        Disabled
CUDA Device Driver Mode (TCC or WDDM):         WDDM (Windows Display Driver Model)
Device supports Unified Addressing (UVA):      Yes
Device supports Compute Preemption:            No
Supports Cooperative Kernel Launch:            No
Supports MultiDevice Co-op Kernel Launch:      No
Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
*/
 
cudaStream_t Matrix::stream = NULL;
cudaStream_t Matrix::cudaStream() {
	if (NULL == stream) {
		cudaError_t err = cudaStreamCreate(&stream);
	}
	return stream;
		
}

Matrix::Matrix(int r, int c,float *vals) :rows(r), cols(c), elements(NULL) {
	if (r > 0 && c > 0) {
		size_t nBytes = cols * rows * sizeof(float);
		cudaError_t err = cudaMalloc((void**)&elements, nBytes);//cudaMallocManaged((void**)&elements, nBytes);
		if (cudaSuccess == err ) {
			// randomize 
			if (vals)
				cudaMemcpy(elements, vals, nBytes, cudaMemcpyHostToDevice);
			
		}
	}
}
Matrix::Matrix(int r, int c, float val):rows(r), cols(c),elements(NULL) {
	if (r > 0 && c > 0) {
		size_t nBytes = cols * rows * sizeof(float);
		cudaError_t err = cudaMallocManaged((void**)&elements, nBytes);
		if (cudaSuccess == err) {
			int i = 0;
			for (int x = 0; x < cols; x++) {
				for (int y = 0; y < rows; y++) {
					elements[i++] = val;
				}
			}
		}
	}
  
}
 
Matrix::~Matrix() {
	if (elements) {
		cudaFree(elements);
		elements = NULL;
	}
	else {
		//cout << "<Matrix Destructor with Empty content>" << endl;
	}
	//
}

Matrix::Matrix(const Matrix & m): rows(m.rows), cols(m.cols), elements(NULL) {
	//cout << "<Copy Contructor>\n";
 	if (m.elements) {
		size_t nBytes = cols * rows * sizeof(float);
		cudaError_t err = cudaMallocManaged((void**)&elements, nBytes);
		if (cudaSuccess != err) {
			cerr << "CUDA Error `" << cudaGetErrorString(err) << "`" << endl;
			//TODO: report error or throw exception
			return ;
		}
		cudaMemcpyAsync(elements, m.elements, nBytes, cudaMemcpyDefault, cudaStream());
	}
}
Matrix::Matrix(const char* filename):rows(0), cols(0),elements(NULL){
	ifstream file(filename);
	if (!file.is_open()) { 
		cerr << "Failed to load matrix from file:" << filename << endl;
		return;
	}


	vector<float*> vals ;	
	string line;
	while (getline(file, line)) {
		if (0 == cols )
			cols = count_fields(line.c_str());		 
		vals.push_back(parse_fields(line.c_str(), line.length(), cols)); 
	}
	rows = vals.size();
	file.close();
	size_t nBytes = cols * rows * sizeof(float);
	cudaError_t err = cudaMallocManaged((void**)&elements, nBytes);
	if (cudaSuccess == err) {  
		float* dest = elements;
		int bytes = cols * sizeof(float);
		for (int i = 0; i < rows; i++) {
			float* p = vals[i];
			err = cudaMemcpyAsync(dest, p, bytes, cudaMemcpyDefault, cudaStream());
			dest += cols;
			delete[]p;
		}
	}


}
__global__ void random_ren_kernel(float* elements, int rows, int cols, long rand) {
	curandState state;
	for (int y = blockIdx.x; y < rows; y += gridDim.x) {
		int offset = cols * y;
		for (int x = threadIdx.x; x < cols; x += blockDim.x) {
			int index = offset + x;
			curand_init(rand, index, 0, &state);
			elements[index] = curand_uniform_double(&state);
		}
	}
}
void Matrix::Randomize() {
	if (NULL == elements) return;
 
	unsigned int rand_val = 0;
	rand_s(&rand_val);
	random_ren_kernel <<< GPUGridSize(rows), GPUBlockSize(cols) >>> (elements,rows, cols, rand_val);
	cudaDeviceSynchronize();
}
const Matrix & Matrix::operator=(const Matrix & m) {
	//cout << "<operator=>\n";
	size_t nBytes = m.cols * m.rows * sizeof(float);
	if (cols != m.cols || rows != m.rows) {
		if (elements) {
			cudaFree(elements);
			elements = NULL;
		}
	}
	cols = m.cols;
	rows = m.rows;
	if (NULL == elements) {
		cudaError_t err = cudaMallocManaged((void**)&elements, nBytes);
		if (cudaSuccess != err) {
			//TODO: report error or throw exception
			return *this;
		}
	}
	cudaMemcpyAsync(elements, m.elements, nBytes, cudaMemcpyDeviceToDevice, cudaStream());
	return *this;
}
__global__ static void mat_assign_kernel(float* dest, int rows, int cols, float val) {

	for (int y = blockIdx.x; y < rows; y += gridDim.x) {
		int offset = cols * y;
		for (int x = threadIdx.x; x < cols; x += blockDim.x) {
			int index = offset + x;
			dest[index] =  val;
		}
	}
}
const Matrix& Matrix::operator=(float val) {
	if (elements) {
		mat_assign_kernel<<<GPUGridSize(rows), GPUBlockSize(cols) >>>(elements, rows, cols, val);
	}
	return *this;
}
 


#if 0
/*
这个函数这么写有内存非法访问的问题，因为C.cols的地址是host的！！！！
*/
__global__ static void mat_add_kernel(Matrix& C, const Matrix& A, const Matrix& B) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int index = y * C.cols + x;
	int range = C.cols * C.rows;
	if (index < range);
	//C.elements[index] = A.elements[index] + B.elements[index]; 

}
#endif 
// dest: rows x cols 
// src1: rows x width
// src2: width x cols
__global__ static void mat_multi_kernel(float* dest, float* src1, float* src2, 
	int rows, int cols, int width) {
	for (int y = blockIdx.x; y < rows; y += gridDim.x) {
		int offset = cols * y;
		for (int x = threadIdx.x; x < cols; x += blockDim.x) {
			int index = offset + x;
			dest[index] = 0.0;
			for (int i = 0; i < width; i++) {
				dest[index] += src1[i + y * width] * src2[i * cols + x];
			}
		}
	}
}
__global__ static void mat_multi_kernel(float* dest, float* src, int rows, int cols,  float val) {

	for (int y = blockIdx.x; y < rows; y += gridDim.x) {
		int offset = cols * y;
		for (int x = threadIdx.x; x < cols; x += blockDim.x) {
			int index = offset + x;
			dest[index] = src[index] * val;
		}
	}
}

Matrix operator*(const Matrix& A, const Matrix& B) {
	
	if (A.cols != B.rows) {
		//TODO: Report Error
		return Matrix(0,0);
	}
	Matrix C(A.rows, B.cols); 
	mat_multi_kernel <<< GPUGridSize(C.rows), GPUBlockSize(C.cols) >>> (C.elements, A.elements, B.elements, C.rows, C.cols, A.cols);
	cudaDeviceSynchronize();
	return C;
}

Matrix& Matrix::operator*=(float val) {

	mat_multi_kernel <<<GPUGridSize(rows), GPUBlockSize(cols) >>> (elements, elements, rows, cols, val);
	cudaDeviceSynchronize();
	return *this;
}

Matrix operator*(const Matrix& A, float val) {
	Matrix C(A.cols, A.rows);
	mat_multi_kernel <<<GPUGridSize(C.rows), GPUBlockSize(C.cols) >>> (C.elements, A.elements, A.rows, A.cols,  val);
	cudaDeviceSynchronize();
	return C;
}
__global__ static void mat_add_kernel(float* dest, float* src1, float* src2, int rows, int cols) {
	
	for (int y = blockIdx.x; y < rows; y += gridDim.x) {
		int offset = cols * y;
		for (int x = threadIdx.x; x < cols; x += blockDim.x) {
			int index = offset + x;
			dest[index] = src1[index] + src2[index];
		}
	} 
	
}
__global__ static void mat_add_kernel(float* dest, float* src, int rows, int cols,  float val) {

	for (int y = blockIdx.x; y < rows; y += gridDim.x) {
		int offset = cols * y;
		for (int x = threadIdx.x; x < cols; x += blockDim.x) {
			int index = offset + x;
			dest[index] = src[index] + val;
		}
	} 

}

Matrix operator+ (const Matrix& A, const Matrix& B) {
	if (A.cols != B.cols || A.rows != B.rows) {
		//TODO: report Error
		return Matrix(0, 0);
	}
	Matrix C(A.rows, A.cols); 
	 
	mat_add_kernel <<<GPUGridSize(C.rows), GPUBlockSize(C.cols) >>> (C.elements, A.elements , B.elements, C.rows, C.cols);
	cudaDeviceSynchronize();	
	return C;
}

Matrix operator+(const Matrix& A, float val) {
	Matrix C(A.rows, A.cols); 
	mat_add_kernel <<<GPUGridSize(C.rows), GPUBlockSize(C.cols) >>> (C.elements, A.elements, A.rows, A.cols, val);
	cudaDeviceSynchronize();
	return C;
}
Matrix operator-(const Matrix& A, float val) {
	Matrix C(A.rows, A.cols);
	mat_add_kernel <<<GPUGridSize(C.rows), GPUBlockSize(C.cols) >>> (C.elements, A.elements, A.rows, A.cols, 0.0 - val);
	cudaDeviceSynchronize();
	return C;
}
Matrix & Matrix::operator+=(const Matrix& B) {
	if (cols == B.cols && rows == B.rows) {
		mat_add_kernel <<<GPUGridSize(rows), GPUBlockSize(cols) >>> (elements, elements, B.elements, rows, cols);
		cudaDeviceSynchronize();
	}
	return *this;
}
Matrix& Matrix::operator+=(float val) {
	mat_add_kernel <<<GPUGridSize(rows), GPUBlockSize(cols) >>> (elements, elements, rows, cols, val);
	cudaDeviceSynchronize();
	return *this;
}
__global__ static void mat_sub_kernel(float* dest, float* src1, float* src2, int rows, int cols) {
	for (int y = blockIdx.x; y < rows; y += gridDim.x) {
		int offset = cols * y;
		for (int x = threadIdx.x; x < cols; x += blockDim.x) {
			int index = offset + x;
			dest[index] = src1[index] - src2[index];
		}
	}
}
Matrix& Matrix::operator-=(const Matrix& B) {
	if (cols == B.cols && rows == B.rows) {
		mat_sub_kernel <<<GPUGridSize(rows), GPUBlockSize(cols) >>> (elements, elements, B.elements, rows, cols);
		cudaDeviceSynchronize();
	}
	return *this;
}
Matrix operator-(const Matrix& A, const Matrix& B) {
	if (A.cols != B.cols || A.rows != B.rows) {
		//TODO: report Error
		return Matrix(0, 0);
	}
	Matrix C(A.rows, A.cols);
	mat_sub_kernel <<<GPUGridSize(C.rows), GPUBlockSize(C.cols) >>> (C.elements, A.elements, B.elements, C.rows,C.cols);
	cudaDeviceSynchronize();

	return C;
}
struct mat_conv_params {
	float *result;
	float *source;
	float *filter;
	
	int res_rows;
	int res_cols;

	int src_rows;
	int src_cols;

	int fil_rows;
	int fil_cols;

	int stride;
	int padding; 
};

__global__ static void mat_conv_kernel(mat_conv_params* params) {
	float sum_val;
	int offset_r,index_r ;
	int offset_s,src_y_top, src_x_left;
	int offset_f;

	// This is very weird
	int fr = params->fil_rows, fc = params->fil_cols; 
	int stride = params->stride;
	int padding = params->padding;
	int width = params->res_cols;
	int height = params->res_rows;
	float* result = params->result;
	float* src = params->source;
	float* filter = params->filter;
	int width_s = params->src_cols;
	int height_s = params->src_rows; 
	int y_s, x_s;
	for (int y = blockIdx.x; y < height; y += gridDim.x) {
		offset_r = width * y;
		src_y_top = y * stride - padding;
		for (int x = threadIdx.x; x < width; x += blockDim.x) {
			src_x_left = x * stride - padding;
			index_r = offset_r + x;
			sum_val = 0.0;
			offset_f = 0;
			for (int y_f = 0; y_f < fr; y_f++) {
				y_s = src_y_top + y_f;
				if (y_s >= 0 && y_s < height_s) {
					offset_s = y_s * width_s;
					for (int x_f = 0; x_f < fc; x_f++) {
						x_s = src_x_left + x_f;
						if (x_s >= 0 && x_s < width_s) {
							sum_val += src[offset_s + x_s] * filter[offset_f + x_f];
						}
					}
				}
				offset_f += fc;//params->fil_cols;
			} 
			result[index_r] = sum_val; 
		} 
	}
}
bool Matrix::Correlate(Matrix& result, const Matrix& filter, int stride , int padding ) {
	// only allow FxF filters
	if (NULL == elements || NULL == filter.elements ||  padding < 1 || stride < 1)
		return false;

	cudaError_t err;
 
	//WxW <- NxN * FxF
	// N = (W - F + 2P) / S + 1 
	int new_cols = (cols - filter.cols + 2 * padding) / stride + 1; // checked
	int new_rows = (rows - filter.rows + 2 * padding) / stride + 1; // checked
	 
	if (result.cols != new_cols || result.rows != new_rows) {
		result.cols = new_cols;
		result.rows = new_rows;
		if (result.elements)  cudaFree(result.elements); 
		err = cudaMalloc(&result.elements, new_cols * new_rows * sizeof(float));
		if (err != cudaSuccess)
			return false; 
	}
	
	dim3 block_size(GPUBlockSize(new_cols));
	dim3 grid_size(GPUGridSize(new_rows));
	
	mat_conv_params* params = NULL;
	if (cudaSuccess != cudaMallocManaged(&params, sizeof(mat_conv_params)))
		return false;

	params->source = elements;
	params->filter = filter.elements;
	params->result = result.elements;

	params->src_cols = cols; params->src_rows = rows;
	params->res_cols = new_cols; params->res_rows = new_rows;
	params->fil_cols = filter.cols; params->fil_rows = filter.rows;

	params->stride = stride; params->padding = padding;
 

	mat_conv_kernel <<<grid_size, block_size >>> (params);
	err = cudaDeviceSynchronize();
	err = cudaFree(params); 
	return cudaSuccess == err;
}
void Matrix::Rob(Matrix& poor) {
	if (elements) {
		cudaFree(elements);
		elements = NULL;
	}
	cols = poor.cols;
	rows = poor.rows;
	elements = poor.elements;
	poor.elements = NULL;
	poor.cols = 0;
	poor.rows = 0;
		

}
void Matrix::SetElement(int row, int col, float val)
{
	if (row < 0 || row >= rows || col < 0 || col >= cols) return;
	elements[row * cols + col] = val;
}
__global__ static void mat_mean_kernel(float* src, int rows, int cols, float* means,
	float* maxes, float* mins) {
	float mean_val = 0.0;
	float max_val = -99999999.0;
	float min_val =  99999999.0;
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x ;
	for (int y = blockIdx.x; y < rows; y += gridDim.x) {
		int offset = cols * y;
		for (int x = threadIdx.x; x < cols; x += blockDim.x) {
			int index = offset + x;
			mean_val +=  src[index] ; 
			if (max_val < src[index]) max_val = src[index];
			if (min_val > src[index]) min_val = src[index];
		}
	}
	means[thread_id] = mean_val;
	if (maxes)
		maxes[thread_id] = max_val;
	if (mins)
		mins[thread_id] = min_val;

}

float Matrix::Mean() const {
	int count = rows * cols;

	dim3 block_size(GPUBlockSize(cols));
	dim3 grid_size(GPUGridSize(rows));
	int threads = block_size.x * grid_size.x ;
	float* mean_vals = NULL;
	cudaMalloc(&mean_vals, threads * sizeof(float));
	if (NULL == mean_vals) {
		//TODO: report error;
		return 0.0;
	}
	mat_mean_kernel <<<grid_size, block_size >>>(elements, rows, cols, mean_vals,NULL,NULL);
	cudaDeviceSynchronize();
	float* mean_v_cpu = new float[threads];
	cudaError_t err = cudaMemcpy(mean_v_cpu, mean_vals, threads * sizeof(float), cudaMemcpyDeviceToHost);
	float mean_val = 0.0;
	if (cudaSuccess == err) {
		for (int i = 0; i < threads; i++) {
			mean_val += mean_v_cpu[i];
		}
		mean_val /= count;

	}
	cudaFree(mean_vals);
	delete []mean_v_cpu;
	return mean_val;
}
float Matrix::Sum() const {
	//int count = rows * cols;

	dim3 block_size(GPUBlockSize(cols));
	dim3 grid_size(GPUGridSize(rows));
	int threads = block_size.x * grid_size.x;
	float* sum_vals = NULL;
	cudaMalloc(&sum_vals, threads * sizeof(float));
	if (NULL == sum_vals) {
		//TODO: report error;
		return 0.0;
	}
	mat_mean_kernel <<<grid_size, block_size >>>(elements, rows, cols, sum_vals, NULL, NULL);
	cudaDeviceSynchronize();
	float* sum_v_cpu = new float[threads];
	cudaError_t err = cudaMemcpy(sum_v_cpu, sum_vals, threads * sizeof(float), cudaMemcpyDeviceToHost);
	float sum_val = 0.0;
	if (cudaSuccess == err) {
		for (int i = 0; i < threads; i++) {
			sum_val += sum_v_cpu[i];
		} 

	}
	cudaFree(sum_vals);
	delete[]sum_v_cpu;
	return sum_val;
}
__global__ static void mat_vriance_kernel(float* src, int rows, int cols, float* variances,float mean) {
	float diff = 0.0;
	float variance_val = 0.0;
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int y = blockIdx.x; y < rows; y += gridDim.x) {
		int offset = cols * y;
		for (int x = threadIdx.x; x < cols; x += blockDim.x) {
			int index = offset + x;
			diff = src[index] - mean;
			variance_val += diff * diff; //TODO: check whether this requires a sqrt()
		}
	}
	variances[thread_id] = variance_val;
	 

}
float Matrix::Variance(float mean) const {
	float variance = 0.0;
	int count = rows * cols;
	dim3 block_size(GPUBlockSize(cols));
	dim3 grid_size(GPUGridSize(rows));
	int threads = block_size.x * grid_size.x;
	float* variance_vals = NULL;
	cudaMalloc(&variance_vals, threads * sizeof(float));
	if (NULL == variance_vals) {
		//TODO: report error;
		return 0.0;
	}
	mat_vriance_kernel <<<grid_size, block_size >>>(elements, rows, cols, variance_vals, mean);
	cudaDeviceSynchronize();
	float* variance_v_cpu = new float[threads];
	cudaError_t err = cudaMemcpy(variance_v_cpu, variance_vals, threads * sizeof(float), cudaMemcpyDeviceToHost);
 
	if (cudaSuccess == err) {
		for (int i = 0; i < threads; i++) {
			variance += variance_v_cpu[i];
		}
		variance /= count;

	}
	cudaFree(variance_vals);
	delete[]variance_v_cpu;
	return variance;
}
__global__ static void mat_constrain_kernel(float* data, int rows, int cols,
	float min_val, float range_reci, float con_min, float con_max) {
	float scale = range_reci * (con_max - con_min);
	for (int y = blockIdx.x; y < rows; y += gridDim.x) {
		int offset = cols * y;
		for (int x = threadIdx.x; x < cols; x += blockDim.x) {
			int index = offset + x;
			data[index] = (data[index] - min_val) * scale;
			 
		}
	}
}
bool Matrix::Constrain(float min_, float max_) {
	int count = rows * cols;

	dim3 block_size(GPUBlockSize(cols));
	dim3 grid_size(GPUGridSize(rows));
	int threads = block_size.x * grid_size.x;
	float*  vals = NULL;
	size_t  bytes = threads * sizeof(float);
	cudaMalloc(&vals, bytes * 3);
	if (NULL == vals) {
		//TODO: report error;
		return false;
	}
	float* max_vals = vals + threads;
	float* min_vals = max_vals + threads;
	mat_mean_kernel <<<grid_size, block_size >>>(elements, rows, cols, vals, max_vals, min_vals);
	cudaDeviceSynchronize();
	float* v_cpu = new float[threads];
	cudaError_t err = cudaMemcpy(v_cpu, vals, bytes, cudaMemcpyDeviceToHost);
	float mean_val = 0.0;
	if (cudaSuccess != err) {
		delete[]v_cpu; cudaFree(vals); return false;
	}

	for (int i = 0; i < threads; i++) {
		mean_val += v_cpu[i];
	}
	mean_val /= count;
	float max_val = -99999999.0;
	err = cudaMemcpy(v_cpu, max_vals, bytes, cudaMemcpyDeviceToHost);
	if (cudaSuccess != err) {
		delete[]v_cpu; cudaFree(vals); return false;
	}

	for (int i = 0; i < threads; i++) {
		if (max_val < v_cpu[i]) max_val = v_cpu[i];
	}
	float min_val = 99999999.0;
	err = cudaMemcpy(v_cpu, min_vals, bytes, cudaMemcpyDeviceToHost);
	if (cudaSuccess != err) {
		delete[]v_cpu; cudaFree(vals); return false;
	}

	for (int i = 0; i < threads; i++) {
		if (min_val > v_cpu[i]) min_val = v_cpu[i];
	}
	cudaFree(vals);
	delete[]v_cpu;
	float range = max_val - min_val;
	if (0 == range) return true;
	float reci = 1.0 / range ;

	mat_constrain_kernel <<<grid_size, block_size >>>(elements, rows, cols, min_val, reci, min_, max_);

	return true;
}
void Matrix::Print(int max_rows, int max_cols) {
	if (max_rows > rows || max_rows <= 0) max_rows = rows;
	if (max_cols > cols || max_cols <= 0) max_cols = cols;
	float* buffer = new float[max_cols + 1];
	cudaError_t err;
	float *s = elements;
	int bytes = max_cols * sizeof(float);
	for (int i = 0; i < max_rows; i++) {
		err = cudaMemcpy(buffer, s, bytes, cudaMemcpyDeviceToHost);
		if (cudaSuccess == err) {
			for (int j = 0; j < max_cols; j++)
				printf("  %2.6f", buffer[j]);
			printf("\n");
			s += cols;
		}
		else {
			cerr << " cudaMemcpy error in Matrix.Print() ! err:" << (int)err << endl;
			break;
		}
	}
	delete[]buffer;
}
__global__ static void mat_norm_kernel(float* data, int rows, int cols,
	float mean, float reci_sqrt_var, bool is_pixel) {
	 
	float val;
	for (int y = blockIdx.x; y < rows; y += gridDim.x) {
		int offset = cols * y;
		for (int x = threadIdx.x; x < cols; x += blockDim.x) {
			int index = offset + x;
			val = (data[index] - mean) * reci_sqrt_var;
			if (is_pixel) {
				val = (val + 1) * 0.5;
				if (val > 1.0) val = 1.0; 
				else if (val < 0.0)   val = 0.0; 
			}
			data[index] = val;
		}
	}
}
bool Matrix::Normalize(float mean, float variance, bool is_pixel){
	dim3 block_size(GPUBlockSize(cols));
	dim3 grid_size(GPUGridSize(rows));
	if (0 == variance) {
		*this += (0.0 - mean);
		return true ;
	}
	float reci_sqrt_var = 1.0 / sqrt(variance);

	mat_norm_kernel <<<grid_size, block_size >>>(elements, rows, cols, mean, reci_sqrt_var, is_pixel);
	return true;
}
 

/*
Notice: do not use GPU memory directly since memory allocated by cudaMallocManaged 
is not directly readable.

*/
 ostream& operator<<(ostream& output, const Matrix& m) {
	if (NULL == m.elements) {
		output << "<Empty Matrix>\n";
		return output;
	}
	size_t nBytes = m.cols * m.rows * sizeof(float);
	float* buffer = new float[nBytes];
	cudaError_t err = cudaMemcpy(buffer, m.elements, nBytes, cudaMemcpyDeviceToHost);
	if (cudaSuccess != err) {
		output << "<cudaMemcpy error " << (int)err << ">" << endl;
		delete[]buffer;
		return output;
	}
	 
	int n = 0;
	for (int i = 0; i < m.rows; i++) {
		for (int j = 0; j < m.cols; j++) {
			output << setw(12)  << setprecision(4) << buffer[n]  ;
			n++;
		}
		output << endl;
	}
	delete[]buffer;
	return output;
}
 __global__ static void mat_sum_ip_kernel(float* a, float* b, int rows, int cols, float* results) {
	 int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	 results[thread_id] = 0.0;
	 for (int y = blockIdx.x; y < rows; y += gridDim.x) {
		 int offset = cols * y;
		 for (int x = threadIdx.x; x < cols; x += blockDim.x) {
			 results[thread_id] += a[offset + x] * b[offset + x];
		 }
	 }
 }
 float sum_of_inner_product(const Matrix& A, const Matrix& B) {
	 float ret = 0;
	 if (A.cols != B.cols || A.rows != B.rows)
		 return ret;
	 dim3 block_size(GPUBlockSize(A.cols));
	 dim3 grid_size(GPUGridSize(A.rows));
	 int threads = block_size.x * grid_size.x;
	 float* sum_vals = NULL;
	 cudaMalloc(&sum_vals, threads * sizeof(float));
	 if (NULL == sum_vals) {
		 //TODO: report error;
		 return 0.0;
	 }
	 mat_sum_ip_kernel <<<grid_size, block_size >>>(A.elements,B.elements, A.rows, A.cols, sum_vals);
	 cudaDeviceSynchronize();
	 float* sum_v_cpu = new float[threads];
	 cudaError_t err = cudaMemcpy(sum_v_cpu, sum_vals, threads * sizeof(float), cudaMemcpyDeviceToHost);
	 
	 if (cudaSuccess == err) {
		 for (int i = 0; i < threads; i++) {
			 ret += sum_v_cpu[i];
		 } 
	 }
	 cudaFree(sum_vals);
	 delete[]sum_v_cpu; 
	 return ret;

 }

 __global__ static void mat_liner_kernel(float* data, int rows, int cols, float a, float b) {
	 int offset;
	 //float val;
	 for (int y = blockIdx.x; y < rows; y += gridDim.x) {
		 offset = cols * y;
		 for (int x = threadIdx.x; x < cols; x += blockDim.x) {
			 int index = offset + x;
			 data[index] = data[index] * a + b;
		 }
	 }
 }

 bool Matrix::LinearTranform(float a, float b) {
 
	mat_liner_kernel <<<GPUGridSize(rows), GPUBlockSize(cols) >>>(elements, rows, cols, a, b);	
	cudaError_t err = cudaDeviceSynchronize();
	return err == cudaSuccess;
	  
 }
 __global__ static void mat_rot180_kernel(float* data, int rows, int cols) {
 
	 int offset1, offset2;
	 int i1, i2;

	 int max_y = rows >> 1;
	 int max_x = cols;//>> 1;
	 float temp;
	 int y = blockIdx.x;
	 while ( y < max_y) {
		 offset1 = y * cols;
		 offset2 = (rows - 1 - y) * cols;
		 int x = threadIdx.x;
		 while ( x < max_x) {
			 i1 = offset2 + cols - 1 - x;
			 i2 = offset1 + x;
			 temp = data[i1];
			 data[i1] = data[i2];
			 data[i2] = temp;
			 x += blockDim.x;
			
		 }
		 y += gridDim.x;
	 }
 }
 bool Matrix::Rot180() {
	 if (1 == cols && 1 == rows) return true;
	 
	 mat_rot180_kernel <<<GPUGridSize(rows), GPUBlockSize(cols) >>>(elements, rows, cols);
	 cudaError_t  err = cudaDeviceSynchronize();
	 
	 return err == cudaSuccess;
 }
 
 __global__ static void mat_safe_add_kernel(float* out, float* A, float* B, int size, int a_size, int b_size){  
 
	 float sample_a = (float)a_size / size;
	 float sample_b = (float)b_size / size; 
	 int offset_a, offset_b,index_a, index_b,index;
	 for (int y = blockIdx.x; y < size; y += gridDim.x) {
		 offset_a = y * sample_a * a_size;
		 offset_b = y * sample_b * b_size;
		 for (int x = threadIdx.x; x < size; x += blockDim.x) { 
			 index = y * size + x;
			 index_a = offset_a + x * sample_a;
			 index_b = offset_b + x * sample_b;
			 out[index] = A[index_a] + B[index_b];
		 }
	 }
 }
 bool Matrix::SafeAdd(const Matrix& A, const Matrix& B) {

	 //Only square matrix supported.
	 if (cols != rows || A.cols != A.rows || B.cols != B.rows)
		 return false;
	 int g;
	 int b = GPUBlockSize(cols);
	 if (b == cols)
		 g = rows;
	 else
		 g = GPUGridSize(rows);

	 mat_safe_add_kernel <<<g, b>>> (elements, A.elements, B.elements, cols, A.cols, B.cols );
	 return cudaSuccess == cudaDeviceSynchronize();
 }

__global__ static void mat_upsample_kernel(float* A, int rows_A, int cols_A, 
										const float* B, int rows_B,int cols_B, 
										float stride_r, float stride_c, float scale) {
	 int offset_A, offset_B;
	 int x_A;
	 for (int y = blockIdx.x; y < rows_B; y += gridDim.x) {
		 offset_A = (int)(cols_A * y * stride_r);
		 offset_B = cols_B * y;
		 for (int x = threadIdx.x; x < cols_B; x += blockDim.x) {
			 x_A = (int)(x * stride_c);
			 if (x_A < cols_A) {
				 A[offset_A + x_A] = B[offset_B + x] * scale;
			 }
		 }
	 }
 }

 // always assuming *this has been set to zero before this 
 bool Matrix::UpSample(const Matrix& B, float stride_r, float stride_c, float scale) {
	 
	if (B.cols * stride_c != cols || B.rows * stride_r != rows)
		return false;
	mat_upsample_kernel <<<GPUGridSize(B.rows), GPUBlockSize(B.cols) >>> (elements, rows, cols, B.elements, B.rows, B.cols, stride_r, stride_c, scale);
 
	 return cudaSuccess == cudaDeviceSynchronize();
 }
 __global__ static void mat_downsample_kernel(float* A, int rows_A, int cols_A,
	 const float* B, int rows_B, int cols_B,
	 float stride_r, float stride_c, float scale) {
	 int offset_A, offset_B;
	 int x_B;
	 for (int y = blockIdx.x; y < rows_A; y += gridDim.x) {
		 offset_A = cols_A * y;
		 offset_B = (int)(cols_B * y * stride_r);
		 for (int x = threadIdx.x; x < cols_A; x += blockDim.x) {
			 x_B = (int)(x * stride_c);
			 if (x_B < cols_B) {
				 //TODO: yolo代码中是 += 而不是+
				 A[offset_A + x] = B[offset_B + x_B] * scale;
			 }
		 }
	 }
 }
 bool Matrix::DownSample(const Matrix& B, float stride_r, float stride_c, float scale) {
	 if (B.cols != cols * stride_c || B.rows != rows * stride_r)
		 return false;
	 mat_downsample_kernel <<<GPUGridSize(rows), GPUBlockSize(cols) >>> (elements, rows, cols, B.elements, B.rows, B.cols, stride_r, stride_c, scale);

	 return cudaSuccess == cudaDeviceSynchronize();
 }
 __global__ static void bm_add_cor_fwd_kernel(float* out, float* A, float* W,
	 int o_size, int a_size, int w_size, int stride, int padding) {
	 int y_start_A, x_start_A,y_A,x_A ;
	 int index,i,j,index_A,index_W;
	 for (int y = blockIdx.x; y < o_size; y += gridDim.x) {
		 y_start_A = y * stride - padding; 
		 for (int x = threadIdx.x; x < o_size; x += blockDim.x) {
			 x_start_A = x * stride - padding; 
			 index = y * o_size + x; 
			 if (1 == w_size) { // weight: 1x1
				 if(0.0f == *W || y_start_A < 0 || y_start_A >= a_size ||
					 x_start_A < 0 || x_start_A >= a_size )
				 continue;
				 index_A = y_start_A * a_size + x_start_A;
				 if (0.0f != A[index_A])
					 out[index] += A[index_A] * (*W);
				 continue;
			 }
			 // weight: 3x3 etc.
			 for (j = 0, y_A = y_start_A; j < w_size; j++, y_A++) {
				 if (y_A >= 0 && y_A < a_size) {
					 for (i = 0, x_A = x_start_A; i < w_size; i++, x_A++) {
						 if (x_A >= 0 && x_A < a_size) {
							 index_A = y_A * a_size + x_A;
							 index_W = j * w_size + i;
							 if (A[index_A] != 0.0f && W[index_W] != 0.0f)
								 out[index] += A[index_A] * W[index_W];
						}//if (x_A >= 0 && x_A < a_size)
					 } // for i
				 }// if (y_A >= 0 && y_A < a_size) 
			 }// for j
		 }// for x 
	 }// for y 

 }
 __global__ static void bm_add_cor_bwd_kernel(float* out, float* A, float* W,
	 int o_size, int a_size, int w_size, int stride, int padding) {

	 int p_A = ((w_size + 1) >> 1) - padding;
	 //int x_A, y_A;
	 int index_A;
	 for (int y = blockIdx.x; y < o_size; y += gridDim.x) {
		 for (int x = threadIdx.x; x < o_size; x += blockDim.x) {
			 int index = y * o_size + x;
			 for (int m = 0, y_A = y - p_A; m < w_size; m++, y_A++) {
				if (y_A < 0 || y_A >= a_size) continue;
				if (1 == stride || (y_A % stride) == 0) {
					for (int n = 0, x_A = x - p_A; n < w_size; n++, x_A++) {
						if (x_A < 0 || x_A >= a_size) continue;
						if (1 == stride || (x_A % stride) == 0) {
							if (1 == stride)
								index_A = y_A  * a_size + x_A;
							else
								index_A = (y_A / stride) * a_size + (x_A / stride);

							out[index] += A[index_A] * W[m * w_size + n];
						}
					}
				}
				 
			 }
		 }
	 }
 }
 /*
 This function is optimized for convolutional forwarding and backwarding
 see https://blog.csdn.net/weixin_41665225/article/details/84986071
 here we do square matrix calculation only
 */
 bool Matrix::AddCorrelate(Matrix* A, Matrix* B, int stride, int padding, bool forwarding) {
	 int g;
	 int b = GPUBlockSize(cols);
	 if(b == cols)
		g = rows ;
	 else 	 
		g = GPUGridSize(rows);
	 
	 if (forwarding)
		 bm_add_cor_fwd_kernel<<<g, b>>>(elements, A->elements, B->elements, cols, A->cols, B->cols, stride, padding);
	 else
		 bm_add_cor_bwd_kernel<<<g , b>>>(elements, A->elements, B->elements, cols, A->cols, B->cols, stride, padding);
	 return cudaSuccess == cudaDeviceSynchronize();
 }