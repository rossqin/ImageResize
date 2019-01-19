#include "StdAfx.h"
#include "batch_matrix.h"
/*

for 3x3 filter,    index

A00 A01 A02         0 1 2
A10 A11 A12         3 4 5
A20 A21 A22         6 7 8

rotate to 

A22 A21 A20
A12 A11 A10
A02 A01 A00

data[0] ~ data[8 - 0]
data[1] ~ data[8 - 1]
data[2] ~ data[8 - 2]
data[3] ~ data[8 - 3]
*/
__global__ static void bm_rot180_kernel(float* data, int elements, int size) {
	data += blockIdx.x * elements; 
	int n = (--elements) >> 1;
	float temp;
	for (int i = 0; i < n; i++) {
		temp = data[i];
		data[i] = data[elements - i];
		data[elements - i] = temp;
	}
	
}
bool BatchMatrix::Rot180() {
	if (cols != rows) return false;
	if (1 >= cols ) return true;
	int e = cols * rows;

	// mats - gridsize 
	bm_rot180_kernel<<<mats, 1>>>(mem_data, e , cols); 

	return cudaSuccess == cudaDeviceSynchronize();
}
__global__ static void bm_add_kernel(float* out, float* op, int elements, int threads) {
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	while (index < elements) {
		out[index] += op[index];
		index += threads;
	}
}
bool BatchMatrix::Add(const BatchMatrix& right) {
	if (cols != right.cols || rows != right.rows || mats != mats || 0 == mem_elements)
		return false;
	int g = GPUGridSize(9999);
	int b = GPUBlockSize(9999);
	int threads = g * b;
	bm_add_kernel <<<g, b >>>(mem_data, right.mem_data, mem_elements, threads);
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		cerr << "BatchMatrix::Add failed err " << err << "!" << endl;
		return false;
	}
	return true; 
}

__global__ static void bm_add_kernel(float* out, float val, int elements, int threads) {
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	while (index < elements) {
		out[index] += val;
		index += threads;
	}
}
bool BatchMatrix::Add(float val) {
	if ( 0 == mem_elements) return false;
	int g = GPUGridSize(9999);
	int b = GPUBlockSize(9999);
	int threads = g * b;
	bm_add_kernel <<<g, b >>>(mem_data, val, mem_elements, threads);
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		cerr << "BatchMatrix::Add failed err " << err << "!" << endl;
		return false;
	}
	return true;
}
__global__ static void bm_mul_kernel(float* out, float val, int elements, int threads) {
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	while (index < elements) {
		out[index] *= val;
		index += threads;
	}
}
bool BatchMatrix::Mul(float val) {
	if (0 == mem_elements) return false;
	int g = GPUGridSize(9999);
	int b = GPUBlockSize(9999);
	int threads = g * b;
	bm_mul_kernel <<<g, b >>>(mem_data, val, mem_elements, threads);
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		cerr << "BatchMatrix::Mul failed cudaSynchronize failed err " << err << "!" << endl;
		return false;
	}
	return true;
}
__global__ static void bm_muladd_kernel(float* out, float scale, float bias, int elements, int threads) {
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	while (index < elements) {
		out[index] = out[index] * scale + bias;
		index += threads;
	}
}
bool BatchMatrix::MulAdd(float scale, float bias) {
	if (0 == mem_elements) return false;
	int g = GPUGridSize(9999);
	int b = GPUBlockSize(9999);
	int threads = g * b;
	bm_muladd_kernel <<<g, b >>>(mem_data, scale, bias, mem_elements, threads);
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		cerr << "BatchMatrix::MulAdd failed cudaSynchronize failed err " << err << "!" << endl;
		return false;
	}
	return true;
}
__global__ static void bm_muladd_kernel(float* out, float scale, float* op, int elements, int threads) {
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	while (index < elements) {
		out[index] = out[index] * scale + op[index];
		index += threads;
	}
}
bool BatchMatrix::MulAdd(float scale, const BatchMatrix& right) {
	if (cols != right.cols || rows != right.rows || mats != mats || 0 == mem_elements)
		return false;
	int g = GPUGridSize(9999);
	int b = GPUBlockSize(9999);
	int threads = g * b;
	bm_muladd_kernel <<<g, b >>>(mem_data, scale, right.mem_data, mem_elements, threads);
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		cerr << "BatchMatrix::MulAdd failed cudaSynchronize failed err " << err << "!" << endl;
		return false;
	}
	return true;
}
//TODO: when different demensions 
bool BatchMatrix::SafeAdd(const BatchMatrix& right) {
	return Add(right);
	
}

__global__ static void bm_upsample_kernel(float* dest, float* src, int rows, 
	int cols, int elements, int stride_r, int stride_c,int threads,float scale) {
	int index = blockDim.x  * blockIdx.x + threadIdx.x;
	int channel_size = rows * cols;
	int n_rows = rows * stride_r;
	int n_cols = cols * stride_c;
	int new_channel_size = n_rows * n_cols ;

	int r, c,channel, temp , i_dest ;
	while (index < elements) {
		channel = index / channel_size;
		temp = index % channel_size;
		r = temp / cols;
		c = temp % cols;
		i_dest = new_channel_size * channel + (r * stride_r) * n_cols + (c * stride_c);
		if (1.0 == scale)
			dest[i_dest] = src[index];
		else
			dest[i_dest] = src[index] * scale;
		index += threads;
	}
}
bool BatchMatrix::UpSample(BatchMatrix& result, int stride_r, int stride_c, float scale )const {
	if (stride_r <= 0 || stride_c <= 0 || 0 == mem_elements ) return false;
	
	int new_rows = rows * stride_r;
	int new_cols = cols * stride_c;

	if (result.rows != new_rows || result.cols != new_cols || result.mats != mats) {
		result.Release();
		if(!result.Init(new_rows, new_cols, mats))
			return false ;
	} 

	int g = GPUGridSize(9999);
	int b = GPUBlockSize(9999);
	int threads = g * b;
	bm_upsample_kernel<<<g,b>>>(result.mem_data, mem_data, rows, cols, mem_elements, stride_r, stride_c, threads, scale);
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		cerr << "Matrix::Upsample failed - cudaSynchronize failed err "<< err <<"!" << endl;		 
		return false;
	}
	return true;
}
__global__ static void bm_downsample_kernel(float* dest, float* src, int rows,
	int cols, int elements, int stride_r, int stride_c, int threads, float scale) {
	int index = blockDim.x  * blockIdx.x + threadIdx.x;

	int channel_size = rows * cols;
	int n_rows = rows * stride_r;
	int n_cols = cols * stride_c;
	int new_channel_size = n_rows * n_cols;

	int r, c, channel, temp, i_src;
	while (index < elements) {
		channel = index / channel_size;
		temp = index % channel_size;
		r = temp / cols;
		c = temp % cols;
		i_src = new_channel_size * channel + (r * stride_r) * n_cols + (c * stride_c);
		if (1.0 == scale)
			dest[index] = src[i_src];
		else
			dest[index] = src[i_src] * scale;
		index += threads;
	}
}
bool BatchMatrix::DownSample(BatchMatrix& result, int stride_r, int stride_c, float scale) const {
	if (stride_r <= 0 || stride_c <= 0 || 0 == mem_elements) return false;

	int new_rows = rows / stride_r;
	int new_cols = cols / stride_c;

	if (result.rows != new_rows || result.cols != new_cols || result.mats != mats) {
		result.Release();
		if (!result.Init(new_rows, new_cols, mats))
			return false;
	}
	
	int g = GPUGridSize(9999);
	int b = GPUBlockSize(9999);
	int threads = g * b;
	bm_downsample_kernel <<<g, b >>>(result.mem_data, mem_data, new_rows, new_cols, result.mem_elements, stride_r, stride_c, threads, scale);
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		cerr << "Matrix::Upsample failed - cudaSynchronize failed err " << err << "!" << endl; 
		return false;
	}
	return true;
}