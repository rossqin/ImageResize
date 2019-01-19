#include "StdAfx.h"
#include "batch_matrix.h"

typedef void* pvoid_t;
BatchMatrix::BatchMatrix() :rows(0), cols(0), mats(0),
mem_elements(0), mem_bytes(0), mem_data(NULL),elements(NULL)
{
	name[0] = 0;
	inited = false;
}
BatchMatrix::BatchMatrix(int r, int c, int n) {
	inited = false;
	name[0] = 0;
	Init(r, c, n);

}
const BatchMatrix & BatchMatrix::operator=(const BatchMatrix & bm) {
	
	if (!bm.inited) return *this;
	cudaError_t err;
	if(cols != bm.cols || rows != bm.rows || mats != bm.mats){
		Release();
		err = cudaMalloc(&mem_data, bm.mem_bytes);
		if (cudaSuccess != err) {  //TODO:Report Error
			return *this;
		}
	}
	
	err = cudaMemcpy(mem_data, bm.mem_data, bm.mem_bytes, cudaMemcpyDeviceToDevice);
	if (cudaSuccess != err) {  //TODO:Report Error
		cudaFree(mem_data);
		mem_data = NULL;
		return *this;
	}
	if (NULL == elements) {
		mem_bytes = bm.mem_bytes;
		mem_elements = bm.mem_elements;
		rows = bm.rows;
		cols = bm.cols;
		mats = bm.mats;
		elements = new LPMatrix[mats];
		float* p = mem_data;
		size_t mat_size = rows * cols;
		for (int i = 0; i < mats; i++, p += mat_size) {
			Matrix* m = new Matrix(0, 0);
			m->cols = cols; m->rows = rows;
			m->elements = p;
			elements[i] = m;
		}
	}
	inited = true;
	

	return *this;
}
const BatchMatrix & BatchMatrix::operator=(float val) {
	for (int i = 0; i < mats; i++) {
		*(elements[i]) = val;
	}
	return *this;
}
bool BatchMatrix::Init(int r, int c, int n ,float val) {
	if (inited) return true;
	rows = r; cols = c; mats = n;
	mem_elements = rows * cols * mats;
	mem_bytes = mem_elements * sizeof(float);
	mem_data = NULL;

	cudaError_t err = cudaMalloc(&mem_data, mem_bytes);
	if (cudaSuccess != err) {
		cerr << "Error: cuda malloc failed (" << (int)err << ") while initializing batch matrix." << endl;
		// TODO: Report error
		rows = 0; cols = 0; mats = 0;
		mem_bytes = 0; mem_elements = 0;
		return false;
	}
	if (0.0f == val) {
		err = cudaMemset(mem_data, 0, mem_bytes);
		if (cudaSuccess != err) {
			cerr << "Error: cuda memset failed (" << (int)err << ") while initializing batch matrix." << endl;
			// TODO: Report error
			cudaGetLastError();
			cudaFree(mem_data); mem_data = NULL;
			rows = 0; cols = 0; mats = 0;
			mem_bytes = 0; mem_elements = 0;
			return false;
		}
	}
	elements = new LPMatrix[mats];
	float* p = mem_data;
	size_t mat_size = rows * cols;
	for (int i = 0; i < mats; i++, p += mat_size) {
		Matrix* m = new Matrix(0, 0);
		m->cols = cols; m->rows = rows;
		m->elements = p;
		elements[i] = m;
		if(val != 0.0f) *m = val;

	}
	
	inited = true;
	return true;
}
Matrix* BatchMatrix::At(int i) const {
	if (i >= 0 && i < mats) {
		return  elements[i];
	}
	return NULL;
}
BatchMatrix::BatchMatrix(const BatchMatrix& bm): cols(bm.cols),rows(bm.rows),mats(bm.mats) {
	inited = false;
	mem_elements = rows * cols * mats;
	mem_bytes = mem_elements * sizeof(float);
	mem_data = NULL;

	cudaError_t err = cudaMalloc(&mem_data, mem_bytes);
	if (cudaSuccess != err) {
		// TODO: Report error
		rows = 0; cols = 0; mats = 0;
		mem_bytes = 0; mem_elements = 0;
		return ;
	} 
	err = cudaMemcpy(mem_data, bm.mem_data, bm.mem_bytes, cudaMemcpyDeviceToDevice);
	if (cudaSuccess != err) {  //TODO:Report Error
		cudaFree(mem_data);
		mem_data = NULL;
		return ;
	}
	elements = new LPMatrix[mats];
	float* p = mem_data;
	size_t mat_size = rows * cols;
	for (int i = 0; i < mats; i++, p += mat_size) {
		Matrix* m = new Matrix(0, 0);
		m->cols = cols; m->rows = rows;
		m->elements = p;
		elements[i] = m;

	}
	inited = true;
}
BatchMatrix::~BatchMatrix() {
	Release();
}
void BatchMatrix::Release() {
	bool b = false;
	if (strcmp("conv.layer.17.weights", name) == 0){
		b = true;
	}
	// cout <<"Batch matrix `" << name << "` releasing ..." ;
	if (elements != NULL) {
		for (int i = 0; i < mats; i++) {
			Matrix* m = elements[i]; 
			if (b) cout << i << ",";
			m->elements = NULL;
			delete m;
		}
		delete []elements ;
		elements = NULL;
	}
	if (mem_data) {
		cudaFree(mem_data);
		mem_data = NULL;
	}
	inited = false; mem_bytes = 0; mem_elements = 0;
	rows = 0; cols = 0; mats = 0;

	//cout << "\tdone.\n";

}
bool BatchMatrix::GetData(float * out, size_t bytes) {
	if (NULL == out || NULL == mem_data) return false;
	if (0 == bytes || bytes > mem_bytes) bytes = mem_bytes;
	cudaError_t err = cudaMemcpy(out, mem_data, bytes, cudaMemcpyDeviceToHost);
	//TODO: report error 
	return (cudaSuccess == err);
}

float BatchMatrix::Mean() const {

	float val = 0.0f;
	if (0 == mats) return val;
	for (int i = 0; i < mats; i++) {
		Matrix* m = elements[i];
		val += m->Mean();
	}
	return val / mats;
}
bool BatchMatrix::SetElement(int i, const Matrix& m) {
	if (i < 0 || i >= mats) return false;
	if (m.cols != cols || m.rows != rows) return false;
	size_t offset = cols * rows * i;
	size_t bytes = cols * rows * sizeof(float);
	return cudaSuccess == cudaMemcpy(mem_data + offset, m.elements, bytes, cudaMemcpyDeviceToDevice);
}
ofstream & operator<<(ofstream & ofile, const BatchMatrix & bm) {
	if (NULL != bm.mem_data) {
		char* cpu_data = new char[bm.mem_bytes];
		cudaError_t err = cudaMemcpy(cpu_data, bm.mem_data, bm.mem_bytes, cudaMemcpyDeviceToHost);
		if (cudaSuccess == err)
			ofile.write(cpu_data, bm.mem_bytes);
		delete[]cpu_data;
	}
	return ofile;
}
//TODO: add mini-batch
//
//see https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e 
bool BatchMatrix::Correlate(BatchMatrix& result, const BatchMatrix& filters, 
	int stride, int padding,int mini_batch) const {
 
	int new_rows = (rows - filters.rows + 2 * padding) / stride + 1;
	int new_cols = (cols - filters.cols + 2 * padding) / stride + 1;
 
	result.Release();
	if (!result.Init(new_rows, new_cols, filters.mats * mini_batch)) {
		cerr << "Error: Result initialized for correlation failed." << endl;
		return false;
	}
	
	int channels = mats / mini_batch; // input channels 
	int r_base_index = 0;
	int i_base_index = 0; 
	for (int b = 0; b < mini_batch; b++) { 
		for (int i = 0; i < filters.mats; i++) {
			Matrix* result_b_i = result.elements[r_base_index + i];
			Matrix* filter_i = filters.elements[i];
			for (int j = 0; j < channels; j++) {
				Matrix* input_b_j = elements[i_base_index + j];
				if (!result_b_i->AddCorrelate(input_b_j, filter_i, stride, padding)) {
					cerr << "Error: operation failed in correlation (mini_batch "<<b << ", index "<<i<<","<<j<<")." << endl;
					return false ;
				}
			}

		}
		i_base_index += channels;
		r_base_index += filters.mats;
	}

	return true;
}
void BatchMatrix::Rob(BatchMatrix& bm) {
	if (elements != NULL) {
		for (int i = 0; i < mats; i++) {
			Matrix* m = elements[i];
			m->elements = NULL;
			delete m;
		}
		delete[]elements;
		elements = NULL;
	}
	if (mem_data) {
		cudaFree(mem_data);
		mem_data = NULL;
	}
	elements = bm.elements;
	mem_data = bm.mem_data;
	cols = bm.cols;
	rows = bm.rows;
	mats = bm.mats;
	mem_elements = bm.mem_elements;
	mem_bytes = bm.mem_bytes;
	bm.elements = NULL;
	bm.mem_data = NULL;
}
bool BatchMatrix::Transpose() {
	return false;//TODO: finish
}
bool BatchMatrix::ZeroMem() {
	if(mem_data )
		return  cudaSuccess == cudaMemset(&mem_data, 0, mem_bytes);
	return false;
}
bool BatchMatrix::EnsureDim(int r, int c, int n) {
	if (r != rows || c != cols || n != mats) {
		Release();
		return Init(r, c, n);
	}
	return true;
}
#if 0
bool BatchMatrix::SafeAdd(const BatchMatrix& right) {
	if (mats != right.mats) return false;
	bool ret = true;
	bool copied = false;

	if (cols == right.cols && rows == right.rows) {
		for (int i = 0; i < mats; i++) {
			Matrix* dest = elements[i];
			Matrix* src = right.elements[i];
			*dest += *(src);

		}
		return true;
	}

	int max_cols = max(cols, right.cols);
	int max_rows = max(rows, right.rows);


	BatchMatrix updated(max_rows, max_cols, mats);
	for (int i = 0; i < mats; i++) {
		Matrix* dest = updated.elements[i];
		Matrix* src1 = elements[i];
		Matrix* src2 = right.elements[i];
		if (!dest->SafeAdd(*src1, *src2)) {
			cerr << "Error: Safe add failed ." << endl;
			return false ;
		}
	}
	Rob(updated); 
	return true;
}
#endif