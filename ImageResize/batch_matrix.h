#pragma once
#include "matrix.h"
/*
A batch of matrixs for deepnet
*/
class Image;
class BatchMatrix {
protected:
	bool inited;
	LPMatrix* elements;
	int mats;
	int rows;
	int cols;
	float* mem_data; // in gpu
	
	size_t mem_bytes;
	size_t mem_elements;
	friend ofstream& operator <<(ofstream& ofile, const BatchMatrix & bm);
	friend class Image;
	void Rob(BatchMatrix& bm);
	 
 public:
	char name[MAX_PATH];
	BatchMatrix();
	BatchMatrix(const BatchMatrix& bm);
	BatchMatrix(int r, int c, int n);
	const BatchMatrix& operator=(const BatchMatrix& bm);
	const BatchMatrix& operator=(float val);
	virtual ~BatchMatrix();
	inline int Rows() const { return rows; }
	inline int Cols() const { return cols; }
	inline int Mats() const { return mats; } 
	inline size_t MemBytes() const { return mem_bytes; }
	inline size_t MemElments() const { return mem_elements; }
	inline float* GetMem() const { return mem_data; }
	// copy gpu data to memory, bytes = 0 means use default bytes;
	bool GetData(float* out, size_t bytes = 0);
	float Mean() const;
	bool Init(int r, int c, int n, float val = 0.0);
	Matrix* At(int i) const;
	bool Correlate(BatchMatrix& result, const BatchMatrix& filters, int stride, int padding,int mini_batch) const;
	
	//for conv backwards
	//bool InterCorrelate(BatchMatrix& filters, const BatchMatrix& inputs, int stride, int padding, int mini_batch) const;
	//bool UpsampleCorrelate(BatchMatrix& inputs, const BatchMatrix& filters, int stride, int padding, int mini_batch) const;
	void Release();
	bool SetElement(int i, const Matrix& m); 
	bool SafeAdd(const BatchMatrix& right);
	bool Add(const BatchMatrix& right);
	bool Add(float val);
	bool Mul(float val);
	bool MulAdd(float scale, float bias);
	bool MulAdd(float scale, const BatchMatrix& right);
	inline bool SameShape(const BatchMatrix& right) const { return (cols == right.cols) && (rows == right.rows) && (mats == right.mats); }
	bool UpSample(BatchMatrix& result, int stride_r , int stride_c, float scale = 1.0f) const;
	bool DownSample(BatchMatrix& result, int stride_r, int stride_c, float scale = 1.0f) const;
	bool Transpose();
	bool Rot180();
	bool ZeroMem();
	bool EnsureDim(int r, int c, int n);

};
ofstream& operator <<(ofstream& ofile, const BatchMatrix & bm);