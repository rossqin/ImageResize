#pragma once
class Image;
class BatchMatrix;
class Matrix {
protected:
	int cols;
	int rows;
	float* elements;
	static cudaStream_t stream;
	static cudaStream_t cudaStream();
	friend class Image;
	friend class BatchMatrix;
	friend Matrix operator*(const Matrix& A, const Matrix& B);
	friend Matrix operator+(const Matrix& A, const Matrix& B);
	friend Matrix operator-(const Matrix& A, const Matrix& B);
	friend ostream& operator<<(ostream& output, const Matrix& m);
	friend Matrix operator*(const Matrix& A, float val);
	friend Matrix operator+(const Matrix& A, float val);
	friend Matrix operator-(const Matrix& A, float val);
	friend  float sum_of_inner_product(const Matrix& A, const Matrix& B); 

public :  
	Matrix(int r,int c, float* vals = NULL);
	Matrix(int r, int c, float val); 
	Matrix(const Matrix& m); 
	Matrix(const char* filename);
	void Randomize();
	virtual ~Matrix();
	
	const Matrix& operator=(const Matrix& m);
	const Matrix& operator=(float val);
 

	Matrix& operator+=(const Matrix& B);
	Matrix& operator-=(const Matrix& B);
	Matrix& operator+=(float val);
	Matrix& operator*=(float val); 
	
	
	bool Correlate(Matrix& result,const Matrix& filter,int stride = 1, int padding = 0);
	bool AddCorrelate(Matrix* A, Matrix* B, int stride, int padding, bool forwarding = true);
	void Rob(Matrix& poor);

	inline int Cols() const { return cols; }
	inline int Rows() const { return rows; }
	inline int Elements() const { return cols * rows ; }
	inline int Bytes() const { return cols * rows * sizeof(float); }

	void SetElement(int row, int col, float val);

	float Mean() const;
	float Sum() const;
	float Variance(float mean) const;
	bool Constrain(float min_, float max_);
	bool Normalize(float mean, float variance, bool is_pixel = false);

	// for debugging
	void Print(int max_rows = 0, int max_cols = 0);

	bool LinearTranform(float a, float b); // x = x * a+ b
	bool Rot180();

	bool SafeAdd(const Matrix& A, const Matrix& B);

	bool UpSample(const Matrix& B, float stride_r, float stride_c, float scale);
	bool DownSample(const Matrix& B, float stride_r, float stride_c, float scale);


};
typedef Matrix* LPMatrix;
Matrix operator*(const Matrix& A, const Matrix& B);
Matrix operator+(const Matrix& A, const Matrix& B);
Matrix operator-(const Matrix& A, const Matrix& B);

Matrix operator*(const Matrix& A, float val);
Matrix operator+(const Matrix& A,float val);
Matrix operator-(const Matrix& A, float val);
ostream& operator<<(ostream& output, const Matrix& m);
float sum_of_inner_product(const Matrix& A, const Matrix& B);