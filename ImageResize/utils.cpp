#include "StdAfx.h"
//#include "activator.h"
#include "utils.h"
static int gpu_processors = 10;
static int gpu_kernels_in_mp = 128;
int gpu_device_cnt = 0;

static cudnnHandle_t cudnnHandle;

int GPUBlockSize(int b) {
	return (b > 0 && b < gpu_kernels_in_mp) ? b : gpu_kernels_in_mp;
}

int GPUGridSize(int g) {
	return (g > 0 && g < gpu_processors) ? g : gpu_processors;
}
void GPUGetGridBlock(int threads, int& g, int& b) {
	if (threads <= gpu_kernels_in_mp) {
		g = 1;
		b = threads; 
	}
	else { 
		g = (threads + gpu_kernels_in_mp - 1) / gpu_kernels_in_mp;
		if (g > gpu_processors) g = gpu_processors;
		b = gpu_kernels_in_mp;
	}
	
}
string& remove_ext(string& name) {
	size_t pos = name.find_last_of('.');
	if (pos != string::npos)
		name.erase(pos, name.length() - pos);
	return name;
}
string& trim(string& s) {
	if (s.empty())
		return s;
	s.erase(0, s.find_first_not_of(' '));
	s.erase(s.find_last_not_of(' ') + 1);
	return s;
}
bool atob(const char* str) {
	if (NULL == str) return false;
	return (0 != strcmp(str, "0") && 0 != _strcmpi(str, "false"));
}
 
void DisplayGPUs() {
	cudaDeviceProp devProp;
	for (int i = 0; i < gpu_device_cnt; i++) {
		cudaError_t err = cudaGetDeviceProperties(&devProp, i);
		if (cudaSuccess == err) {
			if (0 == i) gpu_processors = devProp.multiProcessorCount;
			cout << "** GPU NO. " << i << ": " << devProp.name << endl;
			cout << "   Memory Size: " << (devProp.totalGlobalMem >> 20) << " MB." << endl;
			cout << "   Processors Count: " << devProp.multiProcessorCount << endl;
			cout << "   Shared Memory Per Block: " << (devProp.sharedMemPerBlock >> 10) << " KB" << endl;
			cout << "   Max Threads Per Block: " << devProp.maxThreadsPerBlock << endl;
			cout << "   Max Threads Per MultiProcessor: " << devProp.maxThreadsPerMultiProcessor << endl;
			cout << "   Max Warps Per MultiProcessor: " << (devProp.maxThreadsPerMultiProcessor >> 5) << endl;

		}

	}
	cout << endl << endl;
}


int count_fields(const char* line) {
	int count = 0;
	size_t length = strlen(line);
	for (size_t n = 0; n < length; n++) {
		if (line[n] == ',') ++count;
	}
	if (line[length - 1] != ',')
		++count;
	return count;
};
float *parse_fields(const char* line, unsigned int bytes, int n) {
	if (n <= 0) return NULL;
	float *field = new float[n];
	char buffer[100];
	memset(field, 0, sizeof(float) * n);

	unsigned int start = 0;
	int counter = 0;

	for (unsigned int i = 0; i <= bytes; i++) {
		if (i == bytes || line[i] == ',' || line[i] == '\n' || line[i] == '\r') {
			if (i == start)
				field[counter++] = 0;
			else {
				memcpy(buffer, line + start, i - start);
				buffer[i - start] = 0;
				field[counter++] = atof(buffer);
			}
			start = i + 1;
		}
		if (counter >= n) break;
	}

	return field;
}


ACTIVATION get_activation(const char *s) {
	if (strcmp(s, "logistic") == 0) return LOGISTIC;
	if (strcmp(s, "loggy") == 0) return LOGGY;
	if (strcmp(s, "relu") == 0) return RELU;
	if (strcmp(s, "elu") == 0) return ELU;
	if (strcmp(s, "relie") == 0) return RELIE;
	if (strcmp(s, "plse") == 0) return PLSE;
	if (strcmp(s, "hardtan") == 0) return HARDTAN;
	if (strcmp(s, "lhtan") == 0) return LHTAN;
	if (strcmp(s, "linear") == 0) return LINEAR;
	if (strcmp(s, "ramp") == 0) return RAMP;
	if (strcmp(s, "leaky") == 0) return LEAKY;
	if (strcmp(s, "tanh") == 0) return TANH;
	if (strcmp(s, "stair") == 0) return STAIR;
	cerr << "Couldn't find activation function "<<s<<", going with ReLU "<< endl;
	return RELU;
}


unsigned int random_gen() {
	unsigned int rnd = 0;
#ifdef WIN32
	rand_s(&rnd);
#else
	rnd = rand();
#endif
	return rnd;
}

float random_float() {
#ifdef WIN32
	return ((float)random_gen() / (float)UINT_MAX);
#else
	return ((float)random_gen() / (float)RAND_MAX);
#endif
}

float rand_uniform_strong(float min_, float max_) {
	if (max_ < min_) {
		float swap = min_;
		min_ = max_;
		max_ = swap;
	}
	return (random_float() * (max_ - min_)) + min_;
}
static char time_str_buf[32];
const char* get_time_str() {
	time_t n = time(NULL);
	tm ti;
	localtime_s(&ti, &n);
	sprintf_s(time_str_buf, 32, "%04d%02d%02d%02d%02d%02d",
		ti.tm_year + 1900, ti.tm_mon + 1, ti.tm_mday,
		ti.tm_hour, ti.tm_min, ti.tm_sec);
	return time_str_buf;
}
static char path[300];
const char* make_path(const char* dir, const char* base, const char* ext) {
	sprintf_s(path, 300, "%s\\%s%s", dir, base, ext);
	return path;
}
float get_next_float(const char*& str) {
	const char* p = str;
	while (*p != ',' && *p != ' '&& *p != '\t' && *p != 0)
		p++;
	float r = (float)atof(str);
	if (0 == *p)
		str = p;
	else {
		str = p + 1;
		while (*str == ',' || *str == ' ' || *str == '\t')
			str++;
	}
	return r;
}
int get_next_int(const char*& str) {
	const char* p = str;
	while (*p != ',' && *p != ' '&& *p != '\t' && *p != 0)
		p++;
	int r = atoi(str);
	if (0 == *p)
		str = p;
	else {
		str = p + 1;
		while (*str == ',' || *str == ' ' || *str == '\t')
			str++;
	}
	return r;
}
float* make_float_vector(int n) {
	float* ret = new float[n];
	memset(ret, 0, sizeof(float) * n);
	return ret;
}
float rand_scale(float s) {
	float scale = rand_uniform_strong(1, s);
	if (random_gen() % 2) return scale;
	return 1.0 / scale;
}
 

float square_sum_array(float *a, int n) {
 
	float sum = 0;
	for (int i = 0; i < n; ++i) {
		sum += a[i] * a[i];
	} 
	return sum;
}
float* new_gpu_array(unsigned int elements, float fill_val) {
	float* ret = NULL;
	unsigned int bytes = elements * sizeof(float);
	if (0.0f == fill_val) {
		cudaError_t e = cudaMalloc(&ret, bytes);
		if (e != cudaSuccess) {
			cerr << "Error: cudaMalloc ret " << e << "in new_gpu_array!\n";
			return NULL;
		}
		e = cudaMemset(ret, 0, bytes);

	}
	else {
		cudaError_t e = cudaMallocManaged(&ret, bytes);
		if (e != cudaSuccess) {
			cerr << "Error: cudaMallocManaged ret " << e << "in new_gpu_array!\n";
			return NULL;
		}
		for (unsigned int n = 0; n < elements; n++)
			ret[n] = fill_val;
	}
	return ret;

}
bool is_suffix(const char* filename, const char* ext) {
	size_t l1 = strlen(filename);
	size_t l2 = strlen(ext);
	if (l1 < l2) return false;
	const char* s = filename + (l1 - l2);
	return 0 == strcmp(s, ext);
}