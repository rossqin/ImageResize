#pragma once
enum ACTIVATION {
	LOGISTIC, LINEAR, LEAKY, RELU, HARDTAN, LHTAN, RELIE, RAMP, TANH, PLSE, ELU, LOGGY, STAIR
};

unsigned int random_gen();
float random_float();
float rand_uniform_strong(float min_, float max_);
const char* get_time_str();
const char* make_path(const char* dir, const char* base, const char* ext);
float get_next_float(const char*& str);
int get_next_int(const char*& str);
float* make_float_vector(int n);
float rand_scale(float s); 
float square_sum_array(float *a, int n);
float* new_gpu_array(unsigned int elements, float fill_val);
bool is_suffix(const char* filename, const char* ext);
