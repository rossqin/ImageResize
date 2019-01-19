#ifdef _MSC_VER
#pragma once 
#pragma  warning(disable:4819)
#endif

#ifndef _STD_AFX_H_
#define _STD_AFX_H_ 
 
#include <cstdlib>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <Windows.h>
#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC

#include <stdlib.h>

#include <crtdbg.h> 
#else
#include <stdlib.h>
#endif
#include <cudnn.h>

const float EPSILON = 0.00001f;
using namespace std;
int GPUBlockSize(int b  = 0);
int GPUGridSize(int g = 0);
void GPUGetGridBlock(int threads, int& g, int& b);
void DisplayGPUs();
string& remove_ext(string& name);
string& trim(string& s); 
bool atob(const char* str);
extern int gpu_device_cnt;
int count_fields(const char* line);
float *parse_fields(const char* line, unsigned int bytes, int n);

typedef const char* pconstchar;
//typedef unsigned char byte;
#endif
