/*
Each block cannot have more than 512/1024 threads in total (Compute Capability 1.x or
2.x and later respectively)
The maximum dimensions of each block are limited to [512,512,64]/[1024,1024,64]
(Compute 1.x/2.x or later)
Each block cannot consume more than 8k/16k/32k/64k/32k/64k/32k/64k/32k/64k
registers total (Compute 1.0,1.1/1.2,1.3/2.x-
/3.0/3.2/3.5-5.2/5.3/6-6.1/6.2/7.0)
Each block cannot consume more than 16kb/48kb/96kb of shared memory
(Compute 1.x/2.x-6.2/7.0)

For GTX 106:

Total amount of shared memory per block:       49152 bytes
Total number of registers available per block: 65536
Maximum number of threads per multiprocessor:  2048
(10) Multiprocessors, (128) CUDA Cores/MP:     1280 CUDA Cores

Let's use 1280 at most
*/
#include "StdAfx.h"
#include "image.h"
#include "utils.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" 

__device__ const float PIXEL_NORM_FACTOR = 1.0 / 255.0;
__global__  void img_post_read_kernel(const byte* src, float* fill, int h, int w, int c) {
	//int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("%d of %d blocks, %d of %d threads\n", blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);

	//Now this is one of the 1280 threads(10 x 128),
	// and out mission is to fill the mat `fill`
	int stride = h * w;
	int channel_offset = 0;
	for (int i = 0; i < c; i++, channel_offset += stride) {
		for (int y = blockIdx.x; y < h; y += gridDim.x) {
			int offset = w * y;
			for (int x = threadIdx.x; x < w; x += blockDim.x) {
				int dest_index = channel_offset + x + offset;
				int src_index = i + c * (x + offset);
				fill[dest_index] = src[src_index] * PIXEL_NORM_FACTOR;

			}
		}
	}
}
bool Image::Load(const char * filename, int c) {
	int w = 0, h = 0;
	//
	stbi_uc* io_buffer = NULL;
	size_t size = 0;
	unsigned int start_t = GetTickCount();
	ifstream file(filename, ios::binary);
	if (file.is_open()) {
		file.seekg(0, ios_base::end);
		size = file.tellg();
		if (file.bad())
			cerr << "bad status in file \n";
		//	cout << "File size:" << size << " bytes.\n";
		if (size < 128) {
			cerr << "bad file size.\n";
			return false;
		}
		io_buffer = new stbi_uc[size];
		file.seekg(0, ios_base::beg);
		file.read(reinterpret_cast<char*>(io_buffer), size);
		file.close();
	}
	//stbi_load_from_memory

	unsigned char *buff = stbi_load_from_memory(io_buffer, size, &w, &h, &c, c);
	delete[]io_buffer;
	if (!buff) {
		cerr << "Cannot load image `" << filename << "`\nSTB Reason: " << stbi_failure_reason() << endl;

		return false;
	}
	size_t image_size = w * h * c;

	//unsigned int elapsed = GetTickCount() - start_t;
	//cout << "** std_image_lib takes " << elapsed << " ms to load the image." << endl;
	byte *data_gpu = NULL;
	cudaError_t err = cudaMalloc(&data_gpu, image_size);
	if (cudaSuccess != err) {
		cerr << "Error: Load image `" << filename << "` failed! Allocating " << channels.MemElments()
			<< " bytes of memory ret " << err << ".\n";
		free(buff);
		return false;
	}
	cudaMemcpy(data_gpu, buff, image_size, cudaMemcpyHostToDevice);
	free(buff);
	if (channels.inited) channels.Release();

	if (!channels.Init(h, w, c)) return false;


	img_post_read_kernel << < GPUGridSize(h), GPUBlockSize(w) >> > (data_gpu, channels.GetMem(), h, w, c);

	err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		cerr << "Error: Load image `" << filename << "` failed! cudaDeviceSynchronize ret " << err << ".\n";

	}
	cudaFree(data_gpu);
	return (cudaSuccess == err);
}

/*
x < 1 : 1 + (a+3) x *x + (a+2)* x * x * x
x < 2 : -4a+8ax-5ax*x+a x*x*x
*/
__device__ static float cubic_interpolation(float x) {
	if (0.0 == x) return 1.0;
	if (x < 0.0) x = 0.0 - x;
	if (1.0 == x || 2.0 <= x) return 0.0;
	float x2 = x * x;
	float x3 = x2 * x;
	if (x < 1.0)
		return 1.0 - 2.5 * x2 + 1.5f * x3;
	return  2.0f - 4.0 * x + 2.5 * x2 - 0.5 * x3;

}


__global__ static void im_resize_kernel_cubic(const float* orig, int orig_height, int orig_width,
	int channels, float* fill, int height, int width, float sh, float sw) {
	float C[4];
	float A[4];
	float T[4];
	int offsets[4], index;
	for (int c = 0, offset_cf = 0, offset_co = 0; c < channels;
		c++, offset_cf += height * width, offset_co += orig_height * orig_width) {
		for (int y = blockIdx.x; y < height; y += gridDim.x) {
			float f_ys = y * sh;
			int ys = (int)floor(f_ys);
			float u = f_ys - ys;
			ys--;
			for (int m = 0; m < 4; m++, ys++) {
				offsets[m] = (ys >= 0 && ys < orig_height) ? (offset_co + ys * orig_width) : -1;
			}
			C[0] = cubic_interpolation(1 + u);
			C[1] = cubic_interpolation(u);
			C[2] = cubic_interpolation(1 - u);
			C[3] = cubic_interpolation(2 - u);
			int offset_f = offset_cf + y * width;
			for (int x = threadIdx.x; x < width; x += blockDim.x) {
				float f_xs = x * sw;
				int xs = floor(f_xs);
				float v = f_xs - xs;
				xs--;

				A[0] = cubic_interpolation(1 + v);
				A[1] = cubic_interpolation(v);
				A[2] = cubic_interpolation(1 - v);
				A[3] = cubic_interpolation(2 - v);
				for (int n = 0; n < 4; n++, xs++) {
					T[n] = 0.0;
					if (xs >= 0 && xs < orig_width) {
						for (int k = 0; k < 4; k++) {
							if (offsets[k] >= 0) {
								index = offsets[k] + xs;
								T[n] += A[k] * orig[index];
							}
						}
					}
				}
				float temp = T[0] * C[0] + T[1] * C[1] + T[2] * C[2] + T[3] * C[3];
				if (temp < 0.0) temp = 0.0;
				else if (temp > 1.0) temp = 1.0;
				fill[offset_f + x] = temp;
			}
		}
	}
}
__device__ static const float dilated_factor = 0.5;

__global__ static void im_resize_kernel_dilated_conv(const float* orig, int orig_height, int orig_width,
	int channels, float* fill, int height, int width, float sh, float sw, float cfactor) {

	int step_h = (int)roundf(sh * dilated_factor);
	int step_w = (int)roundf(sw * dilated_factor);
	float corder_factor1 = 0.146447 * (1.0 - cfactor);
	float corder_factor2 = 0.103553 * (1.0 - cfactor);
	for (int c = 0, offset_cf = 0, offset_co = 0; c < channels;
		c++, offset_cf += height * width, offset_co += orig_height * orig_width) {
		for (int y = blockIdx.x; y < height; y += gridDim.x) {

			int src_y = (int)roundf(y * sh);
			if (src_y >= orig_height) src_y = orig_height - 1;
			int src_top = src_y - step_h, src_bottom = src_y + step_h;

			if (src_top < 0) src_top = src_y;
			if (src_bottom >= orig_height) src_bottom = src_y;

			int offset_f = offset_cf + y * width,
				offset_s0 = offset_co + src_top * orig_width,
				offset_s1 = offset_co + src_y * orig_width,
				offset_s2 = offset_co + src_bottom * orig_width;

			for (int x = threadIdx.x; x < width; x += blockDim.x) {
				int src_x = (int)roundf(x * sw);
				if (src_x >= orig_width) src_x = orig_width - 1;
				int src_left = src_x - step_w;
				if (src_left < 0) src_left = src_x;
				int src_right = src_x + step_w;
				if (src_right >= orig_width) src_right = src_x;
				fill[offset_f + x] = corder_factor2 * (orig[offset_s0 + src_left] + orig[offset_s0 + src_right] +
					orig[offset_s2 + src_left] + orig[offset_s2 + src_right])
					+ corder_factor1 * (orig[offset_s1 + src_left] + orig[offset_s1 + src_right] +
						orig[offset_s0 + src_x] + orig[offset_s2 + src_x])
					+ cfactor *   orig[offset_s1 + src_x];
			}
		}
	}
}
__global__ static void im_resize_kernel_bilinear(const float* orig, int orig_height, int orig_width,
	int channels, float* fill, int height, int width, float sh, float sw) {
	for (int c = 0, offset_cf = 0, offset_co = 0; c < channels;
		c++, offset_cf += height * width, offset_co += orig_height * orig_width) {
		for (int y = blockIdx.x; y < height; y += gridDim.x) {
			float t = y * sh;
			int src_y0 = (int)floorf(t), src_y1 = src_y0 + 1;
			float v = t - src_y0;
			if (src_y1 >= orig_height) src_y1 = orig_height - 1;
			int offset_f = offset_cf + y * width;
			int offset_s0 = offset_co + src_y0 * orig_width;
			int offset_s1 = offset_co + src_y1 * orig_width;
			for (int x = threadIdx.x; x < width; x += blockDim.x) {
				t = x * sw;
				int src_x0 = (int)floorf(t), src_x1 = src_x0 + 2;
				float u = t - src_x0;
				if (src_x1 >= orig_width) src_x1 = orig_width - 1;
				fill[offset_f + x] = (1.0 - u) * (1.0 - v) * orig[offset_s0 + src_x0]
					+ (1.0 - u) * v * orig[offset_s1 + src_x0]
					+ u * (1.0 - v) *  orig[offset_s0 + src_x1]
					+ u * v *  orig[offset_s1 + src_x1];
			}
		}
	}
}
//Nearest element method
__global__ static void im_resize_kernel_nearest(const float* orig, int orig_height, int orig_width,
	int channels, float* fill, int height, int width, float sh, float sw) {
	for (int c = 0, offset_cf = 0, offset_co = 0; c < channels;
		c++, offset_cf += height * width, offset_co += orig_height * orig_width) {
		for (int y = blockIdx.x; y < height; y += gridDim.x) {
			int src_y = (int)round(y * sh);
			if (src_y >= orig_height) src_y = orig_height - 1;
			int offset_f = offset_cf + y * width;
			int offset_s = offset_co + src_y * orig_width;
			for (int x = threadIdx.x; x < width; x += blockDim.x) {
				int src_x = (int)round(x * sw);
				if (src_x >= orig_width) src_x = orig_width - 1;
				fill[offset_f + x] = orig[offset_s + src_x];
			}
		}
	}
}

bool Image::ResizeTo(int height, int width, bool fast,float dilated_center_factor) {
	int layers = channels.Mats();
	if (0 == layers || 0 == height || 0 == width) return false;

	float sh = (float)(channels.Rows()) / (float)height;
	float sw = (float)(channels.Cols()) / (float)width;
	BatchMatrix new_channels(height, width, layers);
	int g = GPUGridSize(height);
	int b = GPUBlockSize(width);

	if (fast)
		im_resize_kernel_nearest << <g, b >> >(channels.mem_data, channels.rows, channels.cols,
			channels.mats, new_channels.mem_data, height, width, sh, sw);
	else  if (sh > 2.0 && sw > 2.0) { 
		im_resize_kernel_dilated_conv << <g, b >> >(channels.mem_data, channels.rows, channels.cols,
			channels.mats, new_channels.mem_data, height, width, sh, sw, dilated_center_factor);
	}
	else if (sh < 0.5 && sw < 0.5) {
		im_resize_kernel_cubic << <g, b >> >(channels.mem_data, channels.rows, channels.cols,
			channels.mats, new_channels.mem_data, height, width, sh, sw);
	}
	else {
		im_resize_kernel_bilinear << <g, b >> >(channels.mem_data, channels.rows, channels.cols,
			channels.mats, new_channels.mem_data, height, width, sh, sw);
	}




	channels = new_channels;
	return true;
}

__global__ static void im_compose_kernel(byte* fill, int c, int elements,
	int height, int width, float* src, int i) {
	for (int y = blockIdx.x; y < height; y += gridDim.x) {
		int offset = width * y;
		for (int x = threadIdx.x; x < width; x += blockDim.x) {
			int src_index = x + offset;
			int dest_index = i + c * src_index;
			fill[dest_index] = (byte)(src[src_index] * 255.0);

		}
	}
}
bool Image::Save(const char* filename, int quality) {
	int layers = channels.Mats();
	if (0 == layers) return false;


	int width = channels.Cols();
	int height = channels.Rows();

	unsigned char *dat_buf = NULL;
	unsigned int elements = width * height;
	cudaError_t err = cudaMalloc(&dat_buf, channels.MemElments());
	if (err != cudaSuccess) {
		cerr << "Failed to write image `" << filename << "` due to mem alloc fail.\n";
		return false;
	}

	for (int c = 0; c < layers; c++) {
		Matrix* m = channels.At(c);
		im_compose_kernel <<<  GPUGridSize(height), GPUBlockSize(width) >>>(dat_buf, layers, elements, height, width, m->elements, c);
		err = cudaDeviceSynchronize();
	}
	int success = 0;
	if (cudaSuccess != err) {
		cudaFree(dat_buf);
		return false;
	}
	byte* cpu_data = new byte[channels.MemElments()];
	err = cudaMemcpy(cpu_data, dat_buf, channels.MemElments(), cudaMemcpyDeviceToHost);
	char ext[100];
	_splitpath_s(filename, NULL, 0, NULL, 0, NULL, 0, ext, 100);
	if (0 == strcmpi(ext, ".bmp"))
		success = stbi_write_bmp(filename, width, height, layers, cpu_data);
	else if (0 == strcmpi(ext, ".jpg"))
		success = stbi_write_jpg(filename, width, height, layers, cpu_data, quality);
	else
		success = stbi_write_png(filename, width, height, layers, cpu_data, width * layers);
	delete[]cpu_data;
	cudaFree(dat_buf);
	if (!success) {
		cerr << "Failed to write image `" << filename << "`\n";
		return false;
	}
	//cout << "File `" << s << "` written successfully!\n";
	return true;
}
__device__ inline float three_way_max(float a, float b, float c) {
	return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c);
}

__device__ inline float three_way_min(float a, float b, float c) {
	return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
}
/*

1: max=max(R,G,B)
2: min=min(R,G,B)
3: if R = max, H = (G-B)/(max-min)
4: if G = max, H = 2 + (B-R)/(max-min)
5: if B = max, H = 4 + (R-G)/(max-min)
6:
7: H = H * 60
8: if H < 0, H = H + 360
9:
10: V=max(R,G,B)
11: S=(max-min)/max
*/
__global__ static void img_rgb2hsv_kernel(float* data, int height, int width,
	float hue, float sat, float val) {
	float r, g, b;
	float h, s, v;


	int offset, r_index, g_index, b_index;
	int channel_elements = height * width;
	for (int y = blockIdx.x; y < height; y += gridDim.x) {
		offset = width * y;
		for (int x = threadIdx.x; x < width; x += blockDim.x) {
			r_index = offset + x;
			g_index = r_index + channel_elements;
			b_index = g_index + channel_elements;
			r = data[r_index];
			g = data[g_index];
			b = data[b_index];
			float max_ = three_way_max(r, g, b);
			float min_ = three_way_min(r, g, b);
			float delta = max_ - min_; // what if delta == 0;
			v = max_; // v=max
			if (0.0f == delta)
				h = 0.0f;
			else if (max_ == r) {
				h = (g - b) / delta;
				if (h < 0) h += 6.0;

			}
			else if (max_ == g) {
				h = 2 + (b - r) / delta;
			}
			else {
				h = 4 + (r - g) / delta;
			}

			if (0.0 == max_)
				s = 0.0;
			else
				s = 1 - min_ / max_;

			h += hue;

			if (h > 6.0f) h -= 6.0f;
			else if (h < 0.0f) h += 6.0f;

			data[r_index] = h;
			data[g_index] = s * sat;
			data[b_index] = v * val;
		}
	}
}
bool Image::RGB2HSV(float hue, float sat, float val) {
	if (channels.Mats() != 3) return false;
	hue *= 6.0f;
	img_rgb2hsv_kernel << <GPUGridSize(channels.rows), GPUBlockSize(channels.cols) >> >(channels.mem_data, channels.rows, channels.cols,
		hue, sat, val);
	return cudaSuccess == cudaDeviceSynchronize();

}
__device__ static inline float constrain(float x) {
	if (x < 0.0) return 0.0;
	if (x > 1.0) return 1.0;
	return x;
}
__global__ static void img_hsv2rgb_kernel(float* data, int height, int width) {
	float r, g, b;
	float h, s, v;
	float f, p, q, t;
	int offset, r_index, g_index, b_index;
	int channel_elements = height * width;
	for (int y = blockIdx.x; y < height; y += gridDim.x) {
		offset = width * y;
		for (int x = threadIdx.x; x < width; x += blockDim.x) {
			r_index = offset + x;
			g_index = r_index + channel_elements;
			b_index = g_index + channel_elements;
			h = data[r_index];
			s = data[g_index];
			v = data[b_index];
			int index = floor(h);
			f = h - index;
			p = v * (1 - s);
			q = v *(1 - s * f);
			t = v * (1 - s * (1 - f));
			switch (index) {
			case 0:
				r = v; g = t; b = p;
				break;
			case 1:
				r = q; g = v; b = p;
				break;
			case 2:
				r = p; g = v; b = t;
				break;
			case 3:
				r = p; g = q; b = v;
				break;
			case 4:
				r = t; g = p; b = v;
				break;
			case 5:
				r = v; g = p; b = q;
				break;
			default:
				break;
			}

			data[r_index] = constrain(r);
			data[g_index] = constrain(g);
			data[b_index] = constrain(b);
		}
	}
}
bool Image::HSV2RGB() {
	if (channels.Mats() != 3) return false;
	int height = channels.Rows();
	int width = channels.Cols();
	img_hsv2rgb_kernel << <GPUGridSize(height), GPUBlockSize(width) >> >(channels.GetMem(), height, width);
	return cudaSuccess == cudaDeviceSynchronize();

}
__global__ static void img_scale_kernel(float* data, int height, int width, float scale) {
	int offset, pixel_idx;
	//int channel_elements = height * width;
	for (int y = blockIdx.x; y < height; y += gridDim.x) {
		offset = width * y;
		for (int x = threadIdx.x; x < width; x += blockDim.x) {
			pixel_idx = offset + x;
			data[pixel_idx] *= scale;
			if (data[pixel_idx] > 1.0) data[pixel_idx] -= 1.0;
			else if (data[pixel_idx] < 0.0) data[pixel_idx] += 1.0;
		}
	}
}
bool Image::Distort(float hue, float sat, float val) {
	int height = channels.Rows();
	int width = channels.Cols();
	float* data = channels.GetMem();
	int channel_elements = height * width;
	if (channels.Mats() == 3) {
		if (!RGB2HSV(hue, sat, val)) return false;
		return HSV2RGB();
	}

	img_scale_kernel << <GPUGridSize(height), GPUBlockSize(width) >> >(data, height, width, val);
	return cudaSuccess == cudaDeviceSynchronize();


}
__global__ static void img_rotate_kernel(float* data, int height, int width, RotateType rt, float* transition) {

	int src_index, dest_index;
	int offset, offset1;
	int max_width = width;
	int max_height = height;
	switch (rt) {
	case HorizFlip:
		max_width >>= 1;
		break;
	case VertiFlip:
		max_height >>= 1;
		break;
	default:
		break;

	}
	float temp;
	for (int y = blockIdx.x; y < max_height; y += gridDim.x) {
		offset = y * width;
		if (VertiFlip == rt) offset1 = (height - 1 - y) * width;
		for (int x = threadIdx.x; x < max_width; x += blockDim.x) {
			src_index = offset + x;
			if (HorizFlip == rt) {
				dest_index = offset + (width - 1 - x);
				temp = data[dest_index];
				data[dest_index] = data[src_index];
				data[src_index] = temp;
			}
			else if (VertiFlip == rt) {
				dest_index = offset1 + x;
				temp = data[dest_index];
				data[dest_index] = data[src_index];
				data[src_index] = temp;
			}
			else if (ToLeft == rt) {
				dest_index = (width - 1 - x) * height + y;
				transition[dest_index] = data[src_index];

			}
			else if (ToRight == rt) { //
				dest_index = x * height + (height - 1 - y);
				transition[dest_index] = data[src_index];
			}
			else { // Rotate 180
				;
			}
		}
	}
}
bool Image::Rotate(RotateType rt) {
	float* transition = NULL;
	int height = channels.Rows();
	int width = channels.Cols();
	float* data = channels.mem_data;
	size_t bytes = height * width * sizeof(float);
	cudaError_t err;
	if (ToLeft == rt || ToRight == rt) {
		err = cudaMalloc(&transition, bytes);
		if (cudaSuccess != err) return false;
		int temp = channels.cols;
		channels.cols = channels.rows;
		channels.rows = temp;
	}

	for (int c = 0; c < channels.mats; c++) {
		if (rt == Rotate180) {
			channels.At(c)->Rot180();
		}
		else {

			img_rotate_kernel << <GPUGridSize(height), GPUBlockSize(width) >> > (data, height, width, rt, transition);

			cudaDeviceSynchronize();
			if (ToLeft == rt || ToRight == rt) {

				cudaMemcpy(data, transition, bytes, cudaMemcpyDeviceToDevice);
			}
			data += height * width;
		}
	}

	if (transition)
		cudaFree(transition);
	return true;
}

bool Image::Gray(bool rgb /* = true */) {
	if (3 != channels.mats || 0 == channels.cols || 0 == channels.rows) {
		cerr << "Nothing to do with Image::Gray() \n";
		return false;
	}
	float f[3];
	if (rgb) {
		f[0] = 0.299; f[1] = 0.587, f[2] = 0.114;
	}
	else {
		f[0] = 0.2126; f[1] = 0.7152, f[2] = 0.0722;
	}
	//Matrix filter(1, 3, f);
	return false;//Convolve(filter);

}