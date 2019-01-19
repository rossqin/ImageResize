#include "StdAfx.h"
#include "image.h"

Image::Image() {

}
Image::Image(const char *filename) {
	Load(filename);
}

Image::Image(int height, int width, int channel) {
	if (height > 0 && width > 0 && channel > 0) {
		channels.Init(height, width, channel);
	}
}


void Image::Whiten() {
	for (int i = 0; i < channels.Mats(); i++) {
		Matrix* m = channels.At(i);
		float mean = m->Mean();
		float variance = m->Variance(mean);
		//cout << "in Whiten: layer - " << i << ", mean :" << mean << ", variance: " << sqrt(variance) << endl;
		if (variance == 0)
			*m = 0.0;
		else { 
			m->Normalize(mean,variance,true);
		}
	}
}
bool Image::Crop(Image& result, int dx, int dy, int w, int h) {
	int height = channels.Rows();
	int width = channels.Cols();
	int layers = channels.Mats();
	if (dx < 0 || dx > width || dy < 0 || dy > height) return false;
	if (dx + w > width) w = width - dx;
	if (dy + h > height) h = height - dy;
	if (w <= 0 || h <= 0) return false;
	result.channels.Release();
	result.channels.Init(h, w, layers);

	size_t bytes = w * sizeof(float);

	for (int c = 0; c < layers; c++) {
		for (int y = 0; y < h; y++) {
			float *src_base = channels.mem_data + c * height * width + (y + dy) * width;
			float *dest = result.channels.mem_data + c * w * h + y * w;
			if (cudaSuccess != cudaMemcpy(dest, src_base + dx, bytes, cudaMemcpyDeviceToDevice))
				return false;
		}
	}
	return true;

}
bool Image::Convolve(const Matrix& filter) {
	int padding = filter.rows >> 1;
	BatchMatrix filters(filter.rows, filter.cols, channels.Mats());
	for(int i = 0 ; i < filters.Mats(); i++)
		filters.SetElement(i, filter);
	filters.Rot180();
	BatchMatrix bm;
	if (!channels.Correlate(bm, filters, 1, padding, 1))
		return false;
	channels = bm;
	
	return true;
}
Image::Image(const Image& img):channels(img.channels) { 
}
const Image& Image::operator=(const Image& img) {
	this->channels = img.channels;
	return *this;
}
bool Image::Constrain() {
	bool succ = false;
	for (int i = 0; i < channels.mats; i++) {
		Matrix* m = channels.At(i);
		succ = m->Constrain(0.0,1.0);
	}
	return succ;
}