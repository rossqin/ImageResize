#pragma once 
#include "batch_matrix.h"
 
#define ROTATE_TYPE_COUNT 6
enum RotateType{NotRotate, ToLeft, ToRight, HorizFlip, VertiFlip, Rotate180};
class Image {
protected: 
	
public :
	BatchMatrix channels;
	Image();
	Image(const char* filename);
	Image(int height, int width, int channel);
	Image(const Image& img);
	const Image& operator=(const Image& img);
	bool Load(const char* filename,int c = 3);
	bool Save(const char* filename,int quality = 100);
	bool ResizeTo(int height, int width, bool fast = true, float dilated_center_factor = 0.5);
	//float* LoadLabels(const char* filename, int& boxes);
	bool Distort(float hue, float sat, float val);
	bool RGB2HSV(float hue = 0.0f, float sat = 1.0f, float val = 1.0f);
	bool HSV2RGB();
	bool Crop(Image& result, int dx, int dy, int w, int h);
	bool Rotate(RotateType t);

	bool Convolve(const Matrix& filter);
	void Whiten();
	virtual ~Image() {};
	inline int GetHeight() const { return channels.Rows(); }
	inline int GetWidth() const { return channels.Cols(); }
	inline int GetChannels() const { return channels.Mats(); }
	inline BatchMatrix& GetData() { return channels; }
	bool Gray(bool rgb = true);
	bool Constrain();
};
