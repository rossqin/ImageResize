#include "StdAfx.h"
#include "image.h"

int main(int argc, char* argv[]) {
	
	if (argc < 5) return -1;

	const char* src = argv[1];
	int w = atoi(argv[2]);
	int h = atoi(argv[3]);
	float center_factor = 0.5;
	
	char basename[MAX_PATH];
	char outfile[MAX_PATH];
	_splitpath_s(src, NULL, 0, NULL, 0, basename, MAX_PATH, NULL, 0);
	bool fast =  0 == strcmp(argv[4], "--fast");

	if(fast)
		sprintf_s(outfile, MAX_PATH, "testing/%s_%dx%d.fast-rz.bmp", basename, w, h);
	else {
		center_factor = atof(argv[4]);
		sprintf_s(outfile, MAX_PATH, "testing/%s_%dx%d_%.2f.dilated-rz.bmp", basename, w, h, center_factor);
	}
	Image image;
	if (image.Load(src)) {

		image.ResizeTo(h, w, fast, center_factor);
		image.Save(outfile);
		cout << "saved to file `" << outfile << "`! " << endl;
	}
	else
		cerr << "failed to open file `" << src << "`! " << endl;
	return 0;
}