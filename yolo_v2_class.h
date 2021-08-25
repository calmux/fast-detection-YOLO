
#pragma once
#ifdef YOLODLL_EXPORTS
#if defined(_MSC_VER)
#define YOLODLL_API __declspec(dllexport) 
#else
#define YOLODLL_API __attribute__((visibility("default")))
#endif
#else
#if defined(_MSC_VER)
#define YOLODLL_API __declspec(dllimport) 
#else
#define YOLODLL_API
#endif
#endif

struct bbox_t {
	unsigned int x, y, w, h;	// (x,y) - top-left corner, (w, h) - width & height of bounded box
	float prob;					// confidence - probability that the object was found correctly
	unsigned int obj_id;		// class of object - from range [0, classes-1]
	unsigned int track_id;		// tracking id for video (0 - untracked, 1 - inf - tracked object)
	unsigned int frames_counter;// counter of frames on which the object was detected
};

struct image_t {
	int h;						// height
	int w;						// width
	int c;						// number of chanels (3 - for RGB)
	float *data;				// pointer to the image data
};

#ifdef __cplusplus
#include <memory>
#include <vector>
#include <deque>
#include <algorithm>


#include <opencv2/opencv.hpp>			// C++
#include "opencv2/highgui/highgui_c.h"	// C
#include "opencv2/imgproc/imgproc_c.h"	// C


class Detector {
	std::shared_ptr<void> detector_gpu_ptr;
	std::deque<std::vector<bbox_t>> prev_bbox_vec_deque;
	const int cur_gpu_id;
public:
	float nms = .4;
	bool wait_stream;

	YOLODLL_API Detector(std::string cfg_filename, std::string weight_filename, int gpu_id = 0);
	YOLODLL_API ~Detector();

	YOLODLL_API std::vector<bbox_t> detect(std::string image_filename, float thresh = 0.2, bool use_mean = false);
	YOLODLL_API std::vector<bbox_t> detect(image_t img, float thresh = 0.2, bool use_mean = false);
	static YOLODLL_API image_t load_image(std::string image_filename);
	static YOLODLL_API void free_image(image_t m);
	YOLODLL_API int get_net_width() const;
	YOLODLL_API int get_net_height() const;
	YOLODLL_API int get_net_color_depth() const;