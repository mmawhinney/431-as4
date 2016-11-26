#ifndef _CONTRAST_ENHANCEMENT_CUDA_H_
#define _CONTRAST_ENHANCEMENT_CUDA_H_

HSL_IMG rgb2hsl_gpu(PPM_IMG img_in);
PPM_IMG hsl2rgb_gpu(HSL_IMG img_in);

YUV_IMG rgb2yuv_gpu(PPM_IMG img_in);
PPM_IMG yuv2rgb_gpu(YUV_IMG img_in);

#endif
