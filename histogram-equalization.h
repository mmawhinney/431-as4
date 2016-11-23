#ifndef _HISTOGRAM_EQUALIZATION_H_
#define _HISTOGRAM_EQUALIZATION_H_

void create_histogram_gpu(int *hist, unsigned char *img_in, int img_size, int nbr_bin, unsigned char *img_out);


#endif 