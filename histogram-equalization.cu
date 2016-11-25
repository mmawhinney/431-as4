#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "histogram-equalization.h"

__global__ void histogram_gpu(int * hist_out, unsigned char * img_in, int *img_size){
    extern __shared__ int temp[];

    temp[threadIdx.x] = 0;

    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while(i < *img_size) {
        atomicAdd(&(temp[img_in[i]]), 1);
        i += offset;
    }

    __syncthreads();

    atomicAdd(&(hist_out[threadIdx.x]), temp[threadIdx.x]);
}

__global__ void histogram_equalization_gpu(unsigned char *img_out, unsigned char *img_in, int *hist,
                                            int *img_size, int *d, int *min, int *cdf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int lut[256];

    int lut_val = (int)(((float) cdf[threadIdx.x] - *min) * 255 / *d + 0.5);
    if(lut_val < 0) {
        lut[threadIdx.x] = 0;
    } else {
        lut[threadIdx.x] = lut_val;
    }

    __syncthreads();

    int offset = blockDim.x * gridDim.x;
    while(i < *img_size) {
        if(lut[img_in[i]] > 255) {
            img_out[i] = 255;
        } else {
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
        i += offset;
    }
}

void create_histogram_gpu(int *hist, unsigned char* img_in, int img_size, int nbr_bin, 
                        unsigned char *img_out)
{
    int *hist_d;
    unsigned char *img_in_d;
    unsigned char *img_out_d;
    int *img_size_d;

    cudaMalloc((void**)&hist_d, sizeof(int)*256);
    cudaMalloc((void**)&img_in_d, sizeof(unsigned char) * img_size);
    cudaMalloc((void**)&img_out_d, sizeof(unsigned char) * img_size);
    cudaMalloc((void**)&img_size_d, sizeof(int));

    cudaMemset(hist_d, 0, sizeof(int) * 256);
    cudaMemcpy(img_in_d, img_in, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_out_d, img_out, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_size_d, &img_size, sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount;

    histogram_gpu<<<blocks * 2 ,nbr_bin, nbr_bin * sizeof(int)>>>(hist_d, img_in_d, img_size_d);

    cudaMemcpy(hist, hist_d, sizeof(int)*256, cudaMemcpyDeviceToHost);

    int *size_minus_min_d;
    int *min_d;
    int *cdf_d;

    int size_minus_min;
    int min = 0;
    int i = 0;
    while(min == 0) {
        min = hist[i++];
    }
    size_minus_min = img_size - min;

    // calculate the cdf outside the kernel function
    // then we can get the summed value regardless of the order our threads execute
    int cdf[nbr_bin];
    for(int i = 0; i < nbr_bin; i++) {
        if(i > 0) {
            cdf[i] = cdf[i-1] + hist[i];
        } else {
            cdf[i] = hist[i];
        }
    }

    cudaMalloc((void**)&size_minus_min_d, sizeof(int));
    cudaMalloc((void**)&min_d, sizeof(int));
    cudaMalloc((void**)&cdf_d, sizeof(int) * nbr_bin);

    cudaMemcpy(size_minus_min_d, &size_minus_min, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(min_d, &min, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cdf_d, &cdf, sizeof(int) * nbr_bin, cudaMemcpyHostToDevice);

    histogram_equalization_gpu<<<blocks * 2, nbr_bin>>>(img_out_d, img_in_d, hist_d, img_size_d, size_minus_min_d, min_d, cdf_d);

    cudaMemcpy(img_out, img_out_d, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);

    cudaFree(hist_d);
    cudaFree(img_in_d);
    cudaFree(img_out_d);
    cudaFree(img_size_d);
    cudaFree(size_minus_min_d);
    cudaFree(min_d);
    cudaFree(cdf_d);
}







