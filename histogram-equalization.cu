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

void histogram_equalization_gpu(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist_in[i++];
    }
    d = img_size - min;
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
        
        
    }
    
    /* Get the result image */
    for(i = 0; i < img_size; i ++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
        
    }
}

void create_histogram_gpu(int *hist, unsigned char* img_in, int img_size, int nbr_bin, 
                        unsigned char *img_out)
{
    int *hist_d;
    unsigned char *img_in_d;
    int *img_size_d;

    cudaMalloc((void**)&hist_d, sizeof(int)*256);
    cudaMalloc((void**)&img_in_d, sizeof(unsigned char) * img_size);
    cudaMalloc((void**)&img_size_d, sizeof(int));

    cudaMemcpy(img_in_d, img_in, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_size_d, &img_size, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(hist_d, 0, sizeof(int) * 256);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount;

    histogram_gpu<<<blocks * 2 ,nbr_bin, nbr_bin * sizeof(int)>>>(hist_d, img_in_d, img_size_d);

    cudaMemcpy(hist, hist_d, sizeof(int)*256, cudaMemcpyDeviceToHost);

    cudaFree(hist_d);
    cudaFree(img_in_d);
    cudaFree(img_size_d);

    histogram_equalization_gpu(img_out, img_in, hist, img_size, nbr_bin);
}







