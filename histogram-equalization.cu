#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "histogram-equalization.h"

void print_hist(int *hist) {
    for(int i = 0; i < 256; i++) {
        printf("hist[%d] = %d\n", i, hist[i]);
    }
}

__global__ void histogram_gpu(int * hist_out, unsigned char * img_in, int *img_size, int *nbr_bin){
    printf("executed...\n");
    int i;
    for ( i = 0; i < *nbr_bin; i ++){
        hist_out[i] = 0;
    }

    for ( i = 0; i < *img_size; i ++){
        hist_out[img_in[i]] ++;
    }
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
    int *nbr_bin_d;

    cudaMalloc((void**)&hist_d, sizeof(int)*256);
    cudaMalloc((void**)&img_in_d, sizeof(unsigned char) * img_size);
    cudaMalloc((void**)&img_size_d, sizeof(int));
    cudaMalloc((void**)&nbr_bin_d, sizeof(int));

    cudaMemcpy(hist_d, hist, sizeof(int)*256, cudaMemcpyHostToDevice);
    cudaMemcpy(img_in_d, img_in, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_size_d, &img_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(nbr_bin_d, &nbr_bin, sizeof(int), cudaMemcpyHostToDevice);

    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    histogram_gpu<<<1,1>>>(hist_d, img_in_d, img_size_d, nbr_bin_d);

    printf("%s\n", cudaGetErrorString(cudaGetLastError()));


    cudaMemcpy(hist, hist_d, sizeof(int)*256, cudaMemcpyDeviceToHost);

    print_hist(hist);

    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    cudaFree(hist_d);
    cudaFree(img_in_d);
    cudaFree(img_size_d);
    cudaFree(nbr_bin_d);

    printf("%s\n", cudaGetErrorString(cudaGetLastError()));


    histogram_equalization_gpu(img_out, img_in, hist, img_size, nbr_bin);
}







