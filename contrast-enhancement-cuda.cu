#include <cuda_runtime.h>
#include <stdio.h>
#include "histogram-equalization.h"
#include "hist-equ.h"

__global__ void rgb2hsl_convert(PPM_IMG *img_in, HSL_IMG *img_out, int *img_size) 
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    float L, H, S;
	
    while(i < *img_size) {
        float var_r = ( (float)img_in->img_r[i]/255 );//Convert RGB to [0,1]
        float var_g = ( (float)img_in->img_g[i]/255 );
        float var_b = ( (float)img_in->img_b[i]/255 );
        float var_min = (var_r < var_g) ? var_r : var_g;
        var_min = (var_min < var_b) ? var_min : var_b;   //min. value of RGB
        float var_max = (var_r > var_g) ? var_r : var_g;
        var_max = (var_max > var_b) ? var_max : var_b;   //max. value of RGB
        float del_max = var_max - var_min;              //Delta RGB value
        
        L = ( var_max + var_min ) / 2;
        if ( del_max == 0 ) {                   //This is a gray, no chroma...
            H = 0;         
            S = 0;    
        } else {                                    //Chromatic data...
            if ( L < 0.5 ) {
                S = del_max/(var_max+var_min);
            } else {
                S = del_max/(2-var_max-var_min );
            }

            float del_r = (((var_max-var_r)/6)+(del_max/2))/del_max;
            float del_g = (((var_max-var_g)/6)+(del_max/2))/del_max;
            float del_b = (((var_max-var_b)/6)+(del_max/2))/del_max;
            if( var_r == var_max ){
                H = del_b - del_g;  
            } else {       
                if( var_g == var_max ) {
                    H = (1.0/3.0) + del_r - del_b;
                } else{
                        H = (2.0/3.0) + del_g - del_r;
                }   
            }
        }
        
        if ( H < 0 ) {
            H += 1;
        }
        if ( H > 1 ) {
            H -= 1;
        }

        __syncthreads();
        
        img_out->h[i] = H;
        img_out->s[i] = S;
        img_out->l[i] = (unsigned char)(L*255);

        i+= offset;
    }
    
}

HSL_IMG rgb2hsl_gpu(PPM_IMG img_in) 
{
    int img_size = img_in.w * img_in.h;
    // int i;
    // float H, S, L;
    HSL_IMG img_out;// = (HSL_IMG *)malloc(sizeof(HSL_IMG));
    img_out.width  = img_in.w;
    img_out.height = img_in.h;
    img_out.h = (float *)malloc(img_size * sizeof(float));
    img_out.s = (float *)malloc(img_size * sizeof(float));
    img_out.l = (unsigned char *)malloc(img_size * sizeof(unsigned char));
	
    
	PPM_IMG *img_in_d;
	int ppm_size = sizeof(PPM_IMG);
    printf("ppm_size = %d\n", ppm_size);
	
	HSL_IMG *img_out_d;
	int hsl_size = sizeof(HSL_IMG);
    printf("hsl_size = %d\n", hsl_size);

    int *img_size_d;
	

    // printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMalloc((void**)&img_in_d, ppm_size);
    cudaMalloc((void**)&img_in.img_r, img_size * sizeof(unsigned char));
    cudaMalloc((void**)&img_in.img_g, img_size * sizeof(unsigned char));
    cudaMalloc((void**)&img_in.img_b, img_size * sizeof(unsigned char));
    // printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMalloc((void**)&img_out_d, hsl_size);
    // printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMalloc((void**)&img_out.h, img_size * sizeof(float));
    // printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMalloc((void**)&img_out.s, img_size * sizeof(float));
    // printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMalloc((void**)&img_out.l, img_size * sizeof(unsigned char));
    // printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMalloc((void**)&img_size_d, sizeof(int));


    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(img_in_d, &img_in, ppm_size, cudaMemcpyHostToDevice);
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(img_out_d, &img_out, hsl_size, cudaMemcpyHostToDevice);
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(img_size_d, &img_size, sizeof(int), cudaMemcpyHostToDevice);
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

	// int size = img_in.w * img_in.h;
	
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount;

    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

	rgb2hsl_convert<<<blocks * 2, 1024>>>(img_in_d, img_out_d, img_size_d);
	
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy(&img_out, img_out_d, hsl_size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(&img_out.l, img_out_d->l, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	printf("img_out height = %d\n", img_out.height);
    for(int i = 0; i < img_size; i++) {
        printf("testing...\n");
        printf("img_out.h[%d] = %f\n", i, img_out.h[i]);
    }

    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

	cudaFree(img_in_d);
    cudaFree(img_in.img_r);
    cudaFree(img_in.img_g);
    cudaFree(img_in.img_b);
	cudaFree(img_out_d);
    cudaFree(img_out.h);
    cudaFree(img_out.s);
    cudaFree(img_out.l);
    cudaFree(img_size_d);

    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    return img_out;
}