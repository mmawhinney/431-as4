#include <cuda_runtime.h>
#include <stdio.h>
#include "histogram-equalization.h"
#include "hist-equ.h"

__global__ void rgb2hsl_convert(PPM_IMG *img_in, HSL_IMG *img_out, int *img_size) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    float L = 0.0;
    float H = 0.0;
    float S = 0.0;
	
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
            if(S > 1) {
                printf("S too big...\n");
            }

            float del_r = (((var_max-var_r)/6)+(del_max/2))/del_max;
            float del_g = (((var_max-var_g)/6)+(del_max/2))/del_max;
            float del_b = (((var_max-var_b)/6)+(del_max/2))/del_max;
            if( var_r == var_max ) {
                H = del_b - del_g;  
            } else {       
                if( var_g == var_max ) {
                    H = (1.0/3.0) + del_r - del_b;
                } else {
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
    
    HSL_IMG img_out;
    img_out.width  = img_in.w;
    img_out.height = img_in.h;
    img_out.h = (float *)malloc(img_size * sizeof(float));
    img_out.s = (float *)malloc(img_size * sizeof(float));
    img_out.l = (unsigned char *)malloc(img_size * sizeof(unsigned char));
	

    unsigned char *l = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    float *h = (float *)malloc(img_size * sizeof(float));
    float *s = (float *)malloc(img_size * sizeof(float));

    unsigned char *img_r = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    unsigned char *img_g = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    unsigned char *img_b = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    
	PPM_IMG *img_in_d;
	HSL_IMG *img_out_d;
    int *img_size_d;
	
    cudaMalloc((void**)&img_out_d, sizeof(HSL_IMG));
    cudaMalloc((void**)&l, sizeof(unsigned char) * img_size);
    cudaMalloc((void**)&h, sizeof(float) * img_size);
    cudaMalloc((void**)&s, sizeof(float) * img_size);

    cudaMalloc((void**)&img_in_d, sizeof(PPM_IMG));
    cudaMalloc((void**)&img_r, sizeof(unsigned char) * img_size);
    cudaMalloc((void**)&img_g, sizeof(unsigned char) * img_size);
    cudaMalloc((void**)&img_b, sizeof(unsigned char) * img_size);


    cudaMemcpy(img_out_d, &img_out, sizeof(HSL_IMG), cudaMemcpyHostToDevice);
    cudaMemcpy(l, img_out.l, sizeof(unsigned char) *img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(h, img_out.h, sizeof(float) *img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(s, img_out.s, sizeof(float) *img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(&(img_out_d->l), &l, sizeof(l), cudaMemcpyHostToDevice);
    cudaMemcpy(&(img_out_d->h), &h, sizeof(h), cudaMemcpyHostToDevice);
    cudaMemcpy(&(img_out_d->s), &s, sizeof(s), cudaMemcpyHostToDevice);

    cudaMemcpy(img_in_d, &img_in, sizeof(PPM_IMG), cudaMemcpyHostToDevice);
    cudaMemcpy(img_r, img_in.img_r, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_g, img_in.img_g, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_b, img_in.img_b, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(&(img_in_d->img_r), &img_r, sizeof(img_r), cudaMemcpyHostToDevice);
    cudaMemcpy(&(img_in_d->img_g), &img_g, sizeof(img_g), cudaMemcpyHostToDevice);
    cudaMemcpy(&(img_in_d->img_b), &img_b, sizeof(img_b), cudaMemcpyHostToDevice);


    cudaMalloc((void**)&img_size_d, sizeof(int));
	cudaMemcpy(img_size_d, &img_size, sizeof(int), cudaMemcpyHostToDevice);
	
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount;

	rgb2hsl_convert<<<blocks * 2, 1024>>>(img_in_d, img_out_d, img_size_d);
	
    cudaMemcpy(img_out.l, l, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.h, h, sizeof(float) * img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.s, s, sizeof(float) * img_size, cudaMemcpyDeviceToHost);


	cudaFree(img_in_d);
    cudaFree(img_r);
    cudaFree(img_g);
    cudaFree(img_b);
	cudaFree(img_out_d);
    cudaFree(h);
    cudaFree(s);
    cudaFree(l);
    cudaFree(img_size_d);

    return img_out;
}