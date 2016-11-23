#include <cuda_runtime.h>
#include "histogram-equalization.h"
#include "hist-equ.h"

__global__ void rgb2hsl_convert(PPM_IMG *img_in, HSL_IMG *img_out) 
{
	int i = blockIdx.x;
    float L, H, S;
	
    float var_r = ( (float)img_in->img_r[i]/255 );//Convert RGB to [0,1]
    float var_g = ( (float)img_in->img_g[i]/255 );
    float var_b = ( (float)img_in->img_b[i]/255 );
    float var_min = (var_r < var_g) ? var_r : var_g;
    var_min = (var_min < var_b) ? var_min : var_b;   //min. value of RGB
    float var_max = (var_r > var_g) ? var_r : var_g;
    var_max = (var_max > var_b) ? var_max : var_b;   //max. value of RGB
    float del_max = var_max - var_min;               //Delta RGB value
    
    L = ( var_max + var_min ) / 2;
    if ( del_max == 0 )//This is a gray, no chroma...
    {
        H = 0;         
        S = 0;    
    }
    else                                    //Chromatic data...
    {
        if ( L < 0.5 )
            S = del_max/(var_max+var_min);
        else
            S = del_max/(2-var_max-var_min );

        float del_r = (((var_max-var_r)/6)+(del_max/2))/del_max;
        float del_g = (((var_max-var_g)/6)+(del_max/2))/del_max;
        float del_b = (((var_max-var_b)/6)+(del_max/2))/del_max;
        if( var_r == var_max ){
            H = del_b - del_g;
        }
        else{       
            if( var_g == var_max ){
                H = (1.0/3.0) + del_r - del_b;
            }
            else{
                    H = (2.0/3.0) + del_g - del_r;
            }   
        }
        
    }
    
    if ( H < 0 )
        H += 1;
    if ( H > 1 )
        H -= 1;

    img_out->h[i] = H;
    img_out->s[i] = S;
    img_out->l[i] = (unsigned char)(L*255);
}

HSL_IMG rgb2hsl_gpu(PPM_IMG img_in) 
{
    // int i;
    // float H, S, L;
    HSL_IMG img_out;// = (HSL_IMG *)malloc(sizeof(HSL_IMG));
    img_out.width  = img_in.w;
    img_out.height = img_in.h;
    img_out.h = (float *)malloc(img_in.w * img_in.h * sizeof(float));
    img_out.s = (float *)malloc(img_in.w * img_in.h * sizeof(float));
    img_out.l = (unsigned char *)malloc(img_in.w * img_in.h * sizeof(unsigned char));
	
	PPM_IMG *img_in_d;
	int ppm_size = sizeof(PPM_IMG);
	
	HSL_IMG *img_out_d;
	int hsl_size = sizeof(HSL_IMG);
	
	cudaMalloc((void**)&img_in_d, ppm_size);
	cudaMalloc((void**)&img_out_d, hsl_size);
	
	cudaMemcpy(img_in_d, &img_in, ppm_size, cudaMemcpyHostToDevice);
	cudaMemcpy(img_out_d, &img_out, hsl_size, cudaMemcpyHostToDevice);
	
	int size = img_in.w * img_in.h;
	
	rgb2hsl_convert<<<size, 1>>>(img_in_d, img_out_d);
	
	cudaMemcpy(&img_out, img_out_d, hsl_size, cudaMemcpyDeviceToHost);
	
	cudaFree(img_in_d);
	cudaFree(img_out_d);

    return img_out;
}