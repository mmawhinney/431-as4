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

__device__ float Hue_2_RGB_gpu( float v1, float v2, float vH )             //Function Hue_2_RGB
{
    if ( vH < 0 ) vH += 1;
    if ( vH > 1 ) vH -= 1;
    if ( ( 6 * vH ) < 1 ) return ( v1 + ( v2 - v1 ) * 6 * vH );
    if ( ( 2 * vH ) < 1 ) return ( v2 );
    if ( ( 3 * vH ) < 2 ) return ( v1 + ( v2 - v1 ) * ( ( 2.0f/3.0f ) - vH ) * 6 );
    return ( v1 );
}

//Convert HSL to RGB, assume H, S in [0.0, 1.0] and L in [0, 255]
//Output R,G,B in [0, 255]
__global__ void hsl2rgb_convert(HSL_IMG *img_in, PPM_IMG *img_out, int *img_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    while(i < *img_size) {
        float H = img_in->h[i];
        float S = img_in->s[i];
        float L = img_in->l[i]/255.0f;
        float var_1, var_2;
        
        unsigned char r,g,b;
        
        if ( S == 0 ) {
            r = L * 255;
            g = L * 255;
            b = L * 255;
        } else {
            if ( L < 0.5 ) {
                var_2 = L * ( 1 + S );
            } else {
                var_2 = ( L + S ) - ( S * L );
            }

            var_1 = 2 * L - var_2;
            r = 255 * Hue_2_RGB_gpu( var_1, var_2, H + (1.0f/3.0f) );
            g = 255 * Hue_2_RGB_gpu( var_1, var_2, H );
            b = 255 * Hue_2_RGB_gpu( var_1, var_2, H - (1.0f/3.0f) );
        }

        __syncthreads();

        img_out->img_r[i] = r;
        img_out->img_g[i] = g;
        img_out->img_b[i] = b;

        i += offset;
    }
}

PPM_IMG hsl2rgb_gpu(HSL_IMG img_in) 
{
    int img_size = img_in.width * img_in.height;
    
    PPM_IMG img_out;
    img_out.w = img_in.width;
    img_out.h = img_in.height;
    img_out.img_r = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    img_out.img_g = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    img_out.img_b = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    

    unsigned char *l = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    float *h = (float *)malloc(img_size * sizeof(float));
    float *s = (float *)malloc(img_size * sizeof(float));

    unsigned char *img_r = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    unsigned char *img_g = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    unsigned char *img_b = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    
    HSL_IMG *img_in_d;
    PPM_IMG *img_out_d;
    int *img_size_d;
    
    cudaMalloc((void**)&img_out_d, sizeof(PPM_IMG));
    cudaMalloc((void**)&img_r, sizeof(unsigned char) * img_size);
    cudaMalloc((void**)&img_g, sizeof(unsigned char) * img_size);
    cudaMalloc((void**)&img_b, sizeof(unsigned char) * img_size);
    

    cudaMalloc((void**)&img_in_d, sizeof(HSL_IMG));
    cudaMalloc((void**)&l, sizeof(unsigned char) * img_size);
    cudaMalloc((void**)&h, sizeof(float) * img_size);
    cudaMalloc((void**)&s, sizeof(float) * img_size);


    cudaMemcpy(img_out_d, &img_out, sizeof(PPM_IMG), cudaMemcpyHostToDevice);
    cudaMemcpy(img_r, img_out.img_r, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_g, img_out.img_g, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_b, img_out.img_b, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(&(img_out_d->img_r), &img_r, sizeof(img_r), cudaMemcpyHostToDevice);
    cudaMemcpy(&(img_out_d->img_g), &img_g, sizeof(img_g), cudaMemcpyHostToDevice);
    cudaMemcpy(&(img_out_d->img_b), &img_b, sizeof(img_b), cudaMemcpyHostToDevice);


    cudaMemcpy(img_in_d, &img_in, sizeof(PPM_IMG), cudaMemcpyHostToDevice);
    cudaMemcpy(l, img_in.l, sizeof(unsigned char) *img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(h, img_in.h, sizeof(float) *img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(s, img_in.s, sizeof(float) *img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(&(img_in_d->l), &l, sizeof(l), cudaMemcpyHostToDevice);
    cudaMemcpy(&(img_in_d->h), &h, sizeof(h), cudaMemcpyHostToDevice);
    cudaMemcpy(&(img_in_d->s), &s, sizeof(s), cudaMemcpyHostToDevice);


    cudaMalloc((void**)&img_size_d, sizeof(int));
    cudaMemcpy(img_size_d, &img_size, sizeof(int), cudaMemcpyHostToDevice);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount;

    hsl2rgb_convert<<<blocks * 2, 1024>>>(img_in_d, img_out_d, img_size_d);
    
    cudaMemcpy(img_out.img_r, img_r, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_g, img_g, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_b, img_b, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);


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


//Convert RGB to YUV, all components in [0, 255]
__global__ void rgb2yuv_convert(PPM_IMG *img_in, YUV_IMG *img_out, int *img_size)
{
    unsigned char r, g, b;
    unsigned char y, cb, cr;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    while(i < *img_size) {
        r = img_in->img_r[i];
        g = img_in->img_g[i];
        b = img_in->img_b[i];
        
        y  = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
        cb = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
        cr = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);
        
        __syncthreads();

        img_out->img_y[i] = y;
        img_out->img_u[i] = cb;
        img_out->img_v[i] = cr;
        
        i += offset;
    }
}

//Convert RGB to YUV, all components in [0, 255]
YUV_IMG rgb2yuv_gpu(PPM_IMG img_in)
{
    YUV_IMG img_out;
    int img_size = img_in.w * img_in.h;
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_y = (unsigned char *)malloc(sizeof(unsigned char) * img_size);
    img_out.img_u = (unsigned char *)malloc(sizeof(unsigned char) * img_size);
    img_out.img_v = (unsigned char *)malloc(sizeof(unsigned char) * img_size);

    unsigned char *img_y = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    unsigned char *img_u = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    unsigned char *img_v = (unsigned char *)malloc(img_size * sizeof(unsigned char));

    unsigned char *img_r = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    unsigned char *img_g = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    unsigned char *img_b = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    
    PPM_IMG *img_in_d;
    YUV_IMG *img_out_d;
    int *img_size_d;
    
    cudaMalloc((void**)&img_out_d, sizeof(YUV_IMG));
    cudaMalloc((void**)&img_y, sizeof(unsigned char) * img_size);
    cudaMalloc((void**)&img_u, sizeof(unsigned char) * img_size);
    cudaMalloc((void**)&img_v, sizeof(unsigned char) * img_size);
    

    cudaMalloc((void**)&img_in_d, sizeof(PPM_IMG));
    cudaMalloc((void**)&img_r, sizeof(unsigned char) * img_size);
    cudaMalloc((void**)&img_g, sizeof(unsigned char) * img_size);
    cudaMalloc((void**)&img_b, sizeof(unsigned char) * img_size);



    cudaMemcpy(img_out_d, &img_out, sizeof(YUV_IMG), cudaMemcpyHostToDevice);
    cudaMemcpy(img_y, img_out.img_y, sizeof(unsigned char) *img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_u, img_out.img_u, sizeof(unsigned char) *img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_v, img_out.img_v, sizeof(unsigned char) *img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(&(img_out_d->img_y), &img_y, sizeof(img_y), cudaMemcpyHostToDevice);
    cudaMemcpy(&(img_out_d->img_u), &img_u, sizeof(img_u), cudaMemcpyHostToDevice);
    cudaMemcpy(&(img_out_d->img_v), &img_v, sizeof(img_v), cudaMemcpyHostToDevice);


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

    rgb2yuv_convert<<<blocks * 2, 1024>>>(img_in_d, img_out_d, img_size_d);
    
    cudaMemcpy(img_out.img_y, img_y, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_u, img_u, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_v, img_v, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);


    cudaFree(img_in_d);
    cudaFree(img_r);
    cudaFree(img_g);
    cudaFree(img_b);
    cudaFree(img_out_d);
    cudaFree(img_y);
    cudaFree(img_u);
    cudaFree(img_v);
    cudaFree(img_size_d);


    return img_out;
}


__device__ unsigned char clip_rgb_gpu(int x)
{
    if(x > 255)
        return 255;
    if(x < 0)
        return 0;

    return (unsigned char)x;
}

//Convert YUV to RGB, all components in [0, 255]
__global__ void yuv2rgb_convert(YUV_IMG *img_in, PPM_IMG *img_out, int *img_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    int  rt,gt,bt;
    int y, cb, cr;

    while(i < *img_size){
        y  = (int)img_in->img_y[i];
        cb = (int)img_in->img_u[i] - 128;
        cr = (int)img_in->img_v[i] - 128;
        
        rt  = (int)( y + 1.402*cr);
        gt  = (int)( y - 0.344*cb - 0.714*cr);
        bt  = (int)( y + 1.772*cb);

        __syncthreads();

        img_out->img_r[i] = clip_rgb_gpu(rt);
        img_out->img_g[i] = clip_rgb_gpu(gt);
        img_out->img_b[i] = clip_rgb_gpu(bt);

        i += offset;
    }
}

//Convert RGB to YUV, all components in [0, 255]
PPM_IMG yuv2rgb_gpu(YUV_IMG img_in)
{
    PPM_IMG img_out;
    int img_size = img_in.w * img_in.h;
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_r = (unsigned char *)malloc(sizeof(unsigned char) * img_size);
    img_out.img_g = (unsigned char *)malloc(sizeof(unsigned char) * img_size);
    img_out.img_b = (unsigned char *)malloc(sizeof(unsigned char) * img_size);

    unsigned char *img_y = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    unsigned char *img_u = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    unsigned char *img_v = (unsigned char *)malloc(img_size * sizeof(unsigned char));

    unsigned char *img_r = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    unsigned char *img_g = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    unsigned char *img_b = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    
    YUV_IMG *img_in_d;
    PPM_IMG *img_out_d;
    int *img_size_d;
    
    cudaMalloc((void**)&img_out_d, sizeof(PPM_IMG));
    cudaMalloc((void**)&img_r, sizeof(unsigned char) * img_size);
    cudaMalloc((void**)&img_g, sizeof(unsigned char) * img_size);
    cudaMalloc((void**)&img_b, sizeof(unsigned char) * img_size);
    
    cudaMalloc((void**)&img_in_d, sizeof(YUV_IMG));
    cudaMalloc((void**)&img_y, sizeof(unsigned char) * img_size);
    cudaMalloc((void**)&img_u, sizeof(unsigned char) * img_size);
    cudaMalloc((void**)&img_v, sizeof(unsigned char) * img_size);


    cudaMemcpy(img_out_d, &img_out, sizeof(PPM_IMG), cudaMemcpyHostToDevice);
    cudaMemcpy(img_r, img_out.img_r, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_g, img_out.img_g, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_b, img_out.img_b, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(&(img_out_d->img_r), &img_r, sizeof(img_r), cudaMemcpyHostToDevice);
    cudaMemcpy(&(img_out_d->img_g), &img_g, sizeof(img_g), cudaMemcpyHostToDevice);
    cudaMemcpy(&(img_out_d->img_b), &img_b, sizeof(img_b), cudaMemcpyHostToDevice);
    


    cudaMemcpy(img_in_d, &img_in, sizeof(YUV_IMG), cudaMemcpyHostToDevice);
    cudaMemcpy(img_y, img_in.img_y, sizeof(unsigned char) *img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_u, img_in.img_u, sizeof(unsigned char) *img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_v, img_in.img_v, sizeof(unsigned char) *img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(&(img_in_d->img_y), &img_y, sizeof(img_y), cudaMemcpyHostToDevice);
    cudaMemcpy(&(img_in_d->img_u), &img_u, sizeof(img_u), cudaMemcpyHostToDevice);
    cudaMemcpy(&(img_in_d->img_v), &img_v, sizeof(img_v), cudaMemcpyHostToDevice);
    


    cudaMalloc((void**)&img_size_d, sizeof(int));
    cudaMemcpy(img_size_d, &img_size, sizeof(int), cudaMemcpyHostToDevice);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount;

    yuv2rgb_convert<<<blocks * 2, 1024>>>(img_in_d, img_out_d, img_size_d);
    
    cudaMemcpy(img_out.img_r, img_r, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_g, img_g, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_b, img_b, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);


    cudaFree(img_in_d);
    cudaFree(img_r);
    cudaFree(img_g);
    cudaFree(img_b);
    cudaFree(img_out_d);
    cudaFree(img_y);
    cudaFree(img_u);
    cudaFree(img_v);
    cudaFree(img_size_d);


    return img_out;
}