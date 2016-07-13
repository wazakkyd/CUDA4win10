#include<stdio.h>
#include<math.h>
#include<time.h>
#include<stdlib.h>
#include "common_vc.h"

double func2(double x){
	
	return cos(x);
}
	
	


__global__ 
void integrate(double h, double *idata, double *odata)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
	odata[i] = (idata[i] + idata[i+1])*h/2;
}



int main(int argc, char **argv){
	
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
    // initialization
    int size = 2000; // total number of elements to reduce
    printf("    with array size %d  ", size);

    // execution configuration
    int blocksize = 512;   // initial block size
	

	
	double Xmin,Xmax,Div;
	double h;
	double cpu_integral=0.0;
	double dA;
	double y;
	
	Xmin = 0;
	Xmax = M_PI/2.0;
	Div = (double)size;
	h = (Xmax - Xmin)/Div;
	
    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);
	
	size_t bytes = size*sizeof(double);
	double *h_idata = (double *)malloc(bytes);
	double *h_odata = (double *)malloc(bytes);

	int i=0;
	for(y=Xmin;y<=Xmax;y=y+h){
		h_idata[i]=func2(y);
		i++;
	}
	
    double iStart, iElaps;
	iStart = seconds();
	

	for(int i=0;i<size;i++){
		dA = (h_idata[i] + h_idata[i+1])*h/2;
		cpu_integral += dA;
	}
	iElaps = seconds() - iStart;
	
	printf("cpu \t time %lf sec \t cpu_integral: %lf\n", iElaps, cpu_integral);

	double *d_idata = NULL;
	double *d_odata = NULL;
	CHECK(cudaMalloc((void **) &d_idata,bytes));
	CHECK(cudaMalloc((void **) &d_odata,bytes));
	
	
	
	
	
    // kernel 1: reduceNeighbored
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    integrate<<<grid, block>>>(h,d_idata,d_odata);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
	
    CHECK(cudaMemcpy(h_odata, d_odata, bytes,cudaMemcpyDeviceToHost));
	double gpu_integral=0.0;
	
	for(int i=0;i<size;i++) gpu_integral += h_odata[i];
	
	printf("gpu \t time %lf sec \t gpu_integral: %lf <<<grid %d block "
           "%d>>>\n", iElaps, gpu_integral, grid.x, block.x);

    // free host memory
    free(h_idata);
    free(h_odata);

    // free device memory
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    // reset device
    CHECK(cudaDeviceReset());
    
	return 0;
}