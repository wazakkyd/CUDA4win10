#include <stdio.h>

//カーネル関数vec_sumの宣言
__global__
void vec_sum(float k, float *a, float *b, float *c)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  c[i] = k*a[i] + b[i];
}


int main(void)
{
  //N = 512×2048
	int N = 1<<20;

  //a,b,cはホスト用、d_a,d_b,d_cはデバイス用のポインタ
  float *a, *b, *c  *d_a, *d_b, *d_c;

  //ホスト側の配列を用意
  a = (float*)malloc(N*sizeof(float));
	b = (float*)malloc(N*sizeof(float));
  c = (float*)malloc(N*sizeof(float));

  //デバイス側の配列を用意
  cudaMalloc(&d_a, N*sizeof(float));
  cudaMalloc(&d_b, N*sizeof(float));
  cudaMalloc(&d_c, N*sizeof(float));

 //a,bの配列にそれぞれ1，2を代入し、cを初期化
  for (int i = 0; i < N; i++) {
    a[i] = 1.0f;
    b[i] = 2.0f;
    c[i] = 0.0f;
  }

  //ホスト側の配列の内容をデバイス側にコピー
  cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, N*sizeof(float), cudaMemcpyHostToDevice);


  //スレッドの設定
  int blocksize = 512;

  //ブロックあたりのスレッド数（blocksize)を512、
  //ブロックの総数（gridsize）をN/512用意する
  //したがって総スレッド数は blocksize × gridsize = N 個
  dim3 block (blocksize, 1, 1);
  dim3 grid  (N / block.x, 1, 1);

  // カーネル関数の呼び出し
  vec_sum<<<grid, block>>>(2.0f, d_a, d_b,d_c);

  //計算結果をホストへコピー
  cudaMemcpy(c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;

  //計算結果の確認
	for (int i = 0; i < N; i++) maxError = max(maxError, abs(c[i]-4.0f));
  printf("Max error: %f", maxError);

  //メモリの開放
  free(a);
  free(b);
  free(c);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

 return 0;
}