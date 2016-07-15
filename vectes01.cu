#include <stdio.h>

//�J�[�l���֐�vec_sum�̐錾
__global__
void vec_sum(float k, float *a, float *b, float *c)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  c[i] = k*a[i] + b[i];
}


int main(void)
{
  //N = 512�~2048
	int N = 1<<20;

  //a,b,c�̓z�X�g�p�Ad_a,d_b,d_c�̓f�o�C�X�p�̃|�C���^
  float *a, *b, *c  *d_a, *d_b, *d_c;

  //�z�X�g���̔z���p��
  a = (float*)malloc(N*sizeof(float));
	b = (float*)malloc(N*sizeof(float));
  c = (float*)malloc(N*sizeof(float));

  //�f�o�C�X���̔z���p��
  cudaMalloc(&d_a, N*sizeof(float));
  cudaMalloc(&d_b, N*sizeof(float));
  cudaMalloc(&d_c, N*sizeof(float));

 //a,b�̔z��ɂ��ꂼ��1�C2�������Ac��������
  for (int i = 0; i < N; i++) {
    a[i] = 1.0f;
    b[i] = 2.0f;
    c[i] = 0.0f;
  }

  //�z�X�g���̔z��̓��e���f�o�C�X���ɃR�s�[
  cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, N*sizeof(float), cudaMemcpyHostToDevice);


  //�X���b�h�̐ݒ�
  int blocksize = 512;

  //�u���b�N������̃X���b�h���iblocksize)��512�A
  //�u���b�N�̑����igridsize�j��N/512�p�ӂ���
  //���������đ��X���b�h���� blocksize �~ gridsize = N ��
  dim3 block (blocksize, 1, 1);
  dim3 grid  (N / block.x, 1, 1);

  // �J�[�l���֐��̌Ăяo��
  vec_sum<<<grid, block>>>(2.0f, d_a, d_b,d_c);

  //�v�Z���ʂ��z�X�g�փR�s�[
  cudaMemcpy(c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;

  //�v�Z���ʂ̊m�F
	for (int i = 0; i < N; i++) maxError = max(maxError, abs(c[i]-4.0f));
  printf("Max error: %f", maxError);

  //�������̊J��
  free(a);
  free(b);
  free(c);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

 return 0;
}