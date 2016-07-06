#CUDA 7.5 をWindows10にインストールし、Sampleプログラムを実行するまで

自分のやっていることで並列プログラミングの必要性が出てきたので、かねてから気になっていたGP（汎用）GPUプログラミングの環境を構築しました。

実際環境を整えようとしたところ結構苦労したので、備忘録として今回行ったインストール手順をここに記そうと思います。誰かの助けになれば幸いです。

##検証環境
* OS:Windows10 Pro (64bit)
* CPU: Intel Core i5-4460
* RAM: 8.00GB
* GPU: GeForce GTX 750 Ti


##必要なもの
必要なものは以下の通り。
* [Visual Community 2013](https://www.visualstudio.com/ja-jp/downloads/download-visual-studio-vs.aspx)
* [CUDA7.5](https://developer.nvidia.com/cuda-downloads)
* CUDAに対応したGPU　（自分のもってるGPUが対応してるかどうかは[ここ](https://developer.nvidia.com/cuda-gpus)で調べてくだされ。）

##インストール手順
1. Visual Community 2013のインストール
2. CUDA 7.5のインストール

CUDAは基本的にＣ言語で書かれたプログラムを専用の命令を使って並列処理するツールなので、使用に当たってはC/C++の標準コンパイラが必要です。
LinuxやMacだとgcc、Windows環境ではCUDAバージョンに対応しているVisual Studioのインストールが事前にされている必要があります。（たぶんCUDAを先に入れても問題ないケースもあるのでしょうが、自分の環境ではいろいろと不都合がでたので先にコンパイラをインストールします。）

CUDAのバージョン7.5に対応するVisual Studioは以下の3つです。
* Visual Studio 2013
* Visual Studio 2012
* Visual Studio 2010

#### Visual Studio 2013のインストール
ここでは、Visual Studio 2013をインストールします。

2013のコンパイラは[Visual Community 2013](https://www.visualstudio.com/ja-jp/downloads/download-visual-studio-vs.aspx)をインストールすれば問題なくインストールされます。リンク先にとびダウンロードしてください。

ダウンロードが完了したならば、インストーラを起動してインストールをします。
特別な設定も必要なく、Nextを押していればインストール完了です。


#### CUDA 7.5のインストール
[CUDA7.5](https://developer.nvidia.com/cuda-downloads)のインストールも簡単です。
リンク先にとび、OS、bit、OSversionを適切に選び、インストーラ(network)をダウンロードします。ダウンロードが完了したならば、インストーラーを起動しインストールをします。
事前にVisual Studio 2013がインストールされていれば、CUDA7.5のインストーラは必要な設定などを行ってくれます。

CUDAのインストールが終わったら、コマンドプロンプトで以下のコマンドを打ってみます。
```
> nvcc
```
これを実行して以下のような出力が出たならば、インストールは成功です。
```
nvcc fatal   : No input files specified; use option --help for more information
```
##deviceQueyの実行
さて、上記のインストールがうまくいったかどうかを、今度はCUDAのSampleを使って確認してみます。

CUDAがインストールされると、`C:\ProgramData\NVIDIA Corporation\CUDA Samples\v7.5`
にSampleとなるVisual Studioのソリューションファイルが生成されます。そこで`\NVIDIA Corporation\CUDA Samples\v7.5\1_Utilities\deviceQuery\`のフォルダを開き、`deviceQuery_vs2013.sln`という名前のソリューションファイルをクリックします。すると、Visual Studioが起動しプロジェクトを読み込みます。

プロジェクトの読み込みが完了したら、ソリューションをビルドします。すると`C:\ProgramData\NVIDIA Corporation\CUDA Samples\v7.5\bin\win64\Debug`に`deviceQuery.exe`という名前の実行ファイルが生成されます。

これを確認したら、コマンドプロンプトで実行ファイルのあるディレクトリに移動し、以下のコマンドを打ちます。
```
 > deviceQuery
```
すると、PCに接続されているＧＰＵの情報をピックアップしたものをコマンドライン上に表示してくれます。

##Sampleコードの実行
では実際に打ち込んだコードをコンパイル、実行できるかを試してみます。

サンプルコードは[ここ](https://devblogs.nvidia.com/parallelforall/easy-introduction-cuda-c-and-c/)からいただきました。

コードはこちら
```
#include <stdio.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float));
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
	for (int i = 0; i < N; i++) maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %fn", maxError);
}
```
これは成分が100万個ある要素がすべて１のベクトルｘを2倍し、同じ要素数で、要素がすべて2のベクトルと足し合わせる計算を並列に行うプログラムです。
計算のあとは、各成分の計算が確かに4となっているかを確かめ、その結果を表示する処理が書かれています。

このベクトルの足し算の処理をＣで書くならば
```
for(int i =0 ; i<N; i++)  y[i] = a*x[i] + y[i];
```
ですが、このコードでは
```
int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i < n) y[i] = a*x[i] + y[i];
```
となっているのが特徴です。

さて、このコードを`cudasample01.cu`と保存します。(拡張子`cu`はCUDAのソースコードを意味します)
これをコンパイルをするには、コマンドプロンプトで以下のように実行します。
```
> nvcc cudasample01.cu
```
コンパイルが成功すれば、同じディレクトリに`a.exp`というCUDAの実行ファイルが生成されます。この名前を指定したければ
```
> nvcc -cuda cudasample01.cu
```
とすれば、`cuda.exp`が生成されます。このあたりのオプションはgccなどと共通です。

これを実行して以下のように表示されれば成功です。
```
> cuda
Max error: 0.000000n
```

ここまでできればCUDAプログラミングのスタートラインに立てたと言えるでしょう！たぶん。

##その他エラーについて
###nvccコンパイラでのエラー、警告

#####cl.exeへのパス
windowsでCUDAを使う場合は、C/C++の標準コンパイラがインストールされている必要があることは上記で触れましたが、インストールをしても`cl.exe`へのパスが通っていない場合があります。

その時は環境変数`PATH`に`C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin`を追加してください。

#####警告 C4819
また、コンパイル時に以下のような警告が出る場合があります。
```
c:\program files\nvidia gpu computing toolkit\cuda\v7.5\include\device_functions
.h(1621) : warning C4819: The file contains a character that cannot be represent
ed in the current code page (932). Save the file in Unicode format to prevent da
ta loss
```
警告`C4819`はコードやincudeファイルの文字コードが適切でないために出る警告です。これは、`c:\program files\nvidia gpu computing toolkit\cuda\v7.5\include\`以下にあるヘッダファイル(`*.h,*.hpp`)の文字コードをすべてUTF-8に変換することで解消されます。

ちなみに私の環境ではほとんどのヘッダファイルがASCIIコードになっておりました。
複数のファイルの文字コードを変換するには[FCCchecker](http://www.vector.co.jp/soft/dl/winnt/util/se478635.html)というフリーウェアが便利でした。

FCCcheckerで文字コードを変更できない場合は該当のファイルをVisual Studioで開き、[ファイル]→[保存オプションの詳細設定]から、[Unicode(UTF-8 シグネチャ付き)]を選択して上書き保存すればOKです。
