
#define tile 32 //tile size for tiling
// Create other necessary functions here
__global__ void MatrixMul(int *a,int *b,int *c,int N){
    int i = ((blockDim.y * blockIdx.y) + threadIdx.y);
    int j = ((blockDim.x * blockIdx.x) + threadIdx.x);
    int index = i*N/2+j;
    i = i<<1; 
    j = j<<1;
    int temp = 0;
    for (int k = 0; k < N; k++) {
        temp += (a[i * N + k]+a[(i+1)*N + k]) * (b[k * N + j] + b[(k*N)+(j+1)]);
    }
    c[index] = temp;
}

//Tiled Matrix Mul
__global__ void MatrixMulTiled(int *a,int *b,int *c,int N){
    __shared__ int shareA[tile][tile];
    __shared__ int shareB[tile][tile];
    int i = ((blockDim.y * blockIdx.y) + threadIdx.y);//row
    int j = ((blockDim.x * blockIdx.x) + threadIdx.x);//column
    int index = i*N/2+j;
    i = i<<1;
    j = j<<1;
    int txi = threadIdx.x;
    int tyj = threadIdx.y;
    int temp = 0;
    for(int i1=0;i1<N/tile;i1++){
        shareA[tyj][txi] = a[i * N + txi + tile*i1]+a[(i+1)*N + txi + tile*i1];
        shareB[tyj][txi] = b[tyj * N + j + tile*i1*N] + b[(tyj*N)+(j+1) + tile*i1*N];
        __syncthreads();
        for(int k=0;k<tile;k++){
            temp += shareA[tyj][k] * shareB[k][txi];
            __syncthreads();
        }
    }
    c[index] = temp;
}

// Fill in this function
void gpuThread(int N, int *matA, int *matB, int *output)
{
    size_t bytes = N * N * sizeof(int); //Defining custom size variable 
    // Allocate device memory
    int *a, *b, *c;

    cudaMalloc(&a, bytes);
    cudaMalloc(&b, bytes);
    cudaMalloc(&c, bytes/4);

    // Copy data to the device
    cudaMemcpy(a, matA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b, matB, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(c, output, bytes/4, cudaMemcpyHostToDevice);


    int Threads = tile; // Number of threads in row & column of Thread Block
    int Blocks = (0.5*N)/(Threads); //Number of Blocks per row & column in block Grid

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(Threads, Threads); //2D thread structure
    dim3 blocks(Blocks, Blocks);  // 2D block structure

    int kernel = 1;
    switch (kernel){
        case 0:
            MatrixMul<<<blocks, threads>>>(a, b ,c, N);
            break;
        
        case 1:
            MatrixMulTiled<<<blocks, threads>>>(a, b ,c, N);
            break;
    }



    // Copy results to host
    cudaMemcpy(output, c, bytes/4, cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
