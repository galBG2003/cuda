 #include "../include/kernels.cuh"
 #include "../include/cudaFunctions.cuh"


cudaError_t addWithCuda(float *c, const float *a, const float *b, unsigned int size,unsigned int ThreadsPerBlock, unsigned int n, void(*Kernel)(float*, const float*, const float*, unsigned int)){
    float *dev_a = 0, *dev_b = 0, *dev_c = 0;
    cudaError_t cudaStatus;
    int numBlocks = (size/n + ThreadsPerBlock - 1)/ThreadsPerBlock;
    cudaStatus = cudaSetDevice(0);

    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));

    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

    Kernel<<<numBlocks, ThreadsPerBlock>>>(dev_c, dev_a, dev_b,size);

    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;   
}
cudaError_t addWithCuda(float *c, const float *a, const float *b, unsigned int size,unsigned int ThreadsPerBlock, unsigned int n, void(*Kernel)(float*, const float*, const float*, unsigned int, unsigned int)){
 float *dev_a = 0, *dev_b = 0, *dev_c = 0;
    cudaError_t cudaStatus;
    int numBlocks = (size/n + ThreadsPerBlock - 1)/ThreadsPerBlock;
    cudaStatus = cudaSetDevice(0);

    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));

    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

    Kernel<<<numBlocks, ThreadsPerBlock>>>(dev_c, dev_a, dev_b,size,n);

    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;   
}