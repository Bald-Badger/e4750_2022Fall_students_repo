import pycuda.driver as cuda
import pycuda.autoinit
from   pycuda.compiler import SourceModule
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import math


class PrefixSum:
    def __init__(self, tx=1024):
        self.threads_per_block_x = tx   # max kernel size is 1024
        self.threads_per_block_y = 1
        self.threads_per_block_z = 1
        self.threads_total = self.threads_per_block_x * self.threads_per_block_y * self.threads_per_block_z
        self.getSourceModule()


    def getSourceModule(self):
        section_size        = "#define SECTION_SIZE {}".format(self.threads_per_block_x)

        kernel_naive        = r"""    
        
        /*
        __global__ void kernel_naive_ref (float* input, float* output, int list_size) {
            __shared__ float intersession[SECTION_SIZE];
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < list_size)
                intersession[threadIdx.x]=input[index];
            for (int stride=1;stride<=threadIdx.x;stride<<=1) {
                __syncthreads();
                float inter = intersession[threadIdx.x-stride];
                __syncthreads();
                intersession[threadIdx.x] += inter;
            }
            __syncthreads();
            if (index < list_size)
                output[index]=intersession[threadIdx.x];
        }
        */
         
        __global__ void kernel_naive (float* in, float* out, int size) {
            __shared__ float seg[SECTION_SIZE];
            int tid = threadIdx.x;
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (i < size) {
                seg[tid] = in[i];
            } else {}
            
            //__syncthreads();
            
            printf("hello from id %d \n", i);
            
            //__syncthreads();
            
            for (int stride = 1; stride <= tid; stride <<= 1) {
                printf("thread %i reached pre_sync\n", i);
                //__syncthreads();  // uncomment this will hang the kernel
                printf("thread %i reached post_sync\n", i);
                float partial = 0;
                int index = tid - stride;
                if (index >= 0 && index < SECTION_SIZE) {
                    partial = seg[index];
                } else {}
                //__syncthreads(); // uncomment this will hang the kernel
                seg[tid] += partial;
                printf("thread %i reached loop end\n", i);
            }
            
            __syncthreads();
            
            if (i < size) {
                out[i] = seg[tid];
            }
        }
        
        """
        
        kernel_efficient    = r"""
        
        """

        self.gpu_naive_module       = SourceModule(section_size + kernel_naive)
        self.gpu_efficient_module   = SourceModule(kernel_efficient)


    def reference_sum(self, arr=np.ndarray):
        start = time.time()
        out = np.cumsum(arr)
        end = time.time()
        return (out, (end - start) * 1000000)   #in us 


    def prefix_sum_python(self, arr=np.ndarray):
        start = time.time()
        arr = arr.astype(np.float)
        out = np.zeros_like(arr)

        if (len(arr) == 1):
            return arr
        else:
            for i in range(len(arr)):
                if i == 0:
                    out[i] = arr[i]
                else:
                    out[i] = out[i-1] + arr[i] 
        
        end = time.time()
        return (out, (end - start) * 1000000)   #in us


    def prefix_sum_gpu_naive(self, in_cpu=np.ndarray):
        start = cuda.Event()
        compute = cuda.Event()
        finish = cuda.Event()
        out = np.zeros_like(in_cpu)
        start.record()
        
        in_gpu = cuda.mem_alloc(in_cpu.shape[0] * in_cpu.dtype.itemsize)
        out_gpu = cuda.mem_alloc(in_cpu.shape[0] * in_cpu.dtype.itemsize)
        cuda.memcpy_htod(in_gpu, in_cpu)
        
        blockDim  = (
            self.threads_per_block_x, 
            self.threads_per_block_y, 
            self.threads_per_block_z
        )
        
        gridDim   = (math.ceil(in_cpu.shape[0]/self.threads_per_block_x), 1, 1)
        
        print("array size is {}".format(in_cpu.shape))
        print("block dim is ({},{},{})".format(blockDim[0], blockDim[1], blockDim[2]))
        print("grid dim is ({},{},{})".format(gridDim[0], gridDim[1], gridDim[2]))
        
        func = self.gpu_naive_module.get_function("kernel_naive")
        
        start.synchronize()
        
        func(
            in_gpu,
            out_gpu,
            np.int32(in_cpu.size),
            block=blockDim,
            grid=gridDim
        )
        
        compute.record()
        compute.synchronize()

        cuda.memcpy_dtoh(out, out_gpu)
        finish.record()
        finish.synchronize()
        
        return (out, start.time_till(finish)*1000) # in us


    def prefix_sum_gpu_work_efficient(self):
        pass


    def test_prefix_sum_python(self, arr=np.ndarray):
        ref, tr = self.reference_sum(arr)
        out, to = self.prefix_sum_python(arr)
        assert len(ref) == len(out)
        for i in range(len(ref)):
            assert np.isclose(ref[i], out[i])
        print("test_prefix_sum_python passed")


    def test_prefix_sum_gpu_naive(self, arr=np.ndarray):
        ref, tr = self.reference_sum(arr)
        out, to = self.prefix_sum_gpu_naive(arr)
        assert len(ref) == len(out)
        assert np.isclose(ref, out)
        for i in range(len(ref)):
            assert np.isclose(ref[i], out[i])
        print("test_prefix_sum_gpu_naive passed")


    def test_prefix_sum_gpu_work_efficient(self):
        pass
    
    
def main():
    # init object
    seg_len = 8
    compute = PrefixSum(seg_len)
    
    # Programming Task 1. 1-D Scan - Naive Python algorithm
    arr1 = np.random.random_sample(5)
    arr2 = np.random.random_sample(5)
    compute.test_prefix_sum_python(arr1)
    compute.test_prefix_sum_python(arr2)
    
    # Programming Task 2. 1-D Scan - Programing in PyCuda and PyOpenCL
    # arr = np.random.randint(16, size=5)
    arr = np.ones(32, dtype=np.float32)
    out, to = compute.prefix_sum_gpu_naive(arr)
    print(arr)
    print(out)
    # compute.test_prefix_sum_gpu_naive(arr)


if __name__ == "__main__":
    main()
