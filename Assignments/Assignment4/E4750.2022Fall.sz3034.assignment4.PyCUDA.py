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
        __global__ void partial_sum (float* in, float* out, float* step_sum, int size) {
            __shared__ float seg[SECTION_SIZE];
            const int tid = threadIdx.x;
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (i < size) {
                seg[tid] = in[i];
            }
            
            __syncthreads();
            
            for (int stride = 1; stride <= SECTION_SIZE; stride <<= 1) {
                __syncthreads();
                //if (tid == 0) printf("tid 0 reached 3 \n", tid);
                float partial = 0;
                int index = tid - stride;
                if (index >= 0 && index < SECTION_SIZE) {
                    partial = seg[index];
                }
                __syncthreads();
                seg[tid] += partial; 
            }
            
            __syncthreads();
            
            if (i < size) {
                out[i] = seg[tid];
            }
            
            __syncthreads();
            
            if ( (i + 1) % SECTION_SIZE == 0 ) {
                step_sum[i / SECTION_SIZE] = out[i];
            }
        }
        
        __global__ void add_partial_sum (float* in, float* out, float* step_sum, int size) {
            const int tid = threadIdx.x;
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            int add_val = 0;
            int seg_num = 0;
            if (i < size) {
                seg_num = i / SECTION_SIZE;
            }
            if (seg_num > 0 && i < size) {
                out[i] = out[i] + step_sum[seg_num-1];
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
        arr = arr.astype(np.float32)
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


    def prefix_sum_gpu_naive(self, in_cpu=np.ndarray, seg_size=32):
        # start = cuda.Event()
        # compute = cuda.Event()
        # finish = cuda.Event()
        # print("segment size is {}".format(self.threads_per_block_x))
        out = np.zeros_like(in_cpu)
        step_size = math.ceil(in_cpu.shape[0]/seg_size)
        step_cpu = np.zeros(step_size, dtype=np.float32)
        # print(step_cpu)
        # start.record()
        in_gpu      = cuda.mem_alloc(in_cpu.shape[0] * in_cpu.dtype.itemsize)
        out_gpu     = cuda.mem_alloc(in_cpu.shape[0] * in_cpu.dtype.itemsize)
        step_gpu    = cuda.mem_alloc(step_cpu.shape[0] * step_cpu.dtype.itemsize)
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
        
        func = self.gpu_naive_module.get_function("partial_sum")
        
        func (
            in_gpu,
            out_gpu,
            step_gpu,
            np.int32(in_cpu.size),
            block=blockDim,
            grid=gridDim
        )
        print("step 1")
        cuda.memcpy_dtoh(step_cpu, step_gpu)
        step_cpu,t = self.prefix_sum_python(step_cpu)
        cuda.memcpy_htod(step_gpu, step_cpu)
        #print(step_cpu)
        print("step 2")
        
        func = self.gpu_naive_module.get_function("add_partial_sum")
        blockDim  = (1024, 1, 1)
        
        gridDim   = (math.ceil(in_cpu.shape[0]/1024), 1, 1)
        
        func (
            in_gpu,
            out_gpu,
            step_gpu,
            np.int32(in_cpu.size),
            block=blockDim,
            grid=gridDim
        )
        
        print("step 3")
        
        cuda.memcpy_dtoh(out, out_gpu)
        
        return (out, 0) # in us


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
        out, to = self.prefix_sum_gpu_naive(arr, self.threads_per_block_x)
        assert len(ref) == len(out)
        if (np.isclose(ref[len(ref)-1], out[len(ref)-1])):
            print("test_prefix_sum_gpu_naive passed")
        else:
            print("test_prefix_sum_gpu_naive failed")
            print("expect {}, get {}".format(out[len(ref)-1], ref[len(ref)-1]))


    def test_prefix_sum_gpu_work_efficient(self):
        pass
    
    
def main():
    # init object
    seg_len = 1024
    compute = PrefixSum(seg_len)
    
    # Programming Task 1. 1-D Scan - Naive Python algorithm
    arr1 = np.random.random_sample(5)
    arr2 = np.random.random_sample(5)
    compute.test_prefix_sum_python(arr1)
    compute.test_prefix_sum_python(arr2)
    
    # Programming Task 2. 1-D Scan - Programing in PyCuda and PyOpenCL
    arr = np.ones(4194304, dtype=np.float32)
    # out, to = compute.prefix_sum_gpu_naive(arr, seg_len)
    # print(arr)
    # print(out)
    compute.test_prefix_sum_gpu_naive(arr)


if __name__ == "__main__":
    main()
