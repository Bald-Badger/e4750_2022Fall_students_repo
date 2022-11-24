import numpy as np
import pyopencl as cl
import pyopencl.array as pycl_array
import time
import matplotlib.pyplot as plt
import math


class PrefixSum:
    def __init__(self, section_size=256):
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()
                
        # Create Context:
        self.ctx = cl.Context(devs)
        
        # Setup Command Queue:
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        self.section_size = section_size
        
        kernel_section_size = "#define SECTION_SIZE {}".format(self.section_size)
        
        kernel_naive        = r"""    
        __kernel void partial_sum ( 
            __global float* in, 
            __global float* out, 
            __global float* step_sum, 
            const    int    size
        ) {
            const int gidx = get_global_id(0);
            const int gidy = get_global_id(1);
            const int lidx = get_local_id(0);
            const int lidy = get_local_id(1);
            const int tid  = lidx;
            const int i    = gidx * (gidy + 1);;
            
            /*
            printf(
                "hello from gixd: %d, gidy:%d, lidx: %d, lidy %d, i=%d\n",
                gidx, gidy, lidx, lidy, i
            );
            */
            
            __local float seg[SECTION_SIZE];
             
            if (i < size) {
                seg[tid] = in[i];
            }
            
            barrier(CLK_LOCAL_MEM_FENCE);
            
            for (unsigned int stride = 1; stride <= SECTION_SIZE; stride <<= 1) {
                float partial = 0;
                int index = tid - stride;
                if (index >= 0 && index < SECTION_SIZE) {
                    partial = seg[index];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                seg[tid] += partial;
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            
            barrier(CLK_LOCAL_MEM_FENCE);
            
            if (i < size) {
                out[i] = seg[tid];
            }
            
            barrier(CLK_LOCAL_MEM_FENCE);
            
            if ( (i + 1) % SECTION_SIZE == 0 ) {
                step_sum[i / SECTION_SIZE] = out[i];
            }
        }
        
        
        __kernel void add_partial_sum (
            __global float* in, 
            __global float* out, 
            __global float* step_sum, 
            const    int    size
        ) {
            int gidx = get_global_id(0);
            int gidy = get_global_id(1);
            const int i   = gidx * (gidy + 1);;
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
        __kernel void partial_sum (
            __global float* in, 
            __global float* out, 
            __global float* step_sum, 
            const    int    size
        ) {            
            const int gidx = get_global_id(0);
            const int gidy = get_global_id(1);
            const int lidx = get_local_id(0);
            const int lidy = get_local_id(1);
            const int tid  = lidx;
            const int i    = gidx * (gidy + 1);;
            __local float XY[SECTION_SIZE * 2];
            
            if (i < size) {
                XY[tid] = in[i];
            }
            
            for (unsigned int stride = 1; stride <= SECTION_SIZE; stride <<=1) {
                int index = (tid + 1) * stride * 2 - 1;
                if (index < 2 * SECTION_SIZE) {
                    XY[index] += XY[index - stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            
            for (unsigned int stride = SECTION_SIZE / 2; stride > 0; stride >>= 1) {
                barrier(CLK_LOCAL_MEM_FENCE);
                int index = (tid + 1) * stride * 2 - 1;
                if (index + stride < 2 * SECTION_SIZE) {
                    XY[index + stride] += XY[index];
                }
            }
            
            barrier(CLK_LOCAL_MEM_FENCE);
            
            if (i < size) {
                out[i] = XY[tid];
            }
            
            if ( (i + 1) % SECTION_SIZE == 0 ) {
                step_sum[i / SECTION_SIZE] = out[i];
            }
        }
        """
        

        self.module_navie = cl.Program(self.ctx, kernel_section_size + kernel_naive).build()
        
        self.module_efficient = cl.Program(self.ctx, kernel_section_size + kernel_efficient).build()


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
            end = time.time()
            return arr, ((end - start) * 1000000) #in us
        else:
            for i in range(len(arr)):
                if i == 0:
                    out[i] = arr[i]
                else:
                    out[i] = out[i-1] + arr[i] 
        
        end = time.time()
        return out, ((end - start) * 1000000) #in us


    def prefix_sum_gpu_wrapper(self, in_cpu=np.ndarray, seg_size=32, scheme="naive"):
        # sanity check
        scheme_list = ['naive', 'efficient']
        assert scheme in scheme_list
        
        in_size = in_cpu.shape[0]
        in_cpu = in_cpu.astype(np.float32)
        
        start = time.time()
        
        step_size = math.ceil(in_size / seg_size)
        out_cpu = np.zeros_like(in_cpu)
        step_cpu = np.zeros(step_size, dtype=np.float32)
        
        in_gpu = pycl_array.to_device(self.queue, in_cpu)
        out_gpu = pycl_array.zeros(self.queue, shape=out_cpu.shape, dtype=np.float32)
        step_gpu = pycl_array.zeros(self.queue, shape=step_cpu.shape, dtype=np.float32)
        
        if scheme == 'naive':
            kernel = self.module_navie.partial_sum
        elif scheme == 'efficient':
            kernel = self.module_efficient.partial_sum
        else:
            assert 0
            exit()
            
        kernel (
            self.queue, 
            (in_size, 1),  # global size ?
            (seg_size, 1), # local size ?
            in_gpu.data, 
            out_gpu.data, 
            step_gpu.data, 
            np.int32(in_size)
        )
        
        step_cpu = step_gpu.get()
        
        kernel = self.module_navie.add_partial_sum
        
        # yeah... not quite enough time for recursive GPU call
        step_cpu, t = self.prefix_sum_python(step_cpu)
        
        step_gpu = pycl_array.to_device(self.queue, step_cpu)
        
        kernel (
            self.queue, 
            (in_size, 1),  # global size ?
            (seg_size, 1), # local size ?
            in_gpu.data, 
            out_gpu.data, 
            step_gpu.data, 
            np.int32(in_size)
        )
        
        out_cpu = out_gpu.get()
        
        end = time.time()
        
        return (out_cpu, (end - start) * 1e6) # in us
        

    def prefix_sum_gpu_naive(self, in_cpu=np.ndarray, seg_size=32):
        return self.prefix_sum_gpu_wrapper(in_cpu, seg_size, "naive")


    def prefix_sum_gpu_work_efficient(self, in_cpu=np.ndarray, seg_size=32):
        return self.prefix_sum_gpu_wrapper(in_cpu, seg_size, "efficient")


    def test_prefix_sum_python(self, arr=np.ndarray):
        ref, tr = self.reference_sum(arr)
        out, to = self.prefix_sum_python(arr)
        assert len(ref) == len(out)
        for i in range(len(ref)):
            if (np.isclose(ref[i], out[i])):
                print("test_prefix_sum_python passed")
                return True
            else:
                print("test_prefix_sum_python failed")
                return False


    def __test_prefix_sum_gpu_wrapper(self, arr=np.ndarray, scheme=""):
        ref, tr = self.reference_sum(arr)
        if scheme == "naive":
            out, to = self.prefix_sum_gpu_naive(arr, self.section_size)
        elif scheme == "efficient":
            out, to = self.prefix_sum_gpu_work_efficient(arr, self.section_size)
        assert len(ref) == len(out)
        if (np.isclose(ref[len(ref)-1], out[len(ref)-1])):
            print("test_prefix_sum_gpu_{} passed".format(scheme))
        else:
            print("test_prefix_sum_gpu_{} failed".format(scheme))
            print("expect {}, get {}".format(out[len(ref)-1], ref[len(ref)-1]))


    def test_prefix_sum_gpu_naive(self, arr=np.ndarray):
        self.__test_prefix_sum_gpu_wrapper(arr, "naive")


    def test_prefix_sum_gpu_work_efficient(self, arr=np.ndarray):
        self.__test_prefix_sum_gpu_wrapper(arr, "efficient")
    
    
def main():
    # init object
    seg_len = 128
    compute = PrefixSum(seg_len)
    # size_list = [128, 2048, 262144, 4194304, 134215680]
    size_list = [128, 2048, 262144, 4194304, 134215680]
    
    # Programming Task 1. 1-D Scan - Naive Python algorithm
    arr1 = np.random.random_sample(5)
    arr2 = np.random.random_sample(5)
    compute.test_prefix_sum_python(arr1)
    compute.test_prefix_sum_python(arr2)
    
    # Programming Task 2-3 1-D Scan - Work-efficient scan
    if (False): # takes too much time, change to true if want to run
        for n in size_list:
            arr = np.random.random_sample(n).astype(np.float32)
            res = compute.test_prefix_sum_python(arr)
            if (res == True):
                print("python naive test for size {} passed".format(n))
            else:
                print("python naive test for size {} failed".format(n))
    
    # Programming Task 2-1 1-D Scan - Work-inefficient scan
    arr = np.ones(4194304, dtype=np.float32)
    compute.test_prefix_sum_gpu_naive(arr)
    
    # Programming Task 2-2 1-D Scan - Work-efficient scan
    compute.test_prefix_sum_gpu_work_efficient(arr)
    
    # record and plot time takes for each scheme
    
    py_times        = np.array([])
    naive_times     = np.array([])
    efficient_times = np.array([])
    
    for n in size_list:
        arr = np.ones(n, dtype=np.float32)
        compute.test_prefix_sum_gpu_naive(arr)
        compute.test_prefix_sum_gpu_work_efficient(arr)
        c0, t0          = compute.prefix_sum_python(arr)
        c1, t1          = compute.prefix_sum_gpu_naive(arr, seg_len)
        c2, t2          = compute.prefix_sum_gpu_work_efficient(arr, seg_len)
        py_times        = np.append(py_times, t0)
        naive_times     = np.append(naive_times, t1)
        efficient_times = np.append(efficient_times, t2)
        
    size_list_normalize = np.log2(size_list)
    py_times            = np.log2(py_times)
    naive_times         = np.log2(naive_times)
    efficient_times     = np.log2(efficient_times)
    
    plt.figure(0)
    plt.plot(size_list_normalize, py_times, label="cpu")
    plt.plot(size_list_normalize, naive_times, label="gpu naive")
    plt.plot(size_list_normalize, efficient_times, label="gpu efficient")
    plt.legend()
    plt.xlabel("input length in 2 log scale (2^x)")
    plt.ylabel("time takes in 2 log scale (2^y μs)")
    plt.title("PyOpenCL: time it takes for each workload in 2 log scale")
    plt.savefig("pycl_plot.png")


if __name__ == "__main__":
    main()
