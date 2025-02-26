"""
The code in this file is part of the instructor-provided template for Assignment-1, task-2, Fall 2021. 
"""

from cProfile import label
from cmath import isclose
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import matplotlib.pyplot as plt
import pickle

class CudaModule:
    def __init__(self):
        """
        Attributes for instance of CudaModule
        Includes kernel code and input variables.
        """
        self.threads_per_block_x = 1024 # Students can modify this number.
        self.threads_per_block_y = 1
        self.threads_per_block_z = 1
        self.threads_total = self.threads_per_block_x * self.threads_per_block_y * self.threads_per_block_z

        self.getSourceModule()


    def getSourceModule(self):
        """
        Compiles Kernel in Source Module to be used by functions across the class.
        """
        # define your kernel below.
        kernel_printer_end = """
        #define PRINT_ENABLE_AFTER_COMPUTATION
        """

        kernel_printer = """
        #define PRINT_ENABLE_DEBUG
        """

        kernel_main_wrapper = r"""

        __global__ void main_function(float *input_value, float *computed_value, int n)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if((idx%2) == 0){
                // [TODO]: STUDENTS SHOULD WRITE CODE TO USE CUDA MATH FUNCTION TO COMPUTE SINE OF INPUT VALUE
                computed_value[idx] = sinf(input_value[idx]);
                #ifdef PRINT_ENABLE_DEBUG
                if(idx<n)
                {
                    printf("Hello from index %d \n", idx);
                }
                #endif
            }
            else{
                // [TODO]: STUDENTS SHOULD WRITE CODE TO CALL THE DEVICE FUNCTION sine_taylor TO COMPUTE SINE OF INPUT VALUE
                computed_value[idx] = sine_taylor(input_value[idx]);
                #ifdef PRINT_ENABLE_DEBUG
                if(idx<n)
                {
                    // [TODO]: STUDENTS SHOULD WRITE CODE TO PRINT THE INDEX OF THE ARRAY BEING COMPUTED
                    printf("Hello from index %d \n", idx);
                }
                #endif
            }

            #ifdef PRINT_ENABLE_AFTER_COMPUTATION
            if(idx<n)
            {
                // [TODO]: STUDENTS SHOULD WRITE CODE TO PRINT THE INDEX OF THE ARRAY BEING COMPUTED
                printf("index %d completed\n", idx);
            }
            #endif
        }
        """

        kernel_device = """
            #define TAYLOR_COEFFS 10000

            __device__ float sine_taylor(float in_raw) {
                const float pi = 3.1415926f;
                float in = fmod(in_raw, pi);
                float result;           // final result
                float term = in;             // intermediate term for each iter
                float power = in;       // base case
                float factorial = 1;    // base case
                
                float power_iter = in * in;
                float factorial_iter;
                
                for (unsigned int i = 0; i < TAYLOR_COEFFS; i++) {
                    if (i == 0) {
                        result = in;
                        power = in;
                        factorial = 1;
                        continue;
                    }

                    power = power * power_iter;
                    factorial_iter = (2*i) * (2*i+1);
                    factorial = factorial * factorial_iter;
                    term = power / factorial;
                    
                    if (i & 0x01)   // is odd
                        result -= term;
                    else            // is even
                        result += term;
                }
                return result;
            }
        """

        # Compile kernel code and store it in self.module_*

        self.module_no_print = SourceModule(kernel_device + kernel_main_wrapper)
        self.module_with_print_nosync = SourceModule(kernel_printer + kernel_device + kernel_main_wrapper)
        self.module_with_print_with_sync = SourceModule(kernel_printer_end + kernel_device + kernel_main_wrapper)

        # SourceModule is the Cuda.Compiler and the kernelwrapper text is given as input to SourceModule. 
        # This compiler takes in C code as text inside triple quotes (ie a string) and compiles it to CUDA code.
        # When we call this getSourceModule method for an object of this class, 
        # it will return the compiled kernelwrapper function, 
        # which will now take inputs along with block_specifications and grid_specifications.


    def sine_device_mem_gpu(self, a, length, printing_properties):
        """
        Function to perform on-device parallel vector addition
        by explicitly allocating device memory for host variables.
        Arguments:
            a                               :   numpy array of size: length
            b                               :   numpy array of size: length
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """
        # [TODO: Students should write code for the entire method for all cases of printing_properties]
        a               = a.astype(np.float32)
        o               = np.zeros(length, dtype=np.float32)

        # Event objects to mark the start and end points
        start           = cuda.Event()
        malloc_start    = cuda.Event()
        malloc_end      = cuda.Event()
        compute_start   = cuda.Event()
        compute_end     = cuda.Event()
        finish          = cuda.Event()

        # Device memory allocation for input and output arrays
        start.record()
        a_gpu           = cuda.mem_alloc(a.size * a.dtype.itemsize)
        o_gpu           = cuda.mem_alloc(o.size * o.dtype.itemsize)

        # Copy data from host to device
        cuda.memcpy_htod(a_gpu, a)
        cuda.memcpy_htod(o_gpu, o)

        # Call the kernel function from the compiled module
        if(printing_properties == 'No Print'):
            prg = self.module_no_print.get_function("main_function")
        elif(printing_properties == 'Print'):
            prg = self.module_with_print_nosync.get_function("main_function")
        else:
            prg = self.module_with_print_with_sync.get_function("main_function")

        # Get grid and block dim
        blockDim  = (self.threads_per_block_x, self.threads_per_block_y, self.threads_per_block_z)
        gridDim   = (length // self.threads_total + 1, 1, 1)

        # Record execution time and call the kernel loaded to the device
        # void main_function(float *input_value, float *computed_value, int n)
        compute_start.record()
        prg(a_gpu, o_gpu, np.int32(length), block=blockDim, grid=gridDim)

        # Wait for the event to complete
        compute_end.record()
        compute_end.synchronize()

        # Copy result from device to the host
        cuda.memcpy_dtoh(o, o_gpu)
        finish.record()
        finish.synchronize()

        # return a tuple of output of sine computation and time taken to execute the operation.
        return (o, [start.time_till(finish)*1000, compute_start.time_till(compute_end)*1000]) # in us

 
    def CPU_Sine(self, a, length, printing_properties):
        """
        Function to perform vector addition on host(CPU) using numpy add method
        Arguments:
            a       :   1st Vector
            b       :   number or vector of equal numbers with same length as a
            length  :   length of vector a
        """
        start = time.time()
        c = np.sin(a)
        end = time.time()

        return (c, (end - start) * 1000000) # in us


def question12(): # code for task 2-1, 2-2
    print("running code for question 1 and 2")
    graphicscomputer = CudaModule()
    length = 10
    a_array_np = 0.001*np.arange(1,length+1).astype(np.float32)
    my_answer, t0 = graphicscomputer.sine_device_mem_gpu(a_array_np, length, "No Print")
    reference, t1 = graphicscomputer.CPU_Sine(a_array_np, length, "No Print")
    print(a_array_np)
    print(my_answer)
    print(reference)
    print(np.isclose(my_answer, reference))
    
    
def question3():
    print("running code for question 3")
    graphicscomputer = CudaModule()
    lengths = 10**np.arange(1,5)
    lengths = [ int(x) for x in lengths ]
    iter = 50
    for length in lengths:
        cpu_time = 0
        gpu_time = 0
        reference = np.array([])
        my_answer = np.array([])
        a_array_np = 0.001*np.arange(1,length+1).astype(np.float32)
        for i in range(iter):
            reference, t0 = graphicscomputer.CPU_Sine(a_array_np, length, "No Print")
            my_answer, t1 = graphicscomputer.sine_device_mem_gpu(a_array_np, length, "No Print")
            cpu_time = cpu_time + t0
            gpu_time = gpu_time + t1[0]
        cpu_time = cpu_time / iter
        gpu_time = gpu_time / iter
        print("length = {0}".format(length))
        print("cpu time:    {:.3f}, average {:.3f} us per 100 operation".format(cpu_time, cpu_time/length*100))
        print("gpu time: {:.3f}, average {:.3f} us per 100 operation".format(gpu_time, gpu_time/length*100))
        accuracy = np.isclose(my_answer, reference)
        accurate_cnt = 0
        for item in accuracy:
            if item == True:
                accurate_cnt += 1
        print("accuracy: {}".format(accurate_cnt / length))
        

def question4():
    print("running code for question 4")
    graphicscomputer = CudaModule()
    lengths = 10**np.arange(1,7)
    lengths = [ int(x) for x in lengths ]
    iter = 50
    for length in lengths:
        cpu_time = 0
        gpu_time = 0
        reference = np.array([])
        my_answer = np.array([])
        a_array_np = 0.001*np.arange(1,length+1).astype(np.float32)
        for i in range(iter):
            reference, t0 = graphicscomputer.CPU_Sine(a_array_np, length, "No Print")
            my_answer, t1 = graphicscomputer.sine_device_mem_gpu(a_array_np, length, "No Print")
            cpu_time = cpu_time + t0
            gpu_time = gpu_time + t1[0]
        cpu_time = cpu_time / iter
        gpu_time = gpu_time / iter
        print("length = {0}".format(length))
        print("cpu time:    {:.3f}, average {:.3f} us per 100 operation".format(cpu_time, cpu_time/length*100))
        print("gpu time: {:.3f}, average {:.3f} us per 100 operation".format(gpu_time, gpu_time/length*100))
        accuracy = np.isclose(my_answer, reference)
        accurate_cnt = 0
        for item in accuracy:
            if item == True:
                accurate_cnt += 1
        print("accuracy: {}".format(accurate_cnt / length))
        

def question5_filegen():
    graphicscomputer = CudaModule()
    lengths = 10**np.arange(1,7)
    lengths = [ int(x) for x in lengths ]
    times = []
    iter = 50
    for length in lengths:
        cpu_time = 0
        gpu_time = 0
        a_array_np = 0.001*np.arange(1,length+1).astype(np.float32)
        for i in range(iter):
            reference, t0 = graphicscomputer.CPU_Sine(a_array_np, length, "No Print")
            my_answer, t1 = graphicscomputer.sine_device_mem_gpu(a_array_np, length, "No Print")
            cpu_time = cpu_time + t0
            gpu_time = gpu_time + t1[0]
        cpu_time = cpu_time / iter
        gpu_time = gpu_time / iter
        times.append(gpu_time)

    '''
    step for saving result file are skipped
    f = open ('XXX.pkl','wb')
    pickle.dump(times, f)
    f.close()
    '''


def question5_plot():
    f = open("cpu_time.pkl",'rb')
    cpu_time_list = pickle.load(f)
    cpu_time_list = np.log10(cpu_time_list)
    f.close
    
    f = open("gpu5_time.pkl",'rb')
    gpu5_time_list = pickle.load(f)
    gpu5_time_list = np.log10(gpu5_time_list)
    f.close
    
    f = open("gpu10000_time.pkl",'rb')
    gpu10000_time_list = pickle.load(f)
    gpu10000_time_list = np.log10(gpu10000_time_list)
    f.close
    
    lengths = np.log10(10**np.arange(1,7))
    plt.plot(lengths, cpu_time_list, label="cpu_time")
    plt.plot(lengths, gpu5_time_list, label="gpu_5_time")
    plt.plot(lengths, gpu10000_time_list, label="gpu_10000_time")
    plt.legend()
    plt.xlabel("vector length in 10 log scale (10^x)")
    plt.ylabel("time takes in 10 log scale (10^y μs)")
    plt.title("PyCUDA: time it takes for each workload in 10 log scale")
    plt.savefig("cuda_plot.png")
    
        
if __name__ == "__main__":
    question12()
    question3()
    question4()
    print("question 5 need to change variable around so cannot run out of the box")
