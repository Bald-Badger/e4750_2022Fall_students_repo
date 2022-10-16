"""
The code in this file is part of the instructor-provided template for Assignment-1, task-1, Fall 2021. 
"""

import numpy as np
import pyopencl as cl
import pyopencl.array as pycl_array
import time
import matplotlib.pyplot as plt
import pickle

class clModule:
    def __init__(self):
        """
        **Do not modify this code**
        Attributes for instance of clModule
        Includes OpenCL context, command queue, kernel code.
        """

        # Get platform and device property
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        # Returns a list of platform instances and stores it in a string vector called platforms.
        # Basically gets all components on pc that supports and creates a pyopencl.platforms() instance with name platforms.
        # This platforms is a string vector, with many elements, each element being an instance of GPU, or CPU or any other supported opencl platform.
        # Each of these elements obtained using get_platforms() themselves have attributes (defined already on the device like gpu driver binding to PC)
        # These attributes specifies if it is of type CPU, GPU (mentioned in here as device), etc.

        devs = None
        # Initialize devs to None, basically we are creating a null list.
        # Then we go through each element of this platforms vector. Each such element has a method get_devices() defined on it.
        # This will populate the available processors (like number of available GPU threads etc)
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()

        # Create Context:
        # A context is an abstraction for parallel computation. pyopencl.context() method operates on input device and generates an instance of context (here we name it ctx)
        # All variables and operations will be bound to a context as an input argument for opencl functions. This way we can choose explicitly which device we want the code to run on through openCL.
        # Here devs contains devices (GPU threads) and hence the context self.ctx holds information of, and operates on GPU threads.
        self.ctx = cl.Context(devs)

        # Setup Command Queue:
        # A command queue is used to explicitly specify queues within a context. Context by itself has methods pass information from host memory to device memory and vice versa.
        # But a queue can be used to have fine grained control on which part of the data should be accessed in which sequence or order, or acts as a control on the data flow.
        # Here pyopencl.CommandQueue takes in input context and sets some properties (used for enabling debugging options etc), creates a commandqueue bound to this context and stores it to self.queue
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        # define your kernel below.
        kernel_printer_end = """
        #define PRINT_ENABLE_AFTER_COMPUTATION
        """

        kernel_printer = """
        #define PRINT_ENABLE_DEBUG
        """

        kernel_main_wrapper = r"""

        __kernel void main_function(__global float *input_value, __global float *computed_value, int n)
        {
            int idx = get_global_id(0);
            if((idx%2) == 0){
                // [TODO]: STUDENTS SHOULD WRITE CODE TO USE CUDA MATH FUNCTION TO COMPUTE SINE OF INPUT VALUE
                computed_value[idx] = sin(input_value[idx]);
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

        float sine_taylor(float in_raw)
        {
                const float pi = 3.1415926f;
                float in = fmod(in_raw, pi);
                float result;           // final result
                float term = in;        // intermediate term for each iter
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

        self.module_no_print = cl.Program(self.ctx, kernel_device + kernel_main_wrapper).build()
        self.module_with_print_nosync = cl.Program(self.ctx, kernel_printer + kernel_device + kernel_main_wrapper).build()
        self.module_with_print_with_sync = cl.Program(self.ctx, kernel_printer_end + kernel_device + kernel_main_wrapper).build()
        
        # Build kernel code
        # The context (which holds the GPU on which the code should run in) and the kernel code (stored as a string, allowing for metaprogramming if required) are passed onto cl.Program.
        # pyopencl.Program(context,kernelcode).build is similar to sourceModule in Cuda and it returns the kernel function (equivalent of compiled code) that we can pass inputs to.
        # This is stored in self.prg the same way in cuda it is stored in func.
        # self.prg = cl.Program(self.ctx, kernel_code).build()


    def deviceSine(self, a, length, printing_properties):
        """
        Function to perform vector addition using the cl.array class
        Arguments:
            a             :   1st Vector
            b             :   number or vector of equal numbers with same length as a
            length        :   length of vector a
            is_b_a_vector :   Boolean Describing if b is a vector or a number
        Returns:
            c       :   vector sum of arguments a and b
            time_   :   execution time for pocl function 
        """
        # [TODO: Students should write code for the entire method]
        start = time.time()

        # device memory allocation
        a_gpu = pycl_array.to_device(self.queue, a)
        c_gpu = pycl_array.empty_like(a_gpu)

        # execute operation.
        kernel = None
        if(printing_properties == 'No Print'):
            #[TODO: Students to get appropriate compiled kernel]
            kernel = self.module_no_print.main_function
        elif(printing_properties == 'Print'):
            #[TODO: Students to get appropriate compiled kernel]
            kernel = self.module_with_print_nosync.main_function
        else:
            #[TODO: Students to get appropriate compiled kernel]
            kernel = self.module_with_print_with_sync.main_function

        # wait for execution to complete.
        kernel(self.queue, a.shape, None, a_gpu.data, c_gpu.data, np.int32(length))

        # Copy output from GPU to CPU [Use .get() method]
        c = c_gpu.get()

        # Record execution time.
        end = time.time()

        # return a tuple of output of addition and time taken to execute the operation.
        return c, (end - start) * 1e6 # in us


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

        return c, (end - start) * 1e6 # in us
   

def test():
    graphicscomputer = clModule()
    lengths = 10**np.arange(1,3)
    lengths = [ int(x) for x in lengths ]
    times = []
    iter = 50
    for length in lengths:
        cpu_time = 0
        gpu_time = 0
        a_array_np = 0.001*np.arange(1,length+1).astype(np.float32)
        for i in range(iter):
            reference, t0 = graphicscomputer.CPU_Sine(a_array_np, length, "No Print")
            my_answer, t1 = graphicscomputer.deviceSine(a_array_np, length, "No Print")
            cpu_time = cpu_time + t0
            gpu_time = gpu_time + t1
        cpu_time = cpu_time / iter
        gpu_time = gpu_time / iter
        times.append(gpu_time)


if __name__ == "__main__":
    test()