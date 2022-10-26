# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 16:02:25 2021

@author: manueljenkin
"""

"""
The code in this file is part of the instructor-provided template for Assignment-1, task-1, Fall 2021. 
"""

import numpy as np
import pyopencl as cl
import pyopencl.tools
import pyopencl.array as cl_array
import time

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
        
        # kernel - will not be provided for future assignments!
        # The arguments (output:c and inputs:a,b) stored in global memory are passed with __global type. The other argument n, containing the number of elements is additionally passed
        # with a qualifier const to allow the compiler to optimize for it (it is the same value that is to be passed to each thread)
        # get_global_id will get the Global Item ID (equivalent to thread ID in cuda) for the current instance.
        # The if condition before the actual computation is for bounds checking to ensure threads donot operate on invalid indexes.

        kernel_printer_end = """
        #define SYNC_BEFORE_PRINT
        """

        kernel_printer = """
        #define PRINT_ENABLE_DEBUG
        """

        kernel_main_wrapper = r"""

        __kernel void main_function(__global float *input_value,__global float *computed_value, const unsigned int n)
        {
            unsigned int i = get_global_id(0);
            if((i%2) == 0){
                computed_value[i] = sin(input_value[i]);
                #ifdef PRINT_ENABLE_DEBUG
                if(i<n)
                {
                    printf("Hello from thread %d \n", i);
                }
                #endif
            }
            else{
                computed_value[i] = sine_taylor(input_value[i]);
                #ifdef PRINT_ENABLE_DEBUG
                if(i<n)
                {
                    printf("Hello from thread %d \n", i);
                }
                #endif
            }

            #ifdef SYNC_BEFORE_PRINT
            if(i<n)
            {
                printf("Hello from thread %d \n", i);
            }
            #endif     
        }
        """

        kernel_device = """
        #define TAYLOR_COEFFS 10000

        float sine_taylor(float in)
        {
            float val = 0;
            int sign = 1;
            int coeff_power;
            int factorial_compute;
            int i;
            int j;
            for(i=1; i<=TAYLOR_COEFFS; i++){
                coeff_power = (2*i-1);
                factorial_compute = 1;
                for (j=1; j<=coeff_power; j++){
                    factorial_compute = factorial_compute*j;
                }
                val = val + ((sign*pow(in,coeff_power))/factorial_compute);
                sign *= -1;
            }
            return val;
        }
        """
        
        # Build kernel code
        # The context (which holds the GPU on which the code should run in) and the kernel code (stored as a string, allowing for metaprogramming if required) are passed onto cl.Program.
        # pyopencl.Program(context,kernelcode).build is similar to sourceModule in Cuda and it returns the kernel function (equivalent of compiled code) that we can pass inputs to.
        # This is stored in self.prg the same way in cuda it is stored in func.
        # self.prg = cl.Program(self.ctx, kernel_code).build()

        self.module_no_print = cl.Program(self.ctx, kernel_device + kernel_main_wrapper).build()
        self.module_with_print_nosync = cl.Program(self.ctx, kernel_printer + kernel_device + kernel_main_wrapper).build()
        self.module_with_print_with_sync = self.prg = cl.Program(self.ctx, kernel_printer_end + kernel_device + kernel_main_wrapper).build()

    def deviceSine(self, a, length, printing_properties):
        """
        Function to perform vector addition using the cl.array class
        Arguments:
            a       :   1st Vector
            b       :   2nd Vector
            length  :   length of vectors.
        Returns:
            c       :   vector sum of arguments a and b
            time_   :   execution time for pocl function 
        """
        # We are using queue of the context to write the memory allocation and kernel computation using OpenCL.
        l = np.uintc(length)
        # In kernel code it expects unsigned integer, so we pass as numpy uintc which the pyopencl wrapper will convert to uintc before passing to kernel.

        # device memory allocation
        a_gpu = cl_array.to_device(self.queue, a)
        # This is a commands to the function. They are not executed yet. They are executed only when the kernel function is called (in this deviceAdd method, it is in func)
        # pyopencl.Array is used for working with numpy like syntax on OpenCL using python.
        # The to_device method now creates a command to bind a from self.queue and self.ctx. When the function is executed it will copy to GPU memory of same size and work on it.
        c_gpu = cl_array.empty_like(a_gpu)
        # here we pass a_gpu which is of cl_array type and it create command for a similar array of same size.
        # the empty_like method syntax is a little different from the to_device method in terms of argument order.
        # This empty_like method has also only created a command to create c_gpu inside gpu. This hasn't yet created anything in gpu. During func it will create.

        # execute operation.
        time1_start = time.clock()
        if(printing_properties == 'No Print'):
            func = self.module_no_print.main_function(self.queue, a.shape, None, c_gpu.data, a_gpu.data, l)
        elif(printing_properties == 'Print'):
            func = self.module_with_print_nosync.main_function(self.queue, a.shape, None, c_gpu.data, a_gpu.data, l)
        else:
            func = self.module_with_print_with_sync.main_function(self.queue, a.shape, None, c_gpu.data, a_gpu.data, l)

        # self.prg is defined in init and it calls the kernel compilation. From this we call the Kernel Function (sum) we defined using the self.prg.sum method.
        # we pass the queue along with c_gpu, a_gpu, b_gpu all which we earlier attached to the queue.
        # The second argument is similar to block in cuda, third argument None is similar to grid in cuda.
        # a.shape unlike cuda can take any values (cuda can take only multiples of 32). So this does thread allocation automatically.

        # wait for execution to complete.
        func.wait()
        
        # Copy output from GPU to CPU [Use .get() method]
        c = c_gpu.get(self.queue)
        # Very similar to gpuarray in cuda, here we fetch it from the queue after the program is complete.
        # running this command before evt.wait() will cause erroneous results (in that case we would be fetching before operation is complete on GPU)

        # Record execution time.
        time1_end = (time.clock() - time1_start)
        # time = evt.profile.end - evt.profile.start

        # return a tuple of output of addition and time taken to execute the operation.
        return c, time1_end

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

        return c, end - start

if __name__ == "__main__":
    # List all main methods
    all_main_methods = ['CPU Sine', 'Sine_device_mem_gpu']
    # List the two operations
    all_operations = ['No Print', 'Print', 'Sync then Print']
    # List the size of vectors
    vector_sizes = 10**np.arange(1,3)
    # List iteration indexes
    iteration_indexes = np.arange(1,2)
    # Select the list of valid operations for profiling
    valid_operations = all_operations

    # Create an instance of the clModule class
    graphicscomputer = clModule()

    for current_operation in valid_operations:
        print ("current operation")
        print(current_operation)
        arr_avg_total_cpu_time = np.array([])
        arr_avg_total_gpu_cl_array_time = np.array([])
        
        for vector_size in vector_sizes:

            arr_total_cpu_time = np.array([])
            arr_total_gpu_cl_array_time = np.array([])

            print ("vectorlength")
            print (vector_size)

            a_array_np = 0.001*np.arange(1,vector_size+1).astype(np.float32)
            percentdone = 0
            for iteration in iteration_indexes:
                if (((iteration+1)%5) == 0):
                    percentdone = percentdone + 10
                    # print (percentdone)
                for current_method in all_main_methods:
                    if(current_method == 'CPU Sine'):
                        c_np_cpu_add, cpu_time_add = graphicscomputer.CPU_Sine(a_array_np,vector_size,current_operation)
                        arr_total_cpu_time = np.append(arr_total_cpu_time, cpu_time_add)
                    else:
                        if(current_method == 'Sine_device_mem_gpu'):
                            c_gpu_device_add, gpu_time_cl_array = graphicscomputer.deviceSine(a_array_np,vector_size,current_operation) 
                            sum_diff = c_gpu_device_add - c_np_cpu_add
                            arr_total_gpu_cl_array_time = np.append(arr_total_gpu_cl_array_time, gpu_time_cl_array)
                        total_diff = sum_diff.sum()
                        if (total_diff != 0):
                            print (current_method + " " + current_operation + "sum mismatch")
                            print (total_diff)
            avg_total_cpu_time = ((arr_total_cpu_time.sum())/50)
            arr_avg_total_cpu_time = np.append(arr_avg_total_cpu_time, avg_total_cpu_time)
            avg_total_gpu_cl_array_time = ((arr_total_gpu_cl_array_time.sum())/50)
            arr_avg_total_gpu_cl_array_time = np.append(arr_avg_total_gpu_cl_array_time, avg_total_gpu_cl_array_time)
        print(" The CPU times are")
        print(arr_avg_total_cpu_time)
        print(" The GPU CL Array times are")
        print(arr_avg_total_gpu_cl_array_time)