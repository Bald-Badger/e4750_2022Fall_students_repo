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
        kernel_code = """
            __kernel void Add_to_each_element_GPU(__global float* c, __global float* a, __global float* b, const unsigned int n)
            {
                unsigned int i = get_global_id(0);
                if (i < n) {
                    c[i] = a[i] + b[0];
                }
            }

            __kernel void Add_two_vectors_GPU(__global float* c, __global float* a, __global float* b, const unsigned int n)
            {
                unsigned int i = get_global_id(0);
                if (i < n) {
                    c[i] = a[i] + b[i];
                }
            }
        """ 
        
        # Build kernel code
        # The context (which holds the GPU on which the code should run in) and the kernel code (stored as a string, allowing for metaprogramming if required) are passed onto cl.Program.
        # pyopencl.Program(context,kernelcode).build is similar to sourceModule in Cuda and it returns the kernel function (equivalent of compiled code) that we can pass inputs to.
        # This is stored in self.prg the same way in cuda it is stored in func.
        self.prg = cl.Program(self.ctx, kernel_code).build()

    def deviceAdd(self, a, b, length, is_b_a_vector):
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
        b_gpu = cl_array.to_device(self.queue, b)  
        c_gpu = cl_array.empty_like(a_gpu)
        # here we pass a_gpu which is of cl_array type and it create command for a similar array of same size.
        # the empty_like method syntax is a little different from the to_device method in terms of argument order.
        # This empty_like method has also only created a command to create c_gpu inside gpu. This hasn't yet created anything in gpu. During func it will create.

        # execute operation.
        time1_start = time.clock()
        if(is_b_a_vector is False):
            func = self.prg.Add_to_each_element_GPU(self.queue, a.shape, None, c_gpu.data, a_gpu.data, b_gpu.data, l)
        else:
            func = self.prg.Add_two_vectors_GPU(self.queue, a.shape, None, c_gpu.data, a_gpu.data, b_gpu.data, l)
        
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

    def bufferAdd(self, a, b, length, is_b_a_vector):
        """
        Function to perform vector addition using the cl.Buffer class
        Returns:
            c               :    vector sum of arguments a and b
            end - start     :    execution time for pocl function 
        """
        l = np.uintc(length)
        # Create three buffers (plans for areas of memory on the device)
        a_gpu = cl.Buffer (self.ctx, cl.mem_flags.COPY_HOST_PTR, size=a.nbytes, hostbuf=a)
        b_gpu = cl.Buffer (self.ctx, cl.mem_flags.COPY_HOST_PTR, size=b.nbytes, hostbuf=b)
        c_gpu = cl.Buffer (self.ctx, cl.mem_flags.WRITE_ONLY, size=a.nbytes)

        # execute operation.
        time1_start = time.clock()
        if(is_b_a_vector is False):
            func = self.prg.Add_to_each_element_GPU(self.queue, a.shape, None, c_gpu.data, a_gpu.data, b_gpu.data, l)
        else:
            func = self.prg.Add_two_vectors_GPU(self.queue, a.shape, None, c_gpu.data, a_gpu.data, b_gpu.data, l)
        
        # Wait for execution to complete.
        func.wait()
        
        # Copy output from GPU to CPU [Use enqueue_copy]
        c = np.empty_like(a)
        cl.enqueue_copy(self.queue, c, c_gpu)
        
        # Record execution time.
        time1_end = (time.clock() - time1_start)

        # return a tuple of output of addition and time taken to execute the operation.
        return c, time1_end

    def CPU_Add(self, a, b, length, is_b_a_vector):
        """
        Function to perform vector addition on host(CPU) using numpy add method
        Arguments:
            a       :   1st Vector
            b       :   number or vector of equal numbers with same length as a
            length  :   length of vector a
        """
        start = time.time()
        c = a + b
        end = time.time()

        return c, end - start

    def CPU_Loop_Add(self, a, b, length, is_b_a_vector):
        """
        Function to perform vector addition on host(CPU) by using for loops
        Arguments:
            a       :   1st Vector
            b       :   number or vector of equal numbers with same length as a
            length  :   length of vector a
        """

        start = time.time()
        c = np.empty_like(a)
        for index in np.arange(0,length):
            if (is_b_a_vector == True):
                c[index] = a[index] + b[index]
            else:
                c[index] = a[index] + b
        end = time.time()

        return c, end - start

if __name__ == "__main__":
    # List all main methods
    all_main_methods = ['CPU Add', 'CPU_Loop_Add', 'DeviceAdd', 'BufferAdd']
    # List the two operations
    all_operations = ['Pass Vector and Number', 'Pass Two Vectors']
    # List the size of vectors
    vector_sizes = 10**np.arange(1,9)
    # List iteration indexes
    iteration_indexes = np.arange(1,50)
    # Select the list of valid operations for profiling
    valid_operations = all_operations

    # Create an instance of the clModule class
    graphicscomputer = clModule()

    for current_operation in valid_operations:
        arr_avg_total_cpu_time = np.array([])
        arr_avg_total_cpu_loop_time = np.array([])
        arr_avg_total_gpu_cl_array_time = np.array([])
        arr_avg_total_total_gpu_cl_buffer_time = np.array([])
        
        for vector_size in vector_sizes:

            arr_total_cpu_time = np.array([])
            arr_total_cpu_loop_time = np.array([])
            arr_total_gpu_cl_array_time = np.array([])
            arr_total_gpu_cl_buffer_time = np.array([])

            print ("vectorlength")
            print (vector_size)

            a_array_np = np.arange(1,vector_size+1).astype(np.float32)
            b = 3 # Choose any number you desire
            b_number_np = np.float32(b) # storing as number having value b with datatype Float32
            b_array_np = b*np.ones(vector_size).astype(np.float32) # storing as array with all elements having equal value b as datatype Float32
            percentdone = 0
            for iteration in iteration_indexes:
                if (((iteration+1)%5) == 0):
                    percentdone = percentdone + 10
                    # print (percentdone)
                for current_method in all_main_methods:
                    if(current_operation == 'Pass Vector and Number'):
                        is_b_a_vector = False
                        b_in = b_number_np
                    else:
                        is_b_a_vector = True
                        b_in = b_array_np
                    if(current_method == 'CPU Add'):
                        c_np_cpu_add, cpu_time_add = graphicscomputer.CPU_Add(a_array_np,b_in,vector_size,is_b_a_vector)
                        arr_total_cpu_time = np.append(arr_total_cpu_time, cpu_time_add)
                    else:
                        if(current_method == 'CPU_Loop_Add'):
                            c_np_cpu_loop_add, cpu_time_loop_add = graphicscomputer.CPU_Loop_Add(a_array_np,b_in,vector_size,is_b_a_vector)
                            sum_diff = c_np_cpu_loop_add - c_np_cpu_add
                            arr_total_cpu_loop_time = np.append(arr_total_cpu_loop_time, cpu_time_loop_add)
                        if(current_method == 'DeviceAdd'):
                            c_gpu_device_add, gpu_time_cl_array = graphicscomputer.deviceAdd(a_array_np,b_in,vector_size,is_b_a_vector) 
                            sum_diff = c_gpu_device_add - c_np_cpu_add
                            arr_total_gpu_cl_array_time = np.append(arr_total_gpu_cl_array_time, gpu_time_cl_array)
                        if(current_method == 'bufferAdd'):
                            c_gpu_buffer_add, gpu_time_cl_buffer = graphicscomputer.bufferAdd(a_array_np,b_in,vector_size,is_b_a_vector)
                            sum_diff = c_gpu_buffer_add - c_np_cpu_add
                            arr_total_gpu_cl_buffer_time = np.append(arr_total_gpu_cl_buffer_time, gpu_time_cl_buffer)
                        total_diff = sum_diff.sum()
                        if (total_diff != 0):
                            print (current_method + " " + current_operation + "sum mismatch")
                            print (total_diff)
            avg_total_cpu_time = ((arr_total_cpu_time.sum())/50)
            arr_avg_total_cpu_time = np.append(arr_avg_total_cpu_time, avg_total_cpu_time)
            avg_total_cpu_loop_time = ((arr_total_cpu_loop_time.sum())/50)
            arr_avg_total_cpu_loop_time = np.append(arr_avg_total_cpu_loop_time, avg_total_cpu_loop_time)
            avg_total_gpu_cl_array_time = ((arr_total_gpu_cl_array_time.sum())/50)
            arr_avg_total_gpu_cl_array_time = np.append(arr_avg_total_gpu_cl_array_time, avg_total_gpu_cl_array_time)
            avg_total_total_gpu_cl_buffer_time = ((arr_total_gpu_cl_buffer_time.sum())/50)
            arr_avg_total_total_gpu_cl_buffer_time = np.append(arr_avg_total_total_gpu_cl_buffer_time, avg_total_total_gpu_cl_buffer_time)
        print(" The CPU times are")
        print(arr_avg_total_cpu_time)
        print(" The CPU Loop times are")
        print(arr_avg_total_cpu_loop_time)
        print(" The GPU CL Array times are")
        print(arr_avg_total_gpu_cl_array_time)
        print(" The GPU CL Buffer times are")
        print(arr_avg_total_total_gpu_cl_buffer_time)