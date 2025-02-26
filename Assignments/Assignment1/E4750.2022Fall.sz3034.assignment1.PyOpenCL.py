"""
The code in this file is part of the instructor-provided template for Assignment-1, task-1, Fall 2021. 
"""

import numpy as np
import pyopencl as cl
import pyopencl.array as pycl_array
import time
import matplotlib.pyplot as plt

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

            __kernel void Add_two_vectors_GPU(__global float* c, __global float* a, __global float* b, const unsigned int n)
            {
                unsigned int i = get_global_id(0);
                if (i < n) {
                    c[i] = a[i] + b[i];
                }
            }

            __kernel void Add_to_each_element_GPU(__global float* c, __global float* a, __global float* b, const unsigned int n)
            {
                unsigned int i = get_global_id(0);
                if (i < n) {
                    c[i] = a[i] + b[0];
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
            a             :   1st Vector
            b             :   number or vector of equal numbers with same length as a
            length        :   length of vector a
            is_b_a_vector :   Boolean Describing if b is a vector or a number
        Returns:
            c       :   vector sum of arguments a and b
            time_   :   execution time for pocl function 
        """
        # [TODO: Students should write code for the entire method for both cases of is_b_a_vector]
        # device memory allocation
        start = time.time()

        a_gpu = pycl_array.to_device(self.queue, a)
        if (is_b_a_vector):
            b_gpu = pycl_array.to_device(self.queue, b)
        else:
            b_arr = np.zeros(1).astype(np.float32)
            b_arr[0] = b
            b_gpu = pycl_array.to_device(self.queue, b_arr)
        c_gpu = pycl_array.empty_like(a_gpu)
        
        # execute operation.
        if (is_b_a_vector):
            # Use `Add_two_vectors_GPU` Kernel.
            kernel = self.prg.Add_two_vectors_GPU
        else:
            # Use `Add_to_each_element_GPU` Kernel
            kernel = self.prg.Add_to_each_element_GPU

        # wait for execution to complete.
        kernel(self.queue, a.shape, None, c_gpu.data, a_gpu.data, b_gpu.data, length)

        # Copy output from GPU to CPU [Use .get() method]
        c = c_gpu.get()

        # Record execution time.
        end = time.time()

        # return a tuple of output of addition and time taken to execute the operation.
        return c, (end - start) * 1000000 # in us

    def bufferAdd(self, a, b, length, is_b_a_vector):
        """
        Function to perform vector addition using the cl.Buffer class
        Returns:
            c               :    vector sum of arguments a and b
            end - start     :    execution time for pocl function 
        """
        # [TODO: Students should write code for the entire method for both cases of is_b_a_vector]
        # Create three buffers (plans for areas of memory on the device)
        start = time.time()
        
        mf = cl.mem_flags
        a_gpu = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        if (is_b_a_vector):
            b_gpu = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        else:
            b_arr = np.zeros(1).astype(np.float32)
            b_arr[0] = b
            b_gpu = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_arr)
        c_gpu = cl.Buffer(self.ctx, mf.WRITE_ONLY, a.nbytes)

        # execute operation.
        if (is_b_a_vector):
            # Use `Add_two_vectors_GPU` Kernel.
            kernel = self.prg.Add_two_vectors_GPU
        else:
            # Use `Add_to_each_element_GPU` Kernel
            kernel = self.prg.Add_to_each_element_GPU

        # Wait for execution to complete.
        kernel(self.queue, a.shape, None, c_gpu, a_gpu, b_gpu, length)

        # Copy output from GPU to CPU [Use enqueue_copy]
        c = np.zeros_like(a)
        cl.enqueue_copy(self.queue, c, c_gpu)
        # Record execution time.
        end = time.time()

        # return a tuple of output of addition and time taken to execute the operation.
        return c, (start - end) * 1000000 # in us

    def CPU_numpy_Add(self, a, b, length, is_b_a_vector):
        """
        Function to perform vector addition on host(CPU) using numpy add method
        Arguments:
            a             :   1st Vector
            b             :   number or vector of equal numbers with same length as a
            length        :   length of vector a
            is_b_a_vector :   Boolean Describing if b is a vector or a number
        """
        start = time.time()
        c = a + b
        end = time.time()

        return c, (end - start) * 1000000

    def CPU_Loop_Add(self, a, b, length, is_b_a_vector):
        """
        Function to perform vector addition on host(CPU) using numpy add method
        Arguments:
            a             :   1st Vector
            b             :   number or vector of equal numbers with same length as a
            length        :   length of vector a
            is_b_a_vector :   Boolean Describing if b is a vector or a number
        """

        start = time.time()
        c = np.empty_like(a)
        for index in np.arange(0,length):
            if (is_b_a_vector == True):
                c[index] = a[index] + b[index]
            else:
                c[index] = a[index] + b
        end = time.time()

        return c, (end - start) * 1000000


def main():
    # init plot variables
    cpu_plt, (c1, c2) = plt.subplots(2, 1, figsize=(10, 10))
    gpu_plt, (g1, g2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # List all main methods
    all_main_methods = ['CPU numpy Add', 'CPU_Loop_Add', 'DeviceAdd', 'BufferAdd']
    # List the two operations
    all_operations = ['Pass Vector and Number', 'Pass Two Vectors']
    # List the size of vectors
    vector_sizes = 10**np.arange(1,9)
    # List iteration indexes
    iter_cnt = 2
    iteration_indexes = np.arange(1,iter_cnt)

    # Select the list of valid methods to perform (populate as you complete the methods).
    # Currently in template code only CPU Add and CPU Loop Add are complete.
    valid_main_methods = all_main_methods[0:4]

    # Select the list of valid operations to be run
    valid_operations = all_operations

    # Select the list of valid vector_sizes for current_analysis
    valid_vector_sizes = vector_sizes[0:8]

    # Create an instance of the clModule class
    graphicscomputer = clModule()

    # Nested loop precedence, operations -> vector_size -> iteration -> CPU/GPU method.
    # There are four nested loops, the main loop iterates between performing vector + number, and performing vector + vector cases.
    # The second loop iterates between different vector sizes, for each case of the main loop.
    # The third loop runs 50 repetitions, for each case of the second loop
    # The fourth loop iterates between the different CPU/GPU/Memory-transfer methods, for each case of the third loop.

    for current_operation in valid_operations:
        # Set initial arrays to populate average computation times for different vector sizes
        arr_avg_total_cpu_time = np.array([])
        arr_avg_total_cpu_loop_time = np.array([])
        # [TODO: Students should write Code]
        # Add for the rest of the methods
        arr_avg_total_device_add_time = np.array([])
        arr_avg_total_buffer_add_time = np.array([])
        
        for vector_size in valid_vector_sizes:

            arr_total_cpu_time = np.array([])
            arr_total_cpu_loop_time = np.array([])

            # [TODO: Students should write Code]
            # Add for the rest of the methods
            arr_total_device_add_time = np.array([])
            arr_total_buffer_add_time = np.array([])

            print ("vectorlength")
            print (vector_size)

            a_array_np = np.arange(1,vector_size+1).astype(np.float32) # Generating a vector having values 1 to vector_size as Float32 datatype.
            b = 3 # Choose any number you desire
            b_number_np = np.float32(b) # storing as number having value b with datatype Float32
            b_array_np = b*np.ones(vector_size).astype(np.float32) # storing as array with all elements having equal value b as datatype Float32
            for iteration in iteration_indexes:
                for current_method in valid_main_methods:
                    if(current_operation == 'Pass Vector and Number'):
                        is_b_a_vector = False
                        b_in = b_number_np
                    else:
                        is_b_a_vector = True
                        b_in = b_array_np

                    if(current_method == 'CPU numpy Add' and vector_size <= 1e7):
                        c_np_cpu_add, cpu_time_add = graphicscomputer.CPU_numpy_Add(a_array_np,b_in,vector_size,is_b_a_vector)
                        arr_total_cpu_time = np.append(arr_total_cpu_time, cpu_time_add)
                    
                    if(current_method == 'CPU_Loop_Add' and vector_size <= 1e6):
                        c_np_cpu_loop_add, cpu_time_loop_add = graphicscomputer.CPU_Loop_Add(a_array_np,b_in,vector_size,is_b_a_vector)
                        sum_diff = c_np_cpu_loop_add - c_np_cpu_add
                        arr_total_cpu_loop_time = np.append(arr_total_cpu_loop_time, cpu_time_loop_add)
                        
                    # [TODO: Students should write Code]
                    # Add for the rest of the methods
                    if(current_method == 'DeviceAdd'): # baseline
                        c_np_device_add, c_np_device_time = graphicscomputer.CPU_Loop_Add(a_array_np,b_in,vector_size,is_b_a_vector)
                        arr_total_device_add_time = np.append(arr_total_device_add_time, c_np_device_time)

                    if(current_method == 'BufferAdd'):
                        c_np_buffer_add, c_np_buffer_time = graphicscomputer.CPU_Loop_Add(a_array_np,b_in,vector_size,is_b_a_vector)
                        arr_total_buffer_add_time = np.append(arr_total_buffer_add_time, c_np_buffer_time)
                        
                        # test
                        sum_diff = c_np_buffer_add - c_np_device_add
                        total_diff = sum_diff.sum()
                        if (total_diff != 0):
                            print (current_method + " " + current_operation + "sum mismatch")
                            print (total_diff)

            avg_total_cpu_time = ((arr_total_cpu_time.sum())/iter_cnt)
            if avg_total_cpu_time > 0:
                arr_avg_total_cpu_time = np.append(arr_avg_total_cpu_time, avg_total_cpu_time)
            avg_total_cpu_loop_time = ((arr_total_cpu_loop_time.sum())/iter_cnt)
            if avg_total_cpu_loop_time > 0:
                arr_avg_total_cpu_loop_time = np.append(arr_avg_total_cpu_loop_time, avg_total_cpu_loop_time)
            # [TODO: Students should write Code]
            # Add for the rest of the methods
            avg_total_device_add_time = ((arr_total_device_add_time.sum())/iter_cnt)
            arr_avg_total_device_add_time = np.append(arr_avg_total_device_add_time, avg_total_device_add_time)
            avg_total_buffer_add_time = ((arr_total_buffer_add_time.sum())/iter_cnt)
            arr_avg_total_buffer_add_time = np.append(arr_avg_total_buffer_add_time, avg_total_buffer_add_time)
            
            
        print(current_operation + "The CPU times are")
        print(arr_avg_total_cpu_time)
        print(current_operation + " The CPU Loop times are")
        print(arr_avg_total_cpu_loop_time)
        # [TODO: Students should write Code]
        # Add for the rest of the methods

        print(current_operation + "The device add times are")
        print(arr_avg_total_device_add_time)
        print(current_operation + " The buffer add times are")
        print(arr_avg_total_buffer_add_time)
        
        # Code for Plotting the results (the code for plotting can be skipped, if the student prefers to have a separate code for plotting, or to use a different software for plotting)
        # plot cpu time useage
        plt.figure(1)
        x = np.arange(1,arr_avg_total_cpu_time.shape[0] + 1)
        y = np.log10(arr_avg_total_cpu_time)
        c1.plot(x, y, label="V + V" if current_operation == 'Pass Two Vectors' else "V + S")
        c1.legend()
        c1.title.set_text('CPU vector')
        c1.set_xlabel('vector length in 10 log scale')
        c1.set_ylabel('microsecond in 10 log scale')
        
        x = np.arange(1,arr_avg_total_cpu_loop_time.shape[0] + 1)
        y = np.log10(arr_avg_total_cpu_loop_time)
        c2.plot(x, y, label="V + V" if current_operation == 'Pass Two Vectors' else "V + S")
        c2.legend()
        c2.title.set_text('CPU loop')
        c2.set_xlabel('vector length in 10 log scale')
        c2.set_ylabel('microsecond in 10 log scale')
        
        # plot gpu time useage
        plt.figure(2)
        x = np.arange(1,arr_avg_total_device_add_time.shape[0] + 1)
        y = np.log10(arr_avg_total_device_add_time)
        g1.plot(x, y, label="V + V" if current_operation == 'Pass Two Vectors' else "V + S")
        g1.legend()
        g1.title.set_text('device_add_gpu')
        g1.set_xlabel('vector length in 10 log scale')
        g1.set_ylabel('microsecond in 10 log scale')
        
        y = np.log10(arr_avg_total_buffer_add_time)
        g2.plot(x, y, label="V + V" if current_operation == 'Pass Two Vectors' else "V + S")
        g2.legend()
        g2.title.set_text('buffer_add_gpu')
        g2.set_xlabel('vector length in 10 log scale')
        g2.set_ylabel('microsecond in 10 log scale')
        
    plt.figure(1)
    plt.savefig("opencl_cpu.png")
    
    plt.figure(2)
    plt.savefig("opencl_gpu.png")
        
def myTest():
    size = np.int32(4)
    a = np.random.random(size).astype(np.float32)
    b = np.random.random(size).astype(np.float32)
    print(a)
    print(b)
    graphicscomputer = clModule()
    c, t0 = graphicscomputer.deviceAdd(a,b,size,True)
    print(c)
    c, t0 = graphicscomputer.deviceAdd(a,np.float(4),size,False)
    print(c)

if __name__ == "__main__":
    main()
    # myTest()
