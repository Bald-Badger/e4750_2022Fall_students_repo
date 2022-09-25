"""
The code in this file is part of the instructor-provided template for Assignment-1, task-1, Fall 2021. 
"""

import numpy as np
import relevant.libraries

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

            __kernel void Add_to_each_element_GPU( # Input Arguments )
            {
                # Kernel code to add a number to each element in the vector.
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

        # execute operation.
        if (is_b_a_vector == True):
            # Use `Add_two_vectors_GPU` Kernel.
        else:
            # Use `Add_to_each_element_GPU` Kernel

        # wait for execution to complete.

        # Copy output from GPU to CPU [Use .get() method]

        # Record execution time.

        # return a tuple of output of addition and time taken to execute the operation.
        pass

    def bufferAdd(self, a, b, length, is_b_a_vector):
        """
        Function to perform vector addition using the cl.Buffer class
        Returns:
            c               :    vector sum of arguments a and b
            end - start     :    execution time for pocl function 
        """
        # [TODO: Students should write code for the entire method for both cases of is_b_a_vector]
        # Create three buffers (plans for areas of memory on the device)

        # execute operation.
        if (is_b_a_vector == True):
            # Use `Add_two_vectors_GPU` Kernel.
        else:
            # Use `Add_to_each_element_GPU` Kernel
        
        # Wait for execution to complete.
        
        # Copy output from GPU to CPU [Use enqueue_copy]

        # Record execution time.

        # return a tuple of output of addition and time taken to execute the operation.
        pass

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

        return c, end - start

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

        return c, end - start

if __name__ == "__main__":
    # List all main methods
    all_main_methods = ['CPU numpy Add', 'CPU_Loop_Add', 'DeviceAdd', 'BufferAdd']
    # List the two operations
    all_operations = ['Pass Vector and Number', 'Pass Two Vectors']
    # List the size of vectors
    vector_sizes = 10**np.arange(1,9)
    # List iteration indexes
    iteration_indexes = np.arange(1,50)

    # Select the list of valid methods to perform (populate as you complete the methods).
    # Currently in template code only CPU Add and CPU Loop Add are complete.
    valid_main_methods = all_main_methods[0:2]

    # Select the list of valid operations to be run
    valid_operations = all_operations

    # Select the list of valid vector_sizes for current_analysis
    valid_vector_sizes = vector_sizes[0:6]

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
        
        for vector_size in valid_vector_sizes:

            arr_total_cpu_time = np.array([])
            arr_total_cpu_loop_time = np.array([])

            # [TODO: Students should write Code]
            # Add for the rest of the methods

            print ("vectorlength")
            print (vector_size)

            a_array_np = np.arange(1,vector_size+1).astype(np.float32) # Generating a vector having values 1 to vector_size as Float32 datatype.
            b = 3 # Choose any number you desire
            b_number_np = np.float32(b) # storing as number having value b with datatype Float32
            b_array_np = b*np.ones(vector_size).astype(np.float32) # storing as array with all elements having equal value b as datatype Float32
            percentdone = 0
            for iteration in iteration_indexes:
                for current_method in valid_main_methods:
                    if(current_operation == 'Pass Vector and Number'):
                        is_b_a_vector = False
                        b_in = b_number_np
                    else:
                        is_b_a_vector = True
                        b_in = b_array_np
                    if(current_method == 'CPU numpy Add'):
                        c_np_cpu_add, cpu_time_add = graphicscomputer.CPU_numpy_Add(a_array_np,b_in,vector_size,is_b_a_vector)
                        arr_total_cpu_time = np.append(arr_total_cpu_time, cpu_time_add)
                    else:
                        if(current_method == 'CPU_Loop_Add'):
                            c_np_cpu_loop_add, cpu_time_loop_add = graphicscomputer.CPU_Loop_Add(a_array_np,b_in,vector_size,is_b_a_vector)
                            sum_diff = c_np_cpu_loop_add - c_np_cpu_add
                            arr_total_cpu_loop_time = np.append(arr_total_cpu_loop_time, cpu_time_loop_add)
                        
                        # [TODO: Students should write Code]
                        # Add for the rest of the methods
                       
                        total_diff = sum_diff.sum()
                        if (total_diff != 0):
                            print (current_method + " " + current_operation + "sum mismatch")
                            print (total_diff)
            avg_total_cpu_time = ((arr_total_cpu_time.sum())/50)
            arr_avg_total_cpu_time = np.append(arr_avg_total_cpu_time, avg_total_cpu_time)
            avg_total_cpu_loop_time = ((arr_total_cpu_loop_time.sum())/50)
            arr_avg_total_cpu_loop_time = np.append(arr_avg_total_cpu_loop_time, avg_total_cpu_loop_time)
            # [TODO: Students should write Code]
            # Add for the rest of the methods
        print(current_operation + "The CPU times are")
        print(arr_avg_total_cpu_time)
        print(current_operation + " The CPU Loop times are")
        print(arr_avg_total_cpu_loop_time)
        # [TODO: Students should write Code]
        # Add for the rest of the methods
        # Code for Plotting the results (the code for plotting can be skipped, if the student prefers to have a separate code for plotting, or to use a different software for plotting)