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
        
        # define your kernel below.
        kernel_printer_end = """
        #define PRINT_ENABLE_AFTER_COMPUTATION
        """

        kernel_printer = """
        #define PRINT_ENABLE_DEBUG
        """

        kernel_main_wrapper = r"""

        __kernel void main_function(float *input_value, float *computed_value, int n)
        {
            TODO: STUDENTS TO WRITE KERNEL CODE MATCHING FUNCTIONALITY (INLCUDING PRINT AND COMPILE TIME CONDITIONS) OF THE CUDA KERNEL.
        }
        """

        kernel_device = """
        #define TAYLOR_COEFFS 10000

        float sine_taylor(float in)
        {
            [TODO]: STUDENTS SHOULD WRITE CODE FOR COMPUTING TAYLOR SERIES APPROXIMATION FOR SINE OF INPUT, WITH TAYLOR_COEFFS TERMS.
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
        self.prg = cl.Program(self.ctx, kernel_code).build()

    def deviceSine(self, a, b, length, is_b_a_vector):
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
        # device memory allocation

        # execute operation.
        if(printing_properties == 'No Print'):
            #[TODO: Students to get appropriate compiled kernel]
        elif(printing_properties == 'Print'):
            #[TODO: Students to get appropriate compiled kernel]
        else:
            #[TODO: Students to get appropriate compiled kernel]

        # wait for execution to complete.

        # Copy output from GPU to CPU [Use .get() method]

        # Record execution time.

        # return a tuple of output of addition and time taken to execute the operation.
        pass

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
    all_main_methods = ['CPU Sine', 'deviceSine']
    # List the two operations
    all_operations = ['No Print', 'Print', 'Sync then Print']
    # List the size of vectors
    vector_sizes = 10**np.arange(1,3)
    # List iteration indexes
    iteration_indexes = np.arange(1,3)
    # Select the list of valid operations for profiling
    valid_operations = all_operations
    valid_vector_sizes = vector_sizes
    valid_main_methods = all_main_methods

    # Create an instance of the clModule class
    graphicscomputer = clModule()

    # Nested loop precedence, operations -> vector_size -> iteration -> CPU/GPU method.

    for current_operation in valid_operations:
        #TODO: STUDENTS CAN USE THIS SPACE TO WRITE NECESSARY TIMING ARRAYS, PERSONAL DEBUGGING PRINT STATEMENTS, ETC        
        for vector_size in valid_vector_sizes:
            #TODO: STUDENTS CAN USE THIS SPACE TO WRITE NECESSARY TIMING ARRAYS, PERSONAL DEBUGGING PRINT STATEMENTS, ETC

            #THE FOLLOWING VARIABLE SHOULD NOT BE CHANGED
            a_array_np = 0.001*np.arange(1,vector_size+1).astype(np.float32) #Generates an Array of Numbers 0.001, 0.002, ... 

            for iteration in iteration_indexes:
                #TODO: STUDENTS CAN USE THIS SPACE TO WRITE NECESSARY TIMING ARRAYS, PERSONAL DEBUGGING PRINT STATEMENTS, ETC

                for current_method in valid_main_methods:
                    if(current_method == 'CPU Sine'):
                        #TODO: STUDENTS TO GET OUTPUT TIME AND COMPUTATION FROM CPU_Sine
                    else:
                        if(current_method == 'deviceSine'):
                            #TODO: STUDENTS TO GET OUTPUT TIME AND COMPUTATION FROM sine_device_mem_gpu

                        #TODO: STUDENTS TO COMPARE RESULTS USING ISCLOSE FUNCTION
        #TODO: STUDENTS CAN USE THIS SPACE TO WRITE NECESSARY TIMING ARRAYS, PERSONAL DEBUGGING PRINT STATEMENTS, ETC