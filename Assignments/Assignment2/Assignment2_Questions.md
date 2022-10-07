# EECS E4750: Heterogeneous Computing for Signal and Data Processing (Fall 2022)

## Assignment-2: Modular Code, Compile Time Arguments and Divergence Analysis

Due date: See in the courseworks.

Total points: 100

### Primer

The goal of this assignment is to get students familiar with programming in a modular way, using built in math libraries, generating device functions, and using print statements. The assignment is divided into a Programming Section (for CUDA and OpenCL) and Theory Section.

### Relevant Documentation

1. [Preprocessor Directives and Macros](https://www.informit.com/articles/article.aspx?p=1732873&seqNum=13)
2. [Printing in C](https://cplusplus.com/reference/cstdio/printf/)
3. [Python Raw Strings](https://www.pythontutorial.net/python-basics/python-raw-strings/)
4. [Taylor Series](https://people.math.sc.edu/girardi/m142/handouts/10sTaylorPolySeries.pdf)
5. [Taylor Series Video, if interested](https://www.youtube.com/watch?v=3d6DsjIBzJ4)
6. [CUDA Math Library](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE)

For PyOpenCL:
1. [OpenCL Runtime: Platforms, Devices & Contexts](https://documen.tician.de/pyopencl/runtime_platform.html)
2. [pyopencl.array](https://documen.tician.de/pyopencl/array.html#the-array-class)

For PyCUDA:
1. [Documentation Root](https://documen.tician.de/pycuda/index.html)
2. [Memory tools](https://documen.tician.de/pycuda/util.html#memory-pools)
3. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

Additional Readings (if interested):
1. [Floating Point Accuracy](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
2. [Kahan Summation](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)

### Hints

Few things which can help you in this homework. 

Consult the [git wiki page](https://github.com/eecse4750/e4750_2022Fall_students_repo/wiki) for relevant tutorials.

1. Synchronization:
    1. There are two ways to synchronize threads across blocks in PyCuda:
        1. Using pycuda.driver.Context.synchronize()
        2. Using CUDA Events. Usually using CUDA Events is a better way to synchronize, for details you can go through: [https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/] , its a short interesting read.
            1. For using CUDA events you will get an instance of cuda event using cuda.event(), example: event = cuda.event().
            2. You will also time cuda events by recording particular time instances using event.record().
            3. You will synchronize all threads in an event using event.synchronize().
            4. For example you can refer to a brief part of assignment solution I showed during my recitation :). There is a minor issue with that code though but the idea is correct.
            5. Note: you need to synchronize event.record() too.
            6. You will use cuda events time_till to record execution time. 
    2. To Synchronize PyOpenCL kernels you have to use kernel.wait() functionality. PyOpenCL kernels are by default executed as an event afaik.
2. For examples related to PyOpenCL please refer to https://github.com/HandsOnOpenCL/Exercises-Solutions. However there is no such comprehensive list for PyCuda.
3. Note: Sometimes synchronization is not required because of the order of operations you keep, consider the following example:
4. Consider a case in which a kernel call is followed by an enqueue_copy call/ memcpy call from device to host,  in this case you can leave wait()/event.synchronize() out because the kernel call and the copy function call are enqueued in the proper sequence. Also generally copy from device to host call is blocking on host unless you explicitly use async copy in which case you will have to synchronize.
5. In case you get error like: "device not ready" most likely the error is with synchronization.
You can also use time.time() to record execution time since the difference is usually very small.


## Programming Problem (80 points)

All the timing, and plots should be taken from running the code in the Cloud Machine. DONOT produce analysis on personal machines.

Your submission should contain 3 files.

1. Project Report    : E4750.2022Fall.(uni).assignment2.report.PDF   : In PDF format containing information presented at [Homework-Reports.md](https://github.com/eecse4750/e4750_2022Fall_students_repo/wiki/Homework-Reports) , the plots, print and profiling results, and the answers for theory questions. I recommend using A3 Page template since it gives more space to organize code, plots and print results better.
2. PyCUDA solution   : E4750.2022Fall.(uni).assignment2.PyCUDA.py    : In .py format containing the methods and kernel codes, along with comments.
3. PyOpenCL solution : E4750.2022Fall.(uni).assignment2.PyOpenCL.py  : In .py format containing the methods and kernel codes, along with comments.

Replace (uni) with your uni ID. An example report would be titled E4750.2022Fall.zk2172.assignment2.report.PDF 

### Problem set up

You are given a part of the template code in the main function to iterate between the 1D vectors (Float32 Datatype) having values $(0.001,0.002,0.003...0.001*N)$ with **N** taking values of $(10,10^2,10^3...10^9)$ for different CPU/GPU computation scenarios. The different scenarios to iterate in has been written in the form of a nested for loop with the CPU methods part completed. You are expected to follow the template and complete the code for performing GPU computation.

The programming section contains two tasks.
1. Writing Kernel Code (For PyOpenCL and PyCUDA)
2. Running the Code and Analysing (including Plotting)

You are given a CPU method (**CPU_Sine**) to compare your results to. Please check the end of this README file for the template. It is recommended to refer to the pre filled parts of CUDA Template to supplement with the question context, and implement a similar structure for OpenCL.

#### Task - 1: Kernel Code (50 points)

You will be using python string concatenation to compile multiple kernel, as shown in the template code. The task is to complete the modular parts of kernel as described

1. *(10 CUDA + 10 OpenCL points = 20 Points)* Create a python string `kernel_main_wrapper` and define a kernel function `main_function` inside it. This main function should
    1. take in an input vector, and
        1. for even index of the input vector, use CUDA/OpenCL built in math functions to compute sine of input argument.
        2. for odd index of the input vector, use an user defined device function `sine_taylor` (described in next sub task) for computing sine of input argument. 
    2. Implement print statements of the form (**Hello from index <array_index>**) encapsulated inside compile_time arguments as mentioned in the CUDA Template (One example print statement has already been implemented in the template code). The OpenCL Kernel should have the same print statements in the same locations.

    Out of the 10 points (for both CUDA and OpenCL case), the split up is 5 Points for Task 1.1 (getting the computation parts right) and 5 Points for Task 1.2 (getting the printing parts inside conditional compile right)

2. *(15 CUDA + 15 OpenCL points = 30 Points)* Create a python string `kernel_device` and define a kernel function `sine_taylor` (accepting a float input and returning a float datatype) that computes the sine of input using taylor series approximation upto Q terms, where Q is given as a compile time argument TAYLOR_COEFFS. (Upto the term with $x^(2Q -1) \over (2Q -1)!$.


    Out of the 15 points (for both CUDA and OpenCL case), the split up is 5 Points for variable declaration, 5 points for computation of individual terms, and 5 points for getting final sum of all terms.

#### Task - 2: Analysis (30 points)

This task involves using appropriate compiled kernel to perform different operations in both PyCUDA and PyOpenCL. Complete the GPU methods using explicit memory allocation (using `pycuda.driver.mem_alloc()` in `sine_device_mem_gpu` and `pyopencl.array.to_device` in `deviceSine`). Do not forget to retrieve the result from device memory using the appropriate functions. You will use the variable named `printing_properties` to choose the appropriate kernel to run (look at CUDA Template, end of getSourceModule method and inside the sine_device_mem_gpu method for reference) in both PyOpenCL and PyCUDA. Each sub division carries 5 points (If only one of PyCUDA or PyOpenCL is performed, only 3 out of 5 marks will be awarded in each case).

1. *(5 Points)* For array Sizes $(N = 10,10^2)$ use the kernel compiled in self.module_with_print_nosync (or equivalent in OpenCL) to make the sinusoid computations in GPU, and observe the print messages. Do you see any pattern? Describe the pattern. Why do you think this is so?
2. *(5 Points)* For array Sizes $(N = 10,10^2)$ use the kernel compiled in self.module_with_print_with_sync to make the sinusoid computations in GPU, and observe the print messages. Do you see any pattern? Is it the same as the previous case? Why do you think this is so?
3. *(5 Points)* For array Sizes $(N = 10,10^2,10^3...10^4)$ use the kernel compiled in self.module_no_print to make the sinusoid computations in GPU and time the execution including memory copy. Compare with CPU results (using CPU function in template code). (Use 50 iterations in the main code and take the average). You may use numpy's isclose function for comparing the results.
4. *(5 Points)* Change the sine_taylor function to compute for 5 taylor series terms by modifying the `kernel_device` function (Change to #define TAYLOR_COEFFS 5) For array Sizes $(N = 10,10^2,10^3...10^6)$ use the kernel compiled in self.module_no_print to make the sinusoid computations in GPU and time the execution including memory copy. Compare with CPU results (using CPU function in template code). (Use 50 iterations in the main code and take the average)
5. *(5 CUDA + 5 OpenCL Points = 10 Points)* Plot timing results from GPU with 10000 Taylor Series Terms, GPU with 5 Taylor Series terms and CPU for array Sizes $(N = 10,10^2,10^3...10^8,10^9)$

## Theory Problems (20 points)

1. *(5 points)* Cuda provides a "syncthreads" method, explain what is it and where is it used? Give an example for its application? Consider the following kernel code for doubling each vector, will syncthreads be helpful here?

```
__global__ void doublify(float *c_d, const float *a_d, const int len) {
        int t_id =  blockIdx.x * blockDim.x + threadIdx.x;
        c_d[t_id] = a_d[t_id]*2;
        __syncthreads();
}

```

2. *(5 points)* Briefly explain the difference between private memory, local memory & global memory. What happens when you use too much private memory?

3. *(5 points)* Describe two methods of computing sine of a non trivial angle (say 11.13 degrees) in a computer algorithm, and make an assessment of their computational performance both in terms of speed and accuracy? Refer to online resources to support your answer. Do you think it is possible to have an exact answer represented on a computer?

4. *(5 points)* For a vector addition, assume that the vector length is 7500, each thread calculates one output element, and the thread block size is 512 threads. The programmer configures the kernel launch to have a minimal number of thread blocks to cover all output elements. How many threads will be in the grid? 

## Code Templates

You **must** adhere to the template given in the code below - this is essential for all assignments to be graded fairly and equally. 

#### PyCUDA Starter Code

```python
"""
The code in this file is part of the instructor-provided template for Assignment-1, task-2, Fall 2021. 
"""

import relevant.libraries

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
                [TODO]: STUDENTS SHOULD WRITE CODE TO USE CUDA MATH FUNCTION TO COMPUTE SINE OF INPUT VALUE
                #ifdef PRINT_ENABLE_DEBUG
                if(idx<n)
                {
                    printf("Hello from index %d \n", idx);
                }
                #endif
            }
            else{
                [TODO]: STUDENTS SHOULD WRITE CODE TO CALL THE DEVICE FUNCTION sine_taylor TO COMPUTE SINE OF INPUT VALUE
                #ifdef PRINT_ENABLE_DEBUG
                if(idx<n)
                {
                    [TODO]: STUDENTS SHOULD WRITE CODE TO PRINT THE INDEX OF THE ARRAY BEING COMPUTED
                }
                #endif
            }

            #ifdef PRINT_ENABLE_AFTER_COMPUTATION
            if(idx<n)
            {
                [TODO]: STUDENTS SHOULD WRITE CODE TO PRINT THE INDEX OF THE ARRAY BEING COMPUTED
            }
            #endif     
        }
        """

        kernel_device = """
        #define TAYLOR_COEFFS 10000

        __device__ float sine_taylor(float in)
        {
            [TODO]: STUDENTS SHOULD WRITE CODE FOR COMPUTING TAYLOR SERIES APPROXIMATION FOR SINE OF INPUT, WITH TAYLOR_COEFFS TERMS.
        }
        """

        # Compile kernel code and store it in self.module_*

        self.module_no_print = SourceModule(kernel_device + kernel_main_wrapper)
        self.module_with_print_nosync = SourceModule(kernel_printer + kernel_device + kernel_main_wrapper)
        self.module_with_print_with_sync = SourceModule(kernel_printer_end + kernel_device + kernel_main_wrapper)

        # SourceModule is the Cuda.Compiler and the kernelwrapper text is given as input to SourceModule. This compiler takes in C code as text inside triple quotes (ie a string) and compiles it to CUDA code.
        # When we call this getSourceModule method for an object of this class, it will return the compiled kernelwrapper function, which will now take inputs along with block_specifications and grid_specifications.
    
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

        # Event objects to mark the start and end points

        # Device memory allocation for input and output arrays

        # Copy data from host to device

        # Call the kernel function from the compiled module
        if(printing_properties == 'No Print'):
            prg = self.module_no_print.get_function("main_function")
        elif(printing_properties == 'Print'):
            prg = self.module_with_print_nosync.get_function("main_function")
        else:
            prg = self.module_with_print_with_sync.get_function("main_function")

        # Get grid and block dim
        
        # Record execution time and call the kernel loaded to the device

        # Wait for the event to complete

        # Copy result from device to the host

        # return a tuple of output of sine computation and time taken to execute the operation.
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
    all_main_methods = ['CPU Sine', 'Sine_device_mem_gpu']
    # List the two operations
    all_operations = ['No Print', 'Print', 'Sync then Print']
    # List the size of vectors
    vector_sizes = 10**np.arange(1,3)
    # List iteration indexes
    iteration_indexes = np.arange(1,50)
    # Select the list of valid operations for profiling
    valid_operations = all_operations

    # Create an instance of the clModule class
    graphicscomputer = CudaModule()

    for current_operation in valid_operations:
        #TODO: STUDENTS CAN USE THIS SPACE TO WRITE NECESSARY TIMING ARRAYS, PERSONAL DEBUGGING PRINT STATEMENTS, ETC        
        for vector_size in vector_sizes:
            #TODO: STUDENTS CAN USE THIS SPACE TO WRITE NECESSARY TIMING ARRAYS, PERSONAL DEBUGGING PRINT STATEMENTS, ETC

            #THE FOLLOWING VARIABLE SHOULD NOT BE CHANGED
            a_array_np = 0.001*np.arange(1,vector_size+1).astype(np.float32) #Generates an Array of Numbers 0.001, 0.002, ... 

            for iteration in iteration_indexes:
                #TODO: STUDENTS CAN USE THIS SPACE TO WRITE NECESSARY TIMING ARRAYS, PERSONAL DEBUGGING PRINT STATEMENTS, ETC

                for current_method in all_main_methods:
                    if(current_method == 'CPU Sine'):
                        #TODO: STUDENTS TO GET OUTPUT TIME AND COMPUTATION FROM CPU_Sine
                    else:
                        if(current_method == 'Sine_device_mem_gpu'):
                            #TODO: STUDENTS TO GET OUTPUT TIME AND COMPUTATION FROM sine_device_mem_gpu

                        #TODO: STUDENTS TO COMPARE RESULTS USING ISCLOSE FUNCTION
        #TODO: STUDENTS CAN USE THIS SPACE TO WRITE NECESSARY TIMING ARRAYS, PERSONAL DEBUGGING PRINT STATEMENTS, ETC
```
#### PyOpenCL Starter Code *(with kernel code)*

**Note:** The kernel code is only provided for OpenCL, and for the sole reason that this is the first assignment of the course.  

```python
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
```
