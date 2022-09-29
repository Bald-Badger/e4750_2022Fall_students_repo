# EECS E4750: Heterogeneous Computing for Signal and Data Processing (Fall 2022)

## Assignment-1: Introduction to memory management and profiling in PyCUDA and PyOpenCL.

Due date: See in the courseworks.

Total points: 100

### Primer

The goal of the assignment is to compare and contrast the different method(s) of host-to-device memory allocation for simple elementwise operations on vector(s) and introduction
to profiling. The assignment is divided into a programming section, and a theory section. The programming section contains tasks for CUDA, and tasks for OpenCL. 

### Relevant Documentation

For PyOpenCL:
1. [OpenCL Runtime: Platforms, Devices & Contexts](https://documen.tician.de/pyopencl/runtime_platform.html)
2. [pyopencl.array](https://documen.tician.de/pyopencl/array.html#the-array-class)
3. [pyopencl.Buffer](https://documen.tician.de/pyopencl/runtime_memory.html#buffer)

For PyCUDA:
1. [Documentation Root](https://documen.tician.de/pycuda/index.html)
2. [Memory tools](https://documen.tician.de/pycuda/util.html#memory-pools)
3. [gpuarrays](https://documen.tician.de/pycuda/array.html)

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
5. In case you get error like: "device not ready" most likely the error is with synchronization. One possible solution is to run the following code ```pycuda.driver.Context.synchronize()``` BEFORE ```start.time_till(end)```.
You can also use time.time() to record execution time since the difference is usually very small.


## Programming Problem (80 points)

Your submission should contain 3 files.

1. Project Report    : E4750.2022Fall.(uni).assignment1.report.PDF   : In PDF format containing information presented at [Homework-Reports.md](https://github.com/eecse4750/e4750_2022Fall_students_repo/wiki/Homework-Reports) , the plots, print and profiling results, and the answers for theory questions. I recommend using A3 Page template since it gives more space to organize code, plots and print results better.
2. PyCUDA solution   : E4750.2022Fall.(uni).assignment1.PyCUDA.py    : In .py format containing the methods and kernel codes, along with comments.
3. PyOpenCL solution : E4750.2022Fall.(uni).assignment1.PyOpenCL.py  : In .py format containing the methods and kernel codes, along with comments.

Replace (uni) with your uni ID. An example report would be titled E4750.2022Fall.zk2172.assignment1.report.PDF 

### Problem set up

You are given a part of the template code in the main function to iterate between the 1D vectors (Float32 Datatype) having values $(1,2,3...N)$ with **N** taking values of $(10,10^2,10^3...10^8)$ for different CPU/GPU computation scenarios. The different scenarios to iterate in has been written in the form of a nested for loop with the CPU methods part completed. You are expected to follow the template and complete the code for performing GPU computation.

The Programming problem is split up into 4 tasks.

1. 1. Writing Kernel Code for PyCUDA (as part of E4750.2022Fall.(uni).assignment1.PyCUDA.py)
   2. Writing Kernel Code for PyOpenCL (as part of E4750.2022Fall.(uni).assignment1.PyOpenCL.py)
2. 1. Writing Methods to Call Kernel Code for PyCUDA and Plotting results (as part of E4750.2022Fall.(uni).assignment1.PyCUDA.py). Profiling and contrasting the two kernels with one of the methods.
   2. Writing Methods to Call Kernel Code for PyOpenCL and Plotting results (as part of E4750.2022Fall.(uni).assignment1.PyCUDA.py)

You are given two CPU methods (**CPU_numpy_Add** and **CPU_Loop_Add**) to compare your results to.

#### Task - 1: Kernel Code (15 points)

You will write kernel code for both OpenCL and CUDA to perform two operations (Total 4 kernel functions, 2 each for OpenCL and CUDA). One of the kernel code (`Add_two_vectors_GPU` for OpenCL) is given for reference (Check the end of this README file) and you are expected to complete the other three. Each kernel written carries 5 points.

1. *(10 Points)* Create a Kernel Function `Add_to_each_element_GPU` to add a number `b` to each element by passing a single Float 32 number (like `b_number_np` used in CPU code) as one of the arguments. (5 Points for implementing in CUDA, 5 Points for Implementing in OpenCL)
2. *(5 Points)* Create a Kernel Function `Add_two_vectors_GPU` to add a number `b` passing a vector (like `b_array_np` used in CPU Code having each element equal to `b` and of Float 32 type) as one of the arguments.

#### Task - 2.a: PyOpenCL (25 points)

This task involves using the above kernel and managing the host-to-device memory transfers to perform the necessary computations. You will use `is_b_a_vector` as an argument to these methods to decide which kernel to call.

1. *(10 points)* Complete the function (**deviceAdd**) to perform both operations using `pyopencl.array` to load the input arguments to device memory. Compare your results with `CPU_numpy_Add`. Record the time for the execution of the operation, including the memory transfer steps.

2. *(10 points)*  Complete the function (**bufferAdd**) to perform both operations using `pyopencl.Buffer` to load the input arguments to device memory. Compare your results with `CPU_numpy_Add`. Record the time for the execution of the operation, including the memory transfer steps.

3. 1. Call each of the above four GPU cases (2 cases in `pyopencl.array` and 2 cases in `pyopencl.Buffer`) and the two CPU cases one by one, for each iteration, for each length of vector from size $(N = 10,10^2,10^3...10^6)$ and for each operation and record the **average** running time after 50 repetition.

   2. Then extend the analysis for vector sizes $(N = 10,10^2,10^3...10^8)$ excluding only the slower of the CPU case from this analysis. (You should call each of the 4 GPU cases, and the faster of the CPU cases, for a total of 5 cases).
   
   The precedence for the order is operations -> vector_size -> iteration -> CPU/GPU method. (See template code __main__ part).

4. *(5 points)* Plot the **average** execution times for the four GPU cases and the faster of the CPU case against the increasing array size (in orders of **L**) for array sizes $(N = 10,10^2,10^3...10^8)$.

#### Task - 2.b: PyCUDA (40 points)

For PyCUDA, the coding problem will involve your first practical encounter with kernel codes, host-to-device memory transfers (and vice-versa), and certain key classes that PyCUDA provides for them. Read the instructions below carefully and complete your assignment as outlined:

Complete the relevant kernel code (`Add_to_each_element_GPU` and `Add_two_vectors_GPU`), and perform both operations (Adding by passing a single number, and adding by passing a vector having elements of equal value, unless otherwise stated) for each of the memory transfer methods given below. This task involves managing the host-to-device (and vice-versa) memory transfers to perform the necessary computations. You will use `is_b_a_vector` as an argument to these methods to decide which kernel to call.

1. *(10 points)* Write a function (**add_device_mem_gpu**) to perform both operations taking advantage of explicit device memory allocation using `pycuda.driver.mem_alloc()`. Do not forget to retrieve the result from device memory using the appropriate PyCUDA function. Use `SourceModule` to compile the kernel which you defined earlier. Compare your results with `CPU_numpy_Add`. Record the following:
    1. Time taken to execute the operation including memory transfer.
    2. Time taken to execute the operation excluding memory transfer.

2. *(10 points)* Write a function (**add_host_mem_gpu**)  to perform both operations **without** explicit device memory allocation. Use `SourceModule` to compile the kernel which you defined earlier. Compare your results with `CPU_numpy_Add`. Time the execution. Record the time taken to complete the operation including memory transfer.

3. *(4 points)* Write a function (**add_gpuarray_no_kernel**) to perform vector + vector addition (ignore vector + number for this case) using the `gpuarray` class (instead of allocating with `mem_alloc`), and **without using the kernel**. For this problem use numpy like syntax without actually calling the kernel for vector addition. Record the following:
    1. Time taken to execute the operation including memory transfer.
    2. Time taken to execute the operation excluding memory transfer.

4. *(4 points)* Write a function (**add_gpuarray**)  perform vector + vector addition (ignore vector + number for this case) using the `gpuarray` class (instead of allocating with `mem_alloc`) and **using the kernel** (`Add_two_vectors_GPU`). Compare your results with `CPU_numpy_Add`. Record the following:
    1. Time taken to execute the operation including memory transfer.
    2. Time taken to execute the operation excluding memory transfer.

5. 1. Call each of the above eight GPU cases (2 cases in each of the above 4) and two CPU cases one by one, for each iteration, for each length of vector from size $(N = 10,10^2,10^3...10^6)$ and for each operation and record the **average** running time after 50 repetitions. The precedence for the order is (operations -> vector_size -> iteration -> CPU/GPU method. See template code for PyOpenCL).

   2. Then extend the analysis for vector sizes $(N = 10,10^2,10^3...10^8)$ excluding only the slower of the CPU case from this analysis. (You should call each of the 8 GPU cases, and the faster of the CPU cases, for a total of 9 cases).

   The precedence for the order is operations -> vector_size -> iteration -> CPU/GPU method. (See template code __main__ part).

6. *(4 points)* Plot the **average** execution times (including memory transfer for GPU operations) against the increasing array size (in orders of **L**) for array sizes $(N = 10,10^2,10^3...10^8)$.

7. *(4 points)* Plot the **average** execution times (excluding memory transfer for GPU operations) against the increasing array size (in orders of **L**) for array sizes $(N = 10,10^2,10^3...10^8)$.

8. *(4 points)* Profile **add_device_mem_gpu** CUDA functions, for a. `Add_to_each_element_GPU` and b. `Add_two_vectors_GPU` kernels. What differences do you see, if any?

## Theory Problems (20 points)

1. *(3 points)* What is the difference between a thread, a task and a process? (Clarification: Seek definitions in the book and online, both for CPUs and GPUs. Collect the variety of definitions, compare and contrast.)

2. *(3 points)* What are the advantages and disadvantages of using Python over C/C++?

3. 1. *(2 points)* Which of the two CPU method(s) (`CPU_numpy_add` and `CPU_Loop_Add`) is faster. Why is this so?
   2. *(4 points)* In general, how does the parallel approaches in task-1(PyOpenCL) and task-2(PyCuda) compare to the CPU methods? Which is faster and why? [Consider both cases in case of PyCuda, including memory transfer and excluding memory transfer].

4. *(4 points)* Out of the two approaches explored in task-1 (PyOpenCL), which proved to be faster? Explore the PyOpenCL docs and source code to support your conclusions about the differences in execution time.

5. *(4 points)* Of the different approaches explored in task-2 (PyCUDA), which method(s) proved the fastest? Explore the PyCUDA docs and source code and explain how/why: (a) Normal python syntax can be used to perform operations on gpuarrays; (b) gpuarray execution (non-naive method) is comparable to using `mem_alloc`. 

## Code Templates

You **must** adhere to the template given in the code below - this is essential for all assignments to be graded fairly and equally. 

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
```

#### PyCUDA Starter Code

```python
"""
The code in this file is part of the instructor-provided template for Assignment-1, task-2, Fall 2021. 
"""

import relevant.libraries

class deviceAdd:
    def __init__(self):
        """
        Attributes for instance of deviceAdd module
        Includes kernel code and input variables.
        """

        # Compile the kernel code when an instance
        # of this class is made. This way it only
        # needs to be done once for the 3 functions
        # you will call from this class.
        self.mod = self.getSourceModule()

    def getSourceModule(self):
        """
        Compiles Kernel in Source Module to be used by functions across the class.
        """
        # define your kernel below.
        kernelwrapper = """"""
        return SourceModule(kernelwrapper)

    
    def add_device_mem_gpu(self, a, b, length, is_b_a_vector):
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
        # [TODO: Students should write code for the entire method for both cases of is_b_a_vector]

        # Event objects to mark the start and end points

        # Device memory allocation for input and output arrays

        # Copy data from host to device

        # Call the kernel function from the compiled module
        if (is_b_a_vector == True):
            # Use `Add_two_vectors_GPU` Kernel.
        else:
            # Use `Add_to_each_element_GPU` Kernel

        # Get grid and block dim
        
        # Record execution time and call the kernel loaded to the device

        # Wait for the event to complete

        # Copy result from device to the host

        # return a tuple of output of addition and time taken to execute the operation.
        pass

    
    def add_host_mem_gpu(self, a, b, length, is_b_a_vector):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables.
        Arguments:
            a                               :   numpy array of size: length
            b                               :   numpy array of size: length
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """
        # [TODO: Students should write code for the entire method for both cases of is_b_a_vector]
        # Event objects to mark the start and end points

        # Get grid and block dim

        # Call the kernel function from the compiled module
        if (is_b_a_vector == True):
            # Use `Add_two_vectors_GPU` Kernel.
        else:
            # Use `Add_to_each_element_GPU` Kernel
        
        # Record execution time and call the kernel loaded to the device

        # Wait for the event to complete
        
        # return a tuple of output of addition and time taken to execute the operation.
        pass


    def add_gpuarray_no_kernel(self, a, b, length, is_b_a_vector):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables (use gpuarray.to_gpu instead) and WITHOUT calling the kernel. The operation
        is defined using numpy-like syntax. 
        Arguments:
            a                               :   numpy array of size: length
            b                               :   numpy array of size: length
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """
        # [TODO: Students should write code for the entire method. Sufficient to be able to do for is_b_a_vector == True case alone. Bonus points if is_b_a_vector == False case is solved by passing a single number to GPUarray and performing the addition]
        # Event objects to mark start and end points

        # Allocate device memory using gpuarray class        
        
        # Record execution time and execute operation with numpy syntax

        # Wait for the event to complete

        # Fetch result from device to host
        
        # return a tuple of output of addition and time taken to execute the operation.
        pass
        
    def add_gpuarray(self, a, b, length, is_b_a_vector):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables (use gpuarray.to_gpu instead). In this scenario make sure that 
        you call the kernel function.
        Arguments:
            a                               :   numpy array of size: length
            b                               :   numpy array of size: length
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """
        # [TODO: Students should write code for the entire method. Sufficient to be able to do for is_b_a_vector == True case alone. Bonus points if is_b_a_vector == False case is solved by passing a single number to GPUarray and performing the addition]

        # Create cuda events to mark the start and end of array.

        # Get function defined in class defination

        # Allocate device memory for a, b, output of addition using gpuarray class        
        
        # Get grid and block dim

        # Record execution time and execute operation

        # Wait for the event to complete

        # Fetch result from device to host
        
        # return a tuple of output of addition and time taken to execute the operation.
        pass

    def CPU_Add(self, a, b, length, is_b_a_vector):
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
    all_main_methods = ['CPU Add', 'CPU_Loop_Add', 'add_device_mem_gpu', 'add_host_mem_gpu', 'add_gpuarray_no_kernel', 'add_gpuarray_using_kernel']
    # List the two operations
    all_operations = ['Pass Vector and Number', 'Pass Two Vectors']
    # List the size of vectors
    vector_sizes = 10**np.arange(1,9)
    # List iteration indexes
    iteration_indexes = np.arange(1,50)

    # Select the list of valid operations to be run
    valid_operations = all_operations

    # Select the list of valid methods to perform (populate as you complete the methods).
    # Currently in template code only CPU Add and CPU Loop Add are complete.
    valid_main_methods = all_main_methods[0:2]

    # Select the list of valid vector_sizes for current_analysis
    valid_vector_sizes = vector_sizes[0:6]

    # Create an instance of the CudaModule class
    graphicscomputer = CudaModule()

    # Nested loop precedence, operations -> vector_size -> iteration -> CPU/GPU method.
    # There are four nested loops, the main loop iterates between performing vector + number, and performing vector + vector cases.
    # The second loop iterates between different vector sizes, for each case of the main loop.
    # The third loop runs 50 repetitions, for each case of the second loop
    # The fourth loop iterates between the different CPU/GPU/Memory-transfer methods, for each case of the third loop.

    for current_operation in valid_operations:
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

            a_array_np = np.arange(1,vector_size+1).astype(np.float32)
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
                    if(current_method == 'CPU Add'):
                        c_np_cpu_add, cpu_time_add = graphicscomputer.CPU_Add(a_array_np,b_in,vector_size,is_b_a_vector)
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

        print(current_operation + " The CPU times are")
        print(arr_avg_total_cpu_time)
        print(current_operation + " The CPU Loop times are")
        print(arr_avg_total_cpu_loop_time)
        # [TODO: Students should write Code]
        # Add for the rest of the methods
        # Code for Plotting the results (the code for plotting can be skipped, if the student prefers to have a separate code for plotting, or to use a different software for plotting)
```
