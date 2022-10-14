# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 16:03:20 2021

@author: manueljenkin
"""

"""
The code in this file is part of the instructor-provided template for Assignment-1, task-2, Fall 2021. 
"""

import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import sys

import time

class CudaModule:
    def __init__(self):
        """
        Attributes for instance of deviceAdd module
        Includes kernel code and input variables.
        """
        self.threads_per_block_x = 1024 # Our current input is 1D so it makes sense to assume a 1D thread block. (other two dimensions are 1)
        self.threads_per_block_y = 1
        self.threads_per_block_z = 1
        self.threads_total = self.threads_per_block_x * self.threads_per_block_y * self.threads_per_block_z

        # Whenever an instance of deviceAdd is created, the init function runs as soon as creation happens.
        # The instance of deviceAdd we have created here is named graphicscomputer - check the main code for reference.
        # self.mod, self.threads_per_block_x, etc are present for the whole class and this can be used to refer to these parameters anywhere inside the class
        # These parameters are also available outside as an attribute to the object instance we have created. They can be accessed by graphicscomputer.mod, graphicscomputer.threads_per_block_x outside the class.
        # As we see below, this instance now has methods - getSourceModule, explicitAdd, implicitAdd, gpuarrayAdd_np, gpuarrayAdd and numpyAdd that we can call.
        # In the main code, graphicscomputer.explicitAdd(a_np, b_np, vectorlength) calls the explicitAdd method for that instance and passes (a_np, b_np, vectorlength)
        # It also passes the pointer to graphicscomputer and its attributes, which inside this class will be called by self.attributes (eg: graphicscomputer.mod will be called using self.mod inside the class) to this method.

    def getSourceModule(self):
        """
        Compiles Kernel in Source Module to be used by functions across the class.
        """
        # define your kernel below.
        # a and b denote input, c is output, and n is the length of the vector.
        # Bounds Checking: (idx < n) condition is required to ensure that the extra threads of the final block donot get computed or appended to the output c.
        kernelwrapper = """
        __global__ void Add_to_each_element_GPU(float *a, float *b, float *c, int n)
        {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx < n) c[idx] = a[idx] + b[0];
        }

        __global__ void Add_two_vectors_GPU(float *a, float *b, float *c, int n)
        {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx < n) c[idx] = a[idx] + b[idx];
        }
        """
        return SourceModule(kernelwrapper)
        # SourceModule is the Cuda.Compiler and the kernelwrapper text is given as input to SourceModule. This compiler takes in C code as text inside triple quotes (ie a string) and compiles it to CUDA code.
        # When we call this getSourceModule method for an object of this class, it will return the compiled kernelwrapper function, which will now take inputs a, b, c, n, along with block_specifications and grid_specifications.

    def add_device_mem_gpu(self, a, b, length, is_b_a_vector):
        """
        Function to perform on-device parallel vector addition
        by explicitly allocating device memory for host variables.
        Arguments:
            a                               :   numpy array of size: length
            b                               :   number or vector of equal numbers with same length as a
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """
        # To this function or method explicitAdd of class deviceAdd the value of a, b and length are passed by value.
        # A new copy is made here in main memory (or host memory), apart from the earlier one made in the main code, for a total of two copies in main memory. (Check main code for reference)
        l = np.intc(length) # we pass the length directly as a numpy int

        # Event objects to mark the start and end points
        start_memcpy = cuda.Event()
        start_computation = cuda.Event()
        end_computation = cuda.Event()
        end_memcpy = cuda.Event()

        # We create two pycuda.driver.Event (cuda here is pycuda.driver - check beginning of code for import commands) instances.
        # One instance is named "start" and another is named "end". The name is upto our choice. Both "start" and "end" now has methods for recording which it got from pycuda.driver.Event().
        # We will call this record method later (within this explicitAdd method). First record will start timer, second record will end timer.
        prg = self.getSourceModule()

        # Device memory allocation for input and output arrays
        a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize) # allocating memory inside the gpu (locations stored in a_gpu and b_gpu) for storing the input vectors (a and b)
        b_gpu = cuda.mem_alloc(b.size * b.dtype.itemsize) 
        c_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize) # allocating memory inside the gpu for storing the output after computation.
        # #l_gpu = cuda.mem_alloc(l.size * l.dtype.itemsize) # We are not allocating memory for passing value l. We pass it directly as a numpy intc type for the kernel function.

        # Copy data from host to device, and set timer for including copy from CPU to device
        start_memcpy.record() # calling the record function to record time. Start is an instace of pycuda.driver.Event. Now this will store this time as an attribute to this variable named start_memcpy!
        cuda.memcpy_htod(a_gpu, a) # copying the local variable a in the explicitAdd function (or method) to a_gpu inside the GPU (we allocated this memory inside GPU in the earlier lines)
        cuda.memcpy_htod(b_gpu, b) # copying the local variable b in the explicitAdd function to b_gpu inside the GPU.
        # #cuda.memcpy_htod(l_gpu, l) # Reason above. We donot create l_gpu, instead we pass directly l as np intc type to kernel function.
        # Call the kernel function from the compiled module
        if(is_b_a_vector == False):
            func = prg.get_function("Add_to_each_element_GPU")
        else:
            func = prg.get_function("Add_two_vectors_GPU")
        # the current explicitAdd method (of which the current portion of the code is a part of), is a method defined for deviceAdd Class.
        # self.mod will call self.getSourceModule() method defined for this deviceAdd class (check init method for this deviceAdd Class)
        # this self.getSourceModule() method has the kernel defined inside kernelwrapper and returns the compiled kernel function for GPU.

        # Get grid and block dim
        griddimensions = (int(np.ceil(length/self.threads_total)),1) # A grid is a collection of blocks. And a block is a collection of threads.
        blockdimensions = (self.threads_per_block_x,self.threads_per_block_y,self.threads_per_block_z) # Each block has a 3-D arrangement of threads that can be called by their index in x,y and z directions.

        # Record execution time and call the kernel loaded to the device
        start_computation.record() # timer for excluding memory copy (just the computation part)
        event = func(a_gpu, b_gpu, c_gpu, l, block=blockdimensions, grid=griddimensions) #this event is different from cuda.event. We just named it event and we are calling a function here.
        # It is not even necessary to name it and enough to just directly call func. (refer to other methods defined for this class like implicitAdd where we call func directly without naming it to another variable like we did here).
        # Wait for the event to complete
        end_computation.record() # Executes after the above function is finished. Calling the record function for end variable which is another instance of pycuda.driver.Event. This will store this time, as an attribute to this variable named end.
        # Note that in this case, start, event = func(...) and end are cuda library based commands. It is reasonable to expect all of these to execute sequentially (although the content inside func executes parallel).

        # Copy result from device to the host
        c = np.empty_like(a) # Allocating memory in host to copy the result from GPU memory. This again is a local copy (stored in host memory) inside this method.
        # We will return this value to another variable, again in host memory (see the return statement for this explicitAdd method and check the main code). We will have two copies of this in host memory after the full task is over.
        cuda.memcpy_dtoh(c, c_gpu) # copying result from GPU memory to host memory allocated for local variable.
        
        end_memcpy.record()
        cuda.Context.synchronize()

        time_taken_memcopy = start_memcpy.time_till(end_memcpy) # getting current time subtracting time_start captured before memory copy to GPU.
        # I am doing this time calculation first (before computing gpu running time) to ensure the time measured is only for the total operation activity involving memory copy and GPU computation.
        # The GPU times are already stored in the attributes to variables start and end, computation of this time difference might take a few cycles and those would otherwise get included in the time calculation.
        time_taken_nomemcopy = start_computation.time_till(end_computation) # start is a pycuda.driver.Event instance. It has a method time_till which gets an input which is also a pycuda.driver.event instance (here the other instance is named end).
        # After this it computes time difference between these two events using the time attributes that got stored in start and end when using the record() method.
        
        # return a tuple of output of addition and time taken to execute the operation.
        return (c, time_taken_nomemcopy, time_taken_memcopy) # returning the results to host memory located for variable in main program.
        
    def add_host_mem_gpu(self, a, b, length, is_b_a_vector):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables.
        Arguments:
            a                               :   numpy array of size: length
            b                               :   number or vector of equal numbers with same length as a
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """
        l = np.intc(length)
        # Event objects to mark the start and end points
        start_memcpy = cuda.Event()
        end_memcpy = cuda.Event()
        prg = self.getSourceModule()

        # Get grid and block dim
        griddimensions = (int(np.ceil(length/self.threads_total)),1)
        blockdimensions = (self.threads_per_block_x,self.threads_per_block_y,self.threads_per_block_z)

        # Call the kernel function from the compiled module
        if(is_b_a_vector == False):
            func = prg.get_function("Add_to_each_element_GPU")
        else:
            func = prg.get_function("Add_two_vectors_GPU")
        c = np.empty_like(a) # creating memory in host

        # Record execution time and call the kernel loaded to the device
        start_memcpy.record()
        func (cuda.In(a),cuda.In(b),cuda.Out(c), l, block=blockdimensions, grid=griddimensions)
        # Here, no explicit memory allocation is done. Rest of the function is same, except how we pass the variables.
        # Earlier, in explicitAdd method, we created memory location in GPU (using pycuda.driver.mem_alloc), stored values (using pycuda.driver.memcpy_htod)
        # We then passed this location to the function. Here, pycuda.driver has methods in, out and inout that can directly take in values or locations from host memory.
        # We pass variables a and b [TODO: Does it pass the location or value of a and b?] and location of c present in host memory through this in, out and inout methods.
        # It does automatic GPU memory allocation and finish the process. It returns the value to c in host memory at the end of the process.

        # Wait for the event to complete
        end_memcpy.record()
        cuda.Context.synchronize()
        time_taken_memcopy = start_memcpy.time_till(end_memcpy)
        # time_taken_memcopy = start.time_till(end)
        # GPU computation runtime including and without including memory copy time cannot be differentiated here since both are combined into a single line of code here.

        # return a tuple of output of addition and time taken to execute the operation.
        return (c,time_taken_memcopy)

    def add_gpuarray_no_kernel(self, a, b, length, is_b_a_vector):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables (use gpuarray.to_gpu instead) and WITHOUT calling the kernel. The operation
        is defined using numpy-like syntax. 
        Arguments:
            a                               :   numpy array of size: length
            b                               :   number or vector of equal numbers with same length as a
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """
        l = np.intc(length)
        # Event objects to mark start and end points
        start_memcpy = cuda.Event()
        start_computation = cuda.Event()
        end_computation = cuda.Event()
        end_memcpy = cuda.Event()
        # Allocate device memory using gpuarray class 
        start_memcpy.record()       
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        # a_gpu and b_gpu instances of pycuda.gpuarray and the to_gpu method of this class helps allocate memory pass value of a and b to a_gpu and b_gpu
        # Here we donot explicitly create a kernel function and use the add operation defined for instances of the pycuda.gpuarry class.
        # pycuda supports an add function on these two instances (method for adding these two objects is defined) which we will be using for this gpuarrayAdd_np method.

        # Record execution time and execute operation with numpy syntax
        start_computation.record()
        c_gpu = (a_gpu + b_gpu)
        end_computation.record()
        # a_gpu and b_gpu are gpuarray instances, and the + operation has a method defined for these with pycuda library.
        # it will execute and store results in c_gpu which will also be an instance of pycuda.gpuarray.
       
        # Wait for the event to complete

        # Fetch result from device to host
        c = c_gpu.get() # c_gpu is still in gpu present as an instance of pycuda.gpuarray class. In this step we are passing its value to c, which is in host memory.
        # The method get() for pycuda.gpuarray class (of which c_gpu is an instance of) does both memory allocation in host accessible using variable c and storing c_gpu value there.
        # time_taken_memcopy = (time.clock() - time_start)
        end_memcpy.record()
        cuda.Context.synchronize()
        time_taken_memcopy = start_memcpy.time_till(end_memcpy)
        time_taken_nomemcopy = start_computation.time_till(end_computation)

        # return a tuple of output of addition and time taken to execute the operation.
        return (c, time_taken_nomemcopy, time_taken_memcopy)
        
    def add_gpuarray(self, a, b, length, is_b_a_vector):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables (use gpuarray.to_gpu instead). In this scenario make sure that 
        you call the kernel function.
        Arguments:
            a                               :   numpy array of size: length
            b                               :   number or vector of equal numbers with same length as a
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """
        l = np.intc(length)
        c = np.empty_like(a)
        # Create cuda events to mark the start and end of array.
        start_memcpy = cuda.Event()
        start_computation = cuda.Event()
        end_computation = cuda.Event()
        end_memcpy = cuda.Event()

        prg = self.getSourceModule()

        # Get function defined in class definition
        if(is_b_a_vector == False):
            func = prg.get_function("Add_to_each_element_GPU")
        else:
            func = prg.get_function("Add_two_vectors_GPU")

        # Allocate device memory for a, b, output of addition using gpuarray class        
        start_memcpy.record()
        a_gpu = gpuarray.to_gpu(a)
        b_gpu = gpuarray.to_gpu(b)
        c_gpu = gpuarray.to_gpu(c)

        # Get grid and block dim
        griddimensions = (int(np.ceil(length/self.threads_total)),1)
        blockdimensions = (self.threads_per_block_x,self.threads_per_block_y,self.threads_per_block_z)

        # Record execution time and execute operation
        start_computation.record()
        event = func(a_gpu, b_gpu, c_gpu, l, block=blockdimensions, grid=griddimensions)

        # Wait for the event to complete
        end_computation.record()

        # Fetch result from device to host
        c = c_gpu.get()
        end_memcpy.record()
        cuda.Context.synchronize()
        time_taken_memcopy = start_memcpy.time_till(end_memcpy)
        time_taken_nomemcopy = start_computation.time_till(end_computation)
        
        # return a tuple of output of addition and time taken to execute the operation.
        return (c, time_taken_nomemcopy, time_taken_memcopy)

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
    all_main_methods = ['CPU Add', 'CPU_Loop_Add', 'add_device_mem_gpu', 'add_host_mem_gpu', 'add_gpuarray_no_kernel', 'add_gpuarray_using_kernel']
    # List the two operations
    all_operations = ['Pass Vector and Number', 'Pass Two Vectors']
    # List the size of vectors
    vector_sizes = 10**np.arange(1,9)
    # List iteration indexes
    iteration_indexes = np.arange(1,50)
    # Select the list of valid operations for profiling
    valid_operations = all_operations

    # Create an instance of the clModule class
    graphicscomputer = CudaModule()

    for current_operation in valid_operations:
        arr_avg_total_cpu_time = np.array([])
        arr_avg_total_cpu_loop_time = np.array([])
        arr_avg_total_device_mem_gpu_time_nomemcpy = np.array([])
        arr_avg_total_device_mem_gpu_time = np.array([])
        arr_avg_total_host_mem_gpu_time = np.array([])
        arr_avg_total_gpuarray_no_kernel_time_nomemcpy = np.array([])
        arr_avg_total_gpuarray_no_kernel_time = np.array([])
        arr_avg_total_gpuarray_time_nomemcpy = np.array([])
        arr_avg_total_gpuarray_time = np.array([])
        
        for vector_size in vector_sizes:

            arr_total_cpu_time = np.array([])
            arr_total_cpu_loop_time = np.array([])
            arr_total_device_mem_gpu_time_no_memcpy = np.array([])
            arr_total_device_mem_gpu_time = np.array([])
            arr_total_host_mem_gpu_time = np.array([])
            arr_total_gpuarray_no_kernel_time_no_memcpy = np.array([])
            arr_total_gpuarray_no_kernel_time = np.array([])
            arr_total_gpuarray_time_no_memcpy = np.array([])
            arr_total_gpuarray_time = np.array([])

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
                        if(current_method == 'add_device_mem_gpu'):
                            c_gpu_device_add, nomemcpy_time, memcpy_time = graphicscomputer.add_device_mem_gpu(a_array_np,b_in,vector_size,is_b_a_vector) 
                            sum_diff = c_gpu_device_add - c_np_cpu_add
                            arr_total_device_mem_gpu_time_no_memcpy = np.append(arr_total_device_mem_gpu_time_no_memcpy, nomemcpy_time)
                            arr_total_device_mem_gpu_time = np.append(arr_total_device_mem_gpu_time, memcpy_time)
                        if(current_method == 'add_host_mem_gpu'):
                            c_gpu_buffer_add, memcpy_time = graphicscomputer.add_host_mem_gpu(a_array_np,b_in,vector_size,is_b_a_vector)
                            sum_diff = c_gpu_buffer_add - c_np_cpu_add
                            arr_total_host_mem_gpu_time = np.append(arr_total_host_mem_gpu_time, memcpy_time)
                        if(current_method == 'add_gpuarray_no_kernel'):
                            c_gpu_buffer_add, nomemcpy_time, memcpy_time = graphicscomputer.add_gpuarray_no_kernel(a_array_np,b_array_np,vector_size,is_b_a_vector)
                            sum_diff = c_gpu_buffer_add - c_np_cpu_add
                            arr_total_gpuarray_no_kernel_time_no_memcpy = np.append(arr_total_gpuarray_no_kernel_time_no_memcpy,nomemcpy_time)
                            arr_total_gpuarray_no_kernel_time = np.append(arr_total_gpuarray_no_kernel_time, memcpy_time)
                        if(current_method == 'add_gpuarray_using_kernel'):
                            c_gpu_buffer_add, nomemcpy_time, memcpy_time = graphicscomputer.add_gpuarray(a_array_np,b_array_np,vector_size,is_b_a_vector)
                            sum_diff = c_gpu_buffer_add - c_np_cpu_add
                            arr_total_gpuarray_time_no_memcpy = np.append(arr_total_gpuarray_time_no_memcpy,nomemcpy_time)
                            arr_total_gpuarray_time = np.append(arr_total_gpuarray_time, memcpy_time)
                        total_diff = sum_diff.sum()
                        if (total_diff != 0):
                            print (current_method + " " + current_operation + "sum mismatch")
                            print (total_diff)
            avg_total_cpu_time = ((arr_total_cpu_time.sum())/50)
            arr_avg_total_cpu_time = np.append(arr_avg_total_cpu_time, avg_total_cpu_time)
            avg_total_cpu_loop_time = ((arr_total_cpu_loop_time.sum())/50)
            arr_avg_total_cpu_loop_time = np.append(arr_avg_total_cpu_loop_time, avg_total_cpu_loop_time)
            avg_total_device_mem_gpu_time_nomemcpy = ((arr_total_device_mem_gpu_time_no_memcpy.sum())/50)
            arr_avg_total_device_mem_gpu_time_nomemcpy = np.append(arr_avg_total_device_mem_gpu_time_nomemcpy,avg_total_device_mem_gpu_time_nomemcpy)
            avg_total_device_mem_gpu_time = ((arr_total_device_mem_gpu_time.sum())/50)
            arr_avg_total_device_mem_gpu_time = np.append(arr_avg_total_device_mem_gpu_time, avg_total_device_mem_gpu_time)
            avg_total_host_mem_gpu_time = ((arr_total_host_mem_gpu_time.sum())/50)
            arr_avg_total_host_mem_gpu_time = np.append(arr_avg_total_host_mem_gpu_time, avg_total_host_mem_gpu_time)
            avg_total_gpuarray_no_kernel_time_nomemcpy = ((arr_total_gpuarray_no_kernel_time_no_memcpy.sum())/50)
            arr_avg_total_gpuarray_no_kernel_time_nomemcpy = np.append(arr_avg_total_gpuarray_no_kernel_time_nomemcpy,avg_total_gpuarray_no_kernel_time_nomemcpy)
            avg_total_gpuarray_no_kernel_time = ((arr_total_gpuarray_no_kernel_time.sum())/50)
            arr_avg_total_gpuarray_no_kernel_time = np.append(arr_avg_total_gpuarray_no_kernel_time, avg_total_gpuarray_no_kernel_time)
            avg_total_gpuarray_time_nomemcpy = ((arr_total_gpuarray_time_no_memcpy.sum())/50)
            arr_avg_total_gpuarray_time_nomemcpy = np.append(arr_avg_total_gpuarray_time_nomemcpy,avg_total_gpuarray_time_nomemcpy)
            avg_total_gpuarray_time = ((arr_total_gpuarray_time.sum())/50)
            arr_avg_total_gpuarray_time = np.append(arr_avg_total_gpuarray_time, avg_total_gpuarray_time)
        print(" The CPU times are")
        print(arr_avg_total_cpu_time)
        print(" The CPU Loop times are")
        print(arr_avg_total_cpu_loop_time)
        print(" The add_device_mem_gpu times are")
        print(arr_avg_total_device_mem_gpu_time_nomemcpy)
        print(" The add_device_mem_gpu times with memcpy are")
        print(arr_avg_total_device_mem_gpu_time)
        print(" The add_host_mem_gpu times with memcpy are")
        print(arr_avg_total_host_mem_gpu_time)
        print(" The add_gpuarray_no_kernel times are")
        print(arr_avg_total_gpuarray_no_kernel_time_nomemcpy)
        print(" The add_gpuarray_no_kernel times with memcpy are")
        print(arr_avg_total_gpuarray_no_kernel_time)
        print(" The add_gpuarray_using_kernel times are")
        print(arr_avg_total_gpuarray_time_nomemcpy)
        print(" The add_gpuarray_using_kernel times with memcpy are")
        print(arr_avg_total_gpuarray_time)