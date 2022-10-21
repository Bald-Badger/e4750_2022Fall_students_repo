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

        self.getSourceModule()

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
        kernel_printer_end = """
        #define SYNC_BEFORE_PRINT
        """

        kernel_printer = """
        #define PRINT_ENABLE_DEBUG
        """

        kernel_main_wrapper = r"""

        __global__ void main_function(float *input_value, float *computed_value, int n)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if((idx%2) == 0){
                computed_value[idx] = sinf(input_value[idx]);
                #ifdef PRINT_ENABLE_DEBUG
                if(idx<n)
                {
                    printf("Hello from thread %d \n", idx);
                }
                #endif
            }
            else{
                computed_value[idx] = sine_taylor(input_value[idx]);
                #ifdef PRINT_ENABLE_DEBUG
                if(idx<n)
                {
                    printf("Hello from thread %d \n", idx);
                }
                #endif
            }

            #ifdef SYNC_BEFORE_PRINT
            if(idx<n)
            {
                printf("Hello from thread %d \n", idx);
            }
            #endif     
        }
        """

        kernel_device = """
        #define TAYLOR_COEFFS 10000

        __device__ float sine_taylor(float in)
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

        # Compile kernel code and store it in self.module_*

        self.module_no_print = SourceModule(kernel_device + kernel_main_wrapper)
        self.module_with_print_nosync = SourceModule(kernel_printer + kernel_device + kernel_main_wrapper)
        self.module_with_print_with_sync = SourceModule(kernel_printer_end + kernel_device + kernel_main_wrapper)

        # SourceModule is the Cuda.Compiler and the kernelwrapper text is given as input to SourceModule. This compiler takes in C code as text inside triple quotes (ie a string) and compiles it to CUDA code.
        # When we call this getSourceModule method for an object of this class, it will return the compiled kernelwrapper function, which will now take inputs a, b, c, n, along with block_specifications and grid_specifications.

    def Sine_device_mem_gpu(self, a, length, printing_properties):
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
        start = cuda.Event()
        end = cuda.Event()
        # We create two pycuda.driver.Event (cuda here is pycuda.driver - check beginning of code for import commands) instances.
        # One instance is named "start" and another is named "end". The name is upto our choice. Both "start" and "end" now has methods for recording which it got from pycuda.driver.Event().
        # We will call this record method later (within this explicitAdd method). First record will start timer, second record will end timer.
        if(printing_properties == 'No Print'):
            prg = self.module_no_print.get_function("main_function")
        elif(printing_properties == 'Print'):
            prg = self.module_with_print_nosync.get_function("main_function")
        else:
            prg = self.module_with_print_with_sync.get_function("main_function")

        # Device memory allocation for input and output arrays
        a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize) # allocating memory inside the gpu (locations stored in a_gpu and b_gpu) for storing the input vectors (a and b)
        c_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize) # allocating memory inside the gpu for storing the output after computation.
        # #l_gpu = cuda.mem_alloc(l.size * l.dtype.itemsize) # We are not allocating memory for passing value l. We pass it directly as a numpy intc type for the kernel function.

        # Copy data from host to device, and set timer for including copy from CPU to device
        time_start = time.clock() # timer for including memory copy
        cuda.memcpy_htod(a_gpu, a) # copying the local variable a in the explicitAdd function (or method) to a_gpu inside the GPU (we allocated this memory inside GPU in the earlier lines)
        # #cuda.memcpy_htod(l_gpu, l) # Reason above. We donot create l_gpu, instead we pass directly l as np intc type to kernel function.
        # Call the kernel function from the compiled module
        # the current explicitAdd method (of which the current portion of the code is a part of), is a method defined for deviceAdd Class.
        # self.mod will call self.getSourceModule() method defined for this deviceAdd class (check init method for this deviceAdd Class)
        # this self.getSourceModule() method has the kernel defined inside kernelwrapper and returns the compiled kernel function for GPU.

        # Get grid and block dim
        griddimensions = (int(np.ceil(length/self.threads_total)),1) # A grid is a collection of blocks. And a block is a collection of threads.
        blockdimensions = (self.threads_per_block_x,self.threads_per_block_y,self.threads_per_block_z) # Each block has a 3-D arrangement of threads that can be called by their index in x,y and z directions.

        # Record execution time and call the kernel loaded to the device
        start.record() # calling the record function to record time. Start is an instace of pycuda.driver.Event. Now this will store this time as an attribute to this variable named start!
        time1_start = time.clock()
        event = prg(a_gpu, c_gpu, l, block=blockdimensions, grid=griddimensions) #this event is different from cuda.event. We just named it event and we are calling a function here.
        # It is not even necessary to name it and enough to just directly call func. (refer to other methods defined for this class like implicitAdd where we call func directly without naming it to another variable like we did here).
        #event.synchronize()
        # Wait for the event to complete
        time1_end = (time.clock() - time1_start)
        end.record() # Executes after the above function is finished. Calling the record function for end variable which is another instance of pycuda.driver.Event. This will store this time, as an attribute to this variable named end.
        # Note that in this case, start, event = func(...) and end are cuda library based commands.[TODO : Check if true] It is reasonable to expect all of these to execute sequentially (although the content inside func executes parallel).

        # Copy result from device to the host
        c = np.empty_like(a) # Allocating memory in host to copy the result from GPU memory. This again is a local copy (stored in host memory) inside this method.
        # We will return this value to another variable, again in host memory (see the return statement for this explicitAdd method and check the main code). We will have two copies of this in host memory after the full task is over.
        cuda.memcpy_dtoh(c, c_gpu) # copying result from GPU memory to host memory allocated for local variable.
        time_taken_memcopy = (time.clock() - time_start) # getting current time subtracting time_start captured before memory copy to GPU.
        # I am doing this time calculation first (before computing gpu running time) to ensure the time measured is only for the total operation activity involving memory copy and GPU computation.
        # The GPU times are already stored in the attributes to variables start and end, computation of this time difference might take a few cycles and those would otherwise get included in the time calculation.
        time_taken_nomemcopy = start.time_till(end) # start is a pycuda.driver.Event instance. It has a method time_till which gets an input which is also a pycuda.driver.event instance (here the other instance is named end).
        # After this it computes time difference between these two events using the time attributes that got stored in start and end when using the record() method.
        
        # return a tuple of output of addition and time taken to execute the operation.
        return (c, time1_end, time_taken_memcopy) # returning the results to host memory located for variable in main program.
        # [TODO: Cuda.Event() is not returning proper values, hence using time.clock() for GPU process as well. Check why it is not working]
        
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
    graphicscomputer = CudaModule()

    for current_operation in valid_operations:
        print ("current operation")
        print(current_operation)
        arr_avg_total_cpu_time = np.array([])
        arr_avg_total_device_mem_gpu_time_nomemcpy = np.array([])
        arr_avg_total_device_mem_gpu_time = np.array([])
        
        for vector_size in vector_sizes:

            arr_total_cpu_time = np.array([])
            arr_total_device_mem_gpu_time_no_memcpy = np.array([])
            arr_total_device_mem_gpu_time = np.array([])

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
                            c_gpu_device_add, gpu_time_cl_array, gpu_time_memcpy = graphicscomputer.Sine_device_mem_gpu(a_array_np,vector_size,current_operation) 
                            sum_diff = c_gpu_device_add - c_np_cpu_add
                            arr_total_device_mem_gpu_time_no_memcpy = np.append(arr_total_device_mem_gpu_time_no_memcpy, gpu_time_cl_array)
                            arr_total_device_mem_gpu_time = np.append(arr_total_device_mem_gpu_time, gpu_time_memcpy)

                        total_diff = sum_diff.sum()
            avg_total_cpu_time = ((arr_total_cpu_time.sum())/2)
            arr_avg_total_cpu_time = np.append(arr_avg_total_cpu_time, avg_total_cpu_time)
            avg_total_device_mem_gpu_time = ((arr_total_device_mem_gpu_time.sum())/50)
            arr_avg_total_device_mem_gpu_time = np.append(arr_avg_total_device_mem_gpu_time, avg_total_device_mem_gpu_time)
        print(" The CPU times are")
        print(arr_avg_total_cpu_time)
        print(" The add_device_mem_gpu times with memcpy are")
        print(arr_avg_total_device_mem_gpu_time)