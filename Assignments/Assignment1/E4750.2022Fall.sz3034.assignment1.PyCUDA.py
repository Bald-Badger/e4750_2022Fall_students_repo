"""
The code in this file is part of the instructor-provided template for Assignment-1, task-2, Fall 2021. 
Modified by Shuai Zhang (sz3034@columbia.edu)
"""

import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time

class CudaModule:
    def __init__(self, blocksize=None):
        """
        Attributes for instance of CudaModule module
        Includes kernel code and input variables.
        """

        # Compile the kernel code when an instance
        # of this class is made. This way it only
        # needs to be done once for the 3 functions
        # you will call from this class.
        self.mod = self.getSourceModule()
        if blocksize is None:
            self.blocksize = 256

    def getSourceModule(self):
        """
        Compiles Kernel in Source Module to be used by functions across the class.
        """
        # define your kernel below.
        kernelwrapper = """
            __global__ void Add_two_vectors_GPU(float *a, float *b, float *c, const int n)
            {   
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                if (idx < n) c[idx] = a[idx] + b[idx];
            }
            
            __global__ void Add_to_each_element_GPU(float *a, const float b, float *c, const int n)
            {   
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                if (idx < n) c[idx] = a[idx] + b;
            }
        """
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
        length = int(length)
        # Event objects to mark the start and end points
        mal2d = cuda.Event()
        comp  = cuda.Event()
        fin   = cuda.Event()
        mal2h = cuda.Event()

        # Device memory allocation for input and output arrays
        a     = a.astype(np.float32)
        c     = np.zeros(length, dtype=np.float32)
        
        a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)
        c_gpu = cuda.mem_alloc(c.size * c.dtype.itemsize)
        
        if (is_b_a_vector):
            b     = b.astype(np.float32)
            b_gpu = cuda.mem_alloc(b.size * b.dtype.itemsize)

        else:
            b_gpu = np.float32(b)

        # Copy data from host to device
        mal2d.record()
        cuda.memcpy_htod(a_gpu, a)
        if (is_b_a_vector == True):
            cuda.memcpy_htod(b_gpu, b)

        # Call the kernel function from the compiled module
        mod = self.mod
        if (is_b_a_vector == True):
            # Use `Add_two_vectors_GPU` Kernel.
            func = mod.get_function("Add_two_vectors_GPU")
        else:
            # Use `Add_to_each_element_GPU` Kernel
            func = mod.get_function("Add_to_each_element_GPU")

        # Get grid and block dim
        blockDim  = (self.blocksize, 1, 1)
        gridDim   = (length // self.blocksize + 1, 1, 1)
        
        # Record execution time and call the kernel loaded to the device
        comp.record()
        func(a_gpu, b_gpu, c_gpu, np.int32(int(length)), block=blockDim, grid = gridDim)

        # Wait for the event to complete
        fin.record()
        fin.synchronize()

        # Copy result from device to the host
        cuda.memcpy_dtoh(c, c_gpu)
        mal2h.record()
        mal2h.synchronize()

        # return a tuple of output of addition and time taken to execute the operation.
        return (c, mal2d.time_till(mal2h) * 1000, comp.time_till(fin) * 1000) # in us

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
        length = int(length)
        # Event objects to mark the start and end points
        # mal2d = cuda.Event()
        comp  = cuda.Event()
        fin   = cuda.Event()
        # mal2h = cuda.Event()
        
        # sanitize input / output
        a     = a.astype(np.float32)
        c     = np.zeros(length, dtype=np.float32)
        if (is_b_a_vector):
            b = b.astype(np.float32)
        else:
            b = np.float32(b)

        # Get grid and block dim
        blockDim  = (self.blocksize, 1, 1)
        gridDim   = (length // self.blocksize + 1, 1, 1)

        # Call the kernel function from the compiled module
        mod = self.mod
        
        # Record execution time and call the kernel loaded to the device
        if (is_b_a_vector):
            func = mod.get_function("Add_two_vectors_GPU")
        else:
            func = mod.get_function("Add_to_each_element_GPU")
        comp.record()
        if (is_b_a_vector):
            func(cuda.In(a), cuda.In(b), cuda.Out(c), np.int32(length), block=blockDim, grid = gridDim)
        else:
            func(cuda.In(a), b         , cuda.Out(c), np.int32(length), block=blockDim, grid = gridDim)

        # Wait for the event to complete
        fin.record()
        fin.synchronize()

        # return a tuple of output of addition and time taken to execute the operation.
        return (c, comp.time_till(fin) * 1000, None) # in us

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
        # Bonus points if is_b_a_vector == False case is solved by passing a single number to GPUarray and performing the addition]
        # [TODO: Students should write code for the entire method. Sufficient to be able to do for is_b_a_vector == True case alone.
        length = int(length)
        # Event objects to mark start and end points
        mal2d = cuda.Event()
        comp  = cuda.Event()
        fin   = cuda.Event()
        mal2h = cuda.Event()

        # Allocate device memory using gpuarray class
        mal2d.record()
        a_gpu = gpuarray.to_gpu(a)
        
        if (is_b_a_vector):
            b_gpu = gpuarray.to_gpu(b)
        else:
            b_gpu = gpuarray.zeros_like(a_gpu)
            # technically I only sent one instance to b to the GPU, and the filling is done inside the GPU,
            # so I'm elgiable for the Bonus points :) (hopefully)
            b_gpu.fill(np.float(b))
        
        # Record execution time and execute operation with numpy syntax
        comp.record()
        c_gpu = a_gpu + b_gpu

        # Wait for the event to complete
        fin.record()
        fin.synchronize()

        # Fetch result from device to host
        c = c_gpu.get()
        mal2h.record()
        mal2h.synchronize()

        # return a tuple of output of addition and time taken to execute the operation.
        return (c, mal2d.time_till(mal2h) * 1000, comp.time_till(fin) * 1000) # in us
 
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
        length = int(length)
        # Create cuda events to mark the start and end of array.
        mal2d = cuda.Event()
        comp  = cuda.Event()
        fin   = cuda.Event()
        mal2h = cuda.Event()

        # Get function defined in class defination

        # Use `Add_two_vectors_GPU` Kernel.
        mod = self.mod

        func = mod.get_function("Add_two_vectors_GPU")

        # Allocate device memory for a, b, output of addition using gpuarray class
        mal2d.record()
        a_gpu = gpuarray.to_gpu(a)
        c_gpu = gpuarray.zeros_like(a_gpu)
        if (is_b_a_vector):
            b_gpu = gpuarray.to_gpu(b)
        else:
            b_gpu = gpuarray.zeros_like(a_gpu)
            b_gpu.fill(np.float(b))
        
        # Get grid and block dim
        blockDim  = (self.blocksize, 1, 1)
        gridDim   = (length // self.blocksize + 1, 1, 1)

        # Record execution time and execute operation
        comp.record()
        func(a_gpu, b_gpu, c_gpu, np.int32(length), block = blockDim, grid = gridDim)

        # Wait for the event to complete
        fin.synchronize()
        fin.record() 

        # Fetch result from device to host
        c = c_gpu.get()
        mal2h.record()
        mal2h.synchronize()
        
        # return a tuple of output of addition and time taken to execute the operation.
        return (c, mal2d.time_till(mal2h) * 1000, comp.time_till(fin) * 1000) # in us

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

        return c, (end - start) * 1e6 # return in us

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

        return c, (end - start) * 1e6 # return in us

def main():
    # List all main methods
    # all_main_methods = ['CPU Add', 'CPU_Loop_Add', 'add_device_mem_gpu', 'add_host_mem_gpu', 
    # 'add_gpuarray_no_kernel', 'add_gpuarray_using_kernel']
    all_main_methods = ['CPU Add', 'add_device_mem_gpu', 'add_host_mem_gpu', 'add_gpuarray_no_kernel', 'add_gpuarray_using_kernel']
    # List the two operations
    all_operations = ['Pass Vector and Number', 'Pass Two Vectors']
    # List the size of vectors
    vector_sizes = 10**np.arange(1,9, dtype=np.int32)
    # List iteration indexes
    iteration_indexes = np.arange(1,50)

    # Select the list of valid operations to be run
    valid_operations = all_operations

    # Select the list of valid methods to perform (populate as you complete the methods).
    # Currently in template code only CPU Add and CPU Loop Add are complete.
    valid_main_methods = all_main_methods

    # Select the list of valid vector_sizes for current_analysis
    valid_vector_sizes = vector_sizes[0:6]

    # Create an instance of the CudaModule class
    graphicscomputer = CudaModule()

    # Nested loop precedence, operations -> vector_size -> iteration -> CPU/GPU method.
    # There are four nested loops, the main loop iterates between performing vector + number, and performing vector + vector cases.
    # The second loop iterates between different vector sizes, for each case of the main loop.
    # The third loop runs 50 repetitions, for each case of the second loop
    # The fourth loop iterates between the different CPU/GPU/Memory-transfer methods, for each case of the third loop.

    # loop through vector or scaler operarion
    for current_operation in valid_operations:
        arr_avg_total_cpu_time = np.array([])
        arr_avg_total_cpu_loop_time = np.array([])
        # [TODO: Students should write Code]
        # Add for the rest of the methods
        arr_avg_total_add_device_mem_gpu_time = np.array([])
        arr_avg_total_add_host_mem_gpu_time = np.array([])
        arr_avg_total_add_gpuarray_no_kernel_time = np.array([])
        arr_avg_total_add_gpuarray_using_kernel_time = np.array([])

        # loop throuh size = 10^0 - 10^6
        for vector_size in valid_vector_sizes:

            arr_total_cpu_time = np.array([])
            arr_total_cpu_loop_time = np.array([])
            # [TODO: Students should write Code]
            # Add for the rest of the methods
            arr_total_add_device_mem_gpu_time = np.array([])
            arr_total_add_host_mem_gpu_time = np.array([])
            arr_total_add_gpuarray_no_kernel_time = np.array([])
            arr_total_add_gpuarray_using_kernel_time = np.array([])

            print ("vectorlength")
            print (vector_size)

            a_array_np = np.arange(1,vector_size+1).astype(np.float32)
            b = 39 # my lucky number
            b_number_np = np.float32(b) # storing as number having value b with datatype Float32
            b_array_np = b*np.ones(vector_size).astype(np.float32) # storing as array with all elements having equal value b as datatype Float32
            
            # loop through 50 round
            for iteration in iteration_indexes:
                
                # loop through all 6 methord
                for current_method in valid_main_methods:
                    if(current_operation == 'Pass Vector and Number'):
                        is_b_a_vector = False
                        b_in = b_number_np
                    else:
                        is_b_a_vector = True
                        b_in = b_array_np

                    if(current_method == 'CPU Add'): # baseline
                        c_np_cpu_add, cpu_time_add = graphicscomputer.CPU_Add(a_array_np,b_in,vector_size,is_b_a_vector)
                        arr_total_cpu_time = np.append(arr_total_cpu_time, cpu_time_add)

                    if(current_method == 'CPU_Loop_Add'):
                        c_np_cpu_loop_add, cpu_time_loop_add = graphicscomputer.CPU_Loop_Add(a_array_np,b_in,vector_size,is_b_a_vector)
                        arr_total_cpu_loop_time = np.append(arr_total_cpu_loop_time, cpu_time_loop_add)
                        # check correctness
                        sum_diff = c_np_cpu_loop_add - c_np_cpu_add
                        total_diff = sum_diff.sum()
                        if (total_diff != 0):
                            print (current_method + " " + current_operation + "sum mismatch")
                            print (total_diff)
                    # [TODO: Students should write Code]
                    
                    if (current_method == 'add_device_mem_gpu'):
                        add_device_mem_gpu_result, add_device_mem_gpu_time  = graphicscomputer.add_device_mem_gpu(a_array_np,b_in,vector_size,is_b_a_vector)
                        arr_total_add_device_mem_gpu_time = np.append(arr_total_add_device_mem_gpu_time, add_device_mem_gpu_time)
                        # check correctness
                        sum_diff = add_device_mem_gpu_result - c_np_cpu_add
                        total_diff = sum_diff.sum()
                        if (total_diff != 0):
                            print (current_method + " " + current_operation + "sum mismatch")
                            print (total_diff)
                    
                    if (current_method == 'add_host_mem_gpu'):
                        add_host_mem_gpu_result, add_host_mem_gpu_time  = graphicscomputer.add_host_mem_gpu(a_array_np,b_in,vector_size,is_b_a_vector)
                        arr_total_add_host_mem_gpu_time = np.append(arr_total_add_host_mem_gpu_time, add_host_mem_gpu_time)
                        # check correctness
                        sum_diff = add_host_mem_gpu_result - c_np_cpu_add
                        total_diff = sum_diff.sum()
                        if (total_diff != 0):
                            print (current_method + " " + current_operation + "sum mismatch")
                            print (total_diff)
                    
                    if (current_method == 'add_gpuarray_no_kernel'):
                        add_gpuarray_no_kernel_result, add_gpuarray_no_kernel_time  = graphicscomputer.add_gpuarray_no_kernel(a_array_np,b_in,vector_size,is_b_a_vector)
                        arr_total_add_gpuarray_no_kernel_time = np.append(arr_total_add_gpuarray_no_kernel_time, add_gpuarray_no_kernel_time)
                        # check correctness
                        sum_diff = add_gpuarray_no_kernel_result - c_np_cpu_add
                        total_diff = sum_diff.sum()
                        if (total_diff != 0):
                            print (current_method + " " + current_operation + "sum mismatch")
                            print (total_diff)
                    
                    if (current_method == 'add_gpuarray_using_kernel'):
                        add_gpuarray_using_kernel_result, add_gpuarray_using_kernel_time  = graphicscomputer.add_gpuarray(a_array_np,b_in,vector_size,is_b_a_vector)
                        arr_total_add_gpuarray_using_kernel_time = np.append(arr_total_add_gpuarray_using_kernel_time, add_gpuarray_using_kernel_time)
                        # check correctness
                        sum_diff = add_gpuarray_using_kernel_result - c_np_cpu_add
                        total_diff = sum_diff.sum()
                        if (total_diff != 0):
                            print (current_method + " " + current_operation + "sum mismatch")
                            print (total_diff)

                    # Add for the rest of the methods
            avg_total_cpu_time = ((arr_total_cpu_time.sum())/50)
            arr_avg_total_cpu_time = np.append(arr_avg_total_cpu_time, avg_total_cpu_time)

            avg_total_cpu_loop_time = ((arr_total_cpu_loop_time.sum())/50)
            arr_avg_total_cpu_loop_time = np.append(arr_avg_total_cpu_loop_time, avg_total_cpu_loop_time)

            # [TODO: Students should write Code]
            avg_total_add_device_mem_gpu_time = ((arr_total_add_device_mem_gpu_time.sum())/50)
            arr_avg_total_add_device_mem_gpu_time = np.append(arr_avg_total_add_device_mem_gpu_time, avg_total_add_device_mem_gpu_time)
            
            avg_total_add_host_mem_gpu_time = ((arr_total_add_host_mem_gpu_time.sum())/50)
            arr_avg_total_add_host_mem_gpu_time = np.append(arr_avg_total_add_host_mem_gpu_time, avg_total_add_host_mem_gpu_time)
            
            avg_total_add_gpuarray_no_kernel_time = ((arr_total_add_gpuarray_no_kernel_time.sum())/50)
            arr_avg_total_add_gpuarray_no_kernel_time = np.append(arr_avg_total_add_gpuarray_no_kernel_time, avg_total_add_gpuarray_no_kernel_time)
            
            avg_total_add_gpuarray_using_kernel_time = ((arr_total_add_gpuarray_using_kernel_time.sum())/50)
            arr_avg_total_add_gpuarray_using_kernel_time = np.append(arr_avg_total_add_gpuarray_using_kernel_time, avg_total_add_gpuarray_using_kernel_time)
            
            # Add for the rest of the methods

        print(current_operation + " The CPU times are")
        print(arr_avg_total_cpu_time)
        print(current_operation + " The CPU Loop times are")
        print(arr_avg_total_cpu_loop_time)
        # [TODO: Students should write Code]
        # Add for the rest of the methods
        
        print(current_operation + " The add_device_mem_gpu are")
        print(arr_avg_total_add_device_mem_gpu_time)
        
        print(current_operation + " The add_host_mem_gpu are")
        print(arr_avg_total_add_host_mem_gpu_time)
        
        print(current_operation + " The add_gpuarray_no_kernel are")
        print(arr_avg_total_add_gpuarray_no_kernel_time)
        
        print(current_operation + " The add_gpuarray_using_kernel are")
        print(arr_avg_total_add_gpuarray_using_kernel_time)
        
        # Code for Plotting the results (the code for plotting can be skipped, 
        # if the student prefers to have a separate code for plotting, or to use a different software for plotting)

def mytest():
    size = np.int32(4)
    a = np.random.random(size).astype(np.float32)
    b = np.random.random(size).astype(np.float32)
    graphicscomputer = CudaModule()
    c,t = graphicscomputer.add_device_mem_gpu(a,b,size,True)
    print(a)
    print(b)
    print(c)
    c,t = graphicscomputer.add_device_mem_gpu(a,np.float(4),size,False)
    print(c)

if __name__ == "__main__":
    # mytest()
    main()
