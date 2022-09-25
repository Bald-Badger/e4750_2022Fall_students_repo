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