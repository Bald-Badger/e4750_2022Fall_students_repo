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