"""
The code in this file is part of the instructor-provided template for Assignment-1, task-2, Fall 2021. 
"""

from cProfile import label
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import matplotlib.pyplot as plt

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
                // [TODO]: STUDENTS SHOULD WRITE CODE TO USE CUDA MATH FUNCTION TO COMPUTE SINE OF INPUT VALUE
                computed_value[n] = sinf(input_value[idx]);
                #ifdef PRINT_ENABLE_DEBUG
                if(idx<n)
                {
                    printf("Hello from index %d \n", idx);
                }
                #endif
            }
            else{
                // [TODO]: STUDENTS SHOULD WRITE CODE TO CALL THE DEVICE FUNCTION sine_taylor TO COMPUTE SINE OF INPUT VALUE
                computed_value[n] = sinf(input_value[idx]);
                #ifdef PRINT_ENABLE_DEBUG
                if(idx<n)
                {
                    // [TODO]: STUDENTS SHOULD WRITE CODE TO PRINT THE INDEX OF THE ARRAY BEING COMPUTED
                    printf("Hello from index %d \n", idx);
                }
                #endif
            }

            #ifdef PRINT_ENABLE_AFTER_COMPUTATION
            if(idx<n)
            {
                // [TODO]: STUDENTS SHOULD WRITE CODE TO PRINT THE INDEX OF THE ARRAY BEING COMPUTED
                printf("index %d completed\n", idx);
            }
            #endif
        }
        """

        kernel_device = """
            #define TAYLOR_COEFFS 1000

            double sine_taylor(double in) {
                double result;           // final result
                double term;             // untermediate term for each iter
                double power = in;       // base case
                double factorial = 1;    // base case
                
                double power_iter = in * in;
                double factorial_iter;
                
                for (unsigned int i = 0; i < TAYLOR_COEFFS; i++) {
                    
                    term = power / factorial;
                    
                    power = power * power_iter;
                    factorial_iter = factorial * (i + 2) * (i + 3);
                    factorial = factorial * factorial_iter;
                    
                    if (i & 0x01)   // is odd
                        result -= term;
                    else            // is even
                        result += term;
                }
                return result;
            }
        """

        # Compile kernel code and store it in self.module_*

        self.module_no_print = SourceModule(kernel_device + kernel_main_wrapper)
        self.module_with_print_nosync = SourceModule(kernel_printer + kernel_device + kernel_main_wrapper)
        self.module_with_print_with_sync = SourceModule(kernel_printer_end + kernel_device + kernel_main_wrapper)

        # SourceModule is the Cuda.Compiler and the kernelwrapper text is given as input to SourceModule. 
        # This compiler takes in C code as text inside triple quotes (ie a string) and compiles it to CUDA code.
        # When we call this getSourceModule method for an object of this class, 
        # it will return the compiled kernelwrapper function, 
        # which will now take inputs along with block_specifications and grid_specifications.
    
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
        a               = a.astype(np.float32)
        o               = np.zeros(length, dtype=np.float32)

        # Event objects to mark the start and end points
        start           = cuda.Event()
        malloc_start    = cuda.Event()
        malloc_end      = cuda.Event()
        compute_start   = cuda.Event()
        compute_end     = cuda.Event()
        finish          = cuda.Event()

        # Device memory allocation for input and output arrays
        start.record()
        a_gpu           = cuda.mem_alloc(a.size * a.dtype.itemsize)
        o_gpu           = cuda.mem_alloc(o.size * o.dtype.itemsize)

        # Copy data from host to device
        malloc_start.record()
        cuda.memcpy_htod(a_gpu, a)
        cuda.memcpy_htod(o_gpu, o)
        malloc_end.record()

        # Call the kernel function from the compiled module
        if(printing_properties == 'No Print'):
            prg = self.module_no_print.get_function("main_function")
        elif(printing_properties == 'Print'):
            prg = self.module_with_print_nosync.get_function("main_function")
        else:
            prg = self.module_with_print_with_sync.get_function("main_function")

        # Get grid and block dim
        blockDim  = (self.threads_per_block_x, self.threads_per_block_y, self.threads_per_block_z)
        gridDim   = (length // self.threads_total + 1, 1, 1)

        # Record execution time and call the kernel loaded to the device
        # void main_function(float *input_value, float *computed_value, int n)
        compute_start.record()
        prg(a_gpu, o_gpu, np.int32(length), block=blockDim, grid=gridDim)

        # Wait for the event to complete
        compute_end.record()
        compute_end.synchronize()

        # Copy result from device to the host
        cuda.memcpy_dtoh(o, o_gpu)
        finish.record()
        finish.synchronize()

        # return a tuple of output of sine computation and time taken to execute the operation.
        return (o, [start.time_till(finish)])

 
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

def main():
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
                        pass
                        #TODO: STUDENTS TO GET OUTPUT TIME AND COMPUTATION FROM CPU_Sine
                    if(current_method == 'Sine_device_mem_gpu'):
                        pass
                        #TODO: STUDENTS TO GET OUTPUT TIME AND COMPUTATION FROM sine_device_mem_gpu

                        #TODO: STUDENTS TO COMPARE RESULTS USING ISCLOSE FUNCTION
        #TODO: STUDENTS CAN USE THIS SPACE TO WRITE NECESSARY TIMING ARRAYS, PERSONAL DEBUGGING PRINT STATEMENTS, ETC
        
if __name__ == "__main__":
    graphicscomputer = CudaModule()
    length = 10
    a_array_np = 0.001*np.arange(1,length+1).astype(np.float32)
    my_answer, t0 = graphicscomputer.sine_device_mem_gpu(a_array_np, length, "Print")
    reference, t1 = graphicscomputer.CPU_Sine(a_array_np, length, "Print")
    print(a_array_np)
    print(my_answer)
    print(reference)
