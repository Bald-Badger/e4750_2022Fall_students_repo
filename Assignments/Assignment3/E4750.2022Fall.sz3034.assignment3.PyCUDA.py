import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import matplotlib.pyplot as plt
from scipy import signal

class Convolution:
    def __init__(self):
		# Use this space to define the thread dimensions if required, or it can be incorporated into main function
		# You can also define a lambda function to compute grid dimensions if required.
        self.getSourceModule()
        self.threads_per_block_x = 32 # Students can modify this number.
        self.threads_per_block_y = 32
        self.threads_per_block_z = 1
        self.threads_total = self.threads_per_block_x * self.threads_per_block_y * self.threads_per_block_z

    def getSourceModule(self):
        kernel_enable_shared_mem_optimizations = """
        #define Shared_mem_optimized
        """

        kernel_enable_constant_mem_optimizations = """
        #define Constant_mem_optimized
        """

		# STUDENTS SHOULD NOT MODIFY kernel_enable_shared_mem_optimizations, 
        # kernel_enable_constant_mem_optimizations, self.module_naive_gpu, 
        # self.module_shared_mem_optimized and self.module_const_mem_optimized

		# STUDENTS ARE FREE TO MODIFY ANY PART OF THE CODE INSIDE kernelwrapper 
        # as long as the tasks mentioned in the Programming Part are satisfied. 
        # The examples shown below are just for reference.
        
        kernelwrapper = r"""
		// [TODO: Students to write entire kernel code. An example of using the ifdef and ifndef is shown below. The example can be modified if necessary]
        #ifndef Constant_mem_optimized
        __global__ void conv_gpu(float *a, float *b, float *c, int in_matrix_num_rows, int in_matrix_num_cols, int in_mask_num_rows, int in_mask_num_cols) {
            // assiming full padding
            // height/width of c = a + b - 1
            
            // dimension out output matrix
            int c_width = in_matrix_num_cols + in_mask_num_cols - 1;
            int c_height = in_matrix_num_rows + in_mask_num_rows - 1;
            
            // thread ID, block ID
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int bx = blockIdx.x;
            int by = blockIdx.y;
            
            // row number and col number current thread is working on
            int row = by*blockDim.y + ty;
            int col = bx*blockDim.x + tx
            
            
            
        }
        #endif
        #ifdef Constant_mem_optimized
        __constant__ int matrix_num_rows;
        __constant__ int matrix_num_cols; 
        __constant__ int mask_num_rows;
        __constant__ int in_mask_num_cols;
        __global__ void conv_gpu(float *a, float *c, int in_matrix_num_rows, int in_matrix_num_cols, int in_mask_num_rows, int in_mask_num_cols)
        #endif
        {
			// [TODO: Perform required tasks, likely some variable declaration, and index calculation, maybe more]

            #ifdef Shared_mem_optimized

			// [TODO: Perform some part of Shared memory optimization routine, maybe more]

            #endif

			// [TODO: Perform required tasks, mostly relating to the computation part. More #ifdef and #ifndef can be added as necessary]
        }
        """

        self.module_naive_gpu = SourceModule(kernelwrapper)
        self.module_shared_mem_optimized = SourceModule(kernel_enable_shared_mem_optimizations + kernelwrapper)
        self.module_const_mem_optimized = SourceModule(kernel_enable_shared_mem_optimizations + kernel_enable_constant_mem_optimizations + kernelwrapper)

        # If you wish, you can also include additional compiled kernels and compile-time defines that 
        # you may use for debugging without modifying the above three compiled kernel.

    def conv_gpu_naive(self, inputmatrix, inputmask, input_matrix_numrows, input_matrix_numcolumns, input_mask_numrows, input_mask_numcolumns):
		# Write methods to call self.module_naive_gpu for computing convolution and return the results and time taken. 
        # The above input variable names like inputmask, input_matrix_numrows, etc can be changed as per student requirements.
        pass


    def conv_gpu_shared_mem(self, inputmatrix, inputmask, input_matrix_numrows, input_matrix_numcolumns, input_mask_numrows, input_mask_numcolumns):
        # Write methods to call self.module_shared_mem_optimized for computing convolution and return the results and time taken. 
        # The above input variable names like inputmask, input_matrix_numrows, etc can be changed as per student requirements.
        pass


    def conv_gpu_shared_and_constant_mem(self, inputmatrix, inputmask, input_matrix_numrows, input_matrix_numcolumns, input_mask_numrows, input_mask_numcolumns):
        # Write methods to call self.module_const_mem_optimized for computing convolution and return the results and time taken. 
        # The above input variable names like inputmask, input_matrix_numrows, etc can be changed as per student requirements.
        pass


    def test_conv_pycuda(self, inputmatrix, inputmask):
        # Write methods to perform convolution on the same dataset using scipy's convolution methods running on CPU and return the results and time. 
        # Students are free to experiment with different variable names in place of inputmatrix and inputmask.
        return signal.convolve2d(inputmatrix, inputmask)


def main():
    a = np.array(
        [
            [1, 2, 3, 4, 5 ],
            [6, 7, 8, 9, 10],
            [11,12,13,14,15],
            [16,17,18,19,20]
        ]
    , dtype=np.float32)
    k = np.array(
        [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 1]
        ]
    , dtype=np.float32)
    
    computer = Convolution()
    reference = computer.test_conv_pycuda(a, k)
    print(reference)


if __name__ == "__main__":
    # Main code
    # Write methods to perform the computations, get the timings for all the tasks mentioned in programming sections
    # and also comparing results and mentioning if there is a sum mismatch. Students can experiment with numpy.math.isclose function.
    main()
