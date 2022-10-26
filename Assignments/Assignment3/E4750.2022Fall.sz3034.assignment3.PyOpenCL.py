import relevant.libraries

class Convolution:
    def __init__(self):
        """
        Attributes for instance of clModule
        Includes OpenCL context, command queue, kernel code.
        """

        # Get platform and device property
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()       

        # Create Context:
        self.ctx = cl.Context(devs)

        # Setup Command Queue:
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

		# I do not recommend modifying the code above this line (from `NAME =` till `self.queue = `)

        # Use this space to define the thread dimensions if required, or it can be incorporated into main function
		# You can also define a lambda function to compute grid dimensions if required.

        kernel_enable_shared_mem_optimizations = """
        #define Shared_mem_optimized
        """

        kernel_enable_constant_mem_optimizations = """
        #define Constant_mem_optimized
        """

		# STUDENTS SHOULD NOT MODIFY kernel_enable_shared_mem_optimizations, kernel_enable_constant_mem_optimizations, self.module_naive_gpu, self.module_shared_mem_optimized and self.module_const_mem_optimized

		# STUDENTS ARE FREE TO MODIFY ANY PART OF THE CODE INSIDE kernelwrapper as long as the tasks mentioned in the Programming Part are satisfied. The examples shown below are just for reference.

        kernel_code = r"""

		[TODO: Students to write entire kernel code. An example of using the ifdef and ifndef is shown below. The example can be modified if necessary]

        #ifndef Constant_mem_optimized
        __kernel void conv_gpu(__global float* a, __global float* b, __global float* c, const unsigned int in_matrix_num_rows, const unsigned int in_matrix_num_cols, const unsigned int in_mask_num_rows, const unsigned int in_mask_num_cols)
        #endif
        #ifdef Constant_mem_optimized
        __kernel void conv_gpu(__global float* a, __constant float* mask, __global float* c, const unsigned int in_matrix_num_rows, const unsigned int in_matrix_num_cols, const unsigned int in_mask_num_rows, const unsigned int in_mask_num_cols)
        #endif
        {
            [TODO: Perform required tasks, likely some variable declaration, and index calculation, maybe more]

            #ifdef Shared_mem_optimized

			[TODO: Perform some part of Shared memory optimization routine, maybe more]

            #endif

			[TODO: Perform required tasks, mostly relating to the computation part. More #ifdef and #ifndef can be added as necessary]
        }
        """

        self.module_naive_gpu = cl.Program(self.ctx, kernel_code).build()
        self.module_shared_mem_optimized = cl.Program(self.ctx, kernel_enable_shared_mem_optimizations + kernel_code).build()
        self.module_const_mem_optimized = self.prg = cl.Program(self.ctx, kernel_enable_shared_mem_optimizations + kernel_enable_constant_mem_optimizations + kernel_code).build()

        # If you wish, you can also include additional compiled kernels and compile-time defines that you may use for debugging without modifying the above three compiled kernel.

    def conv_gpu_naive(self, inputmatrix, inputmask, input_matrix_numrows, input_matrix_numcolumns, input_mask_numrows, input_mask_numcolumns):
        # Write methods to call self.module_naive_gpu for computing convolution and return the results and time taken. The above input variable names like inputmask, input_matrix_numrows, etc can be changed as per student requirements.

    def conv_gpu_shared_mem(self, inputmatrix, inputmask, input_matrix_numrows, input_matrix_numcolumns, input_mask_numrows, input_mask_numcolumns):
        # Write methods to call self.module_shared_mem_optimized for computing convolution and return the results and time taken. The above input variable names like inputmask, input_matrix_numrows, etc can be changed as per student requirements.
    
    def conv_gpu_shared_and_constant_mem(self, inputmatrix, inputmask, input_matrix_numrows, input_matrix_numcolumns, input_mask_numrows, input_mask_numcolumns):
        # Write methods to call self.module_const_mem_optimized for computing convolution and return the results and time taken. The above input variable names like inputmask, input_matrix_numrows, etc can be changed as per student requirements.

    def test_conv_pycuda(self, inputmatrix, inputmask):
        # Write methods to perform convolution on the same dataset using scipy's convolution methods running on CPU and return the results and time. Students are free to experiment with different variable names in place of inputmatrix and inputmask.

if __name__ == "__main__":
    # Main code
    # Write methods to perform the computations, get the timings for all the tasks mentioned in programming sections and also comparing results and mentioning if there is a sum mismatch. Students can experiment with numpy.math.isclose function. 


