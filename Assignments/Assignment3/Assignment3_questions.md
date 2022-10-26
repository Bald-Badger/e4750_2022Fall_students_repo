# EECS E4750: Heterogeneous Computing for Signal and Data Processing (Fall 2022)

## Assignment-3: 2D kernels, Matrices, Memory Locality optimizations using Shared Memory, Constant Memory, using Modular Programming

Total points: 100

### References to help you with assignment
* [PyCuda Examples](https://github.com/inducer/pycuda/tree/main/examples)
* [PyOpenCL Examples](https://github.com/inducer/pyopencl/tree/main/examples)
* [NVidia Blog](https://developer.nvidia.com/blog/tag/cuda/)
* [Video about Convolution](https://www.youtube.com/watch?v=A6qR7UhLvng)
* [Preprocessor Directives and Macros](https://www.informit.com/articles/article.aspx?p=1732873&seqNum=13)
* Other references from previous assignments may also be relevant

### Primer
For this assignment you will implement 2D convolution using GPU. The kernel code will employ varying levels of memory locality optimizations based on compile time defines used. With neither `Shared_mem_optimized` nor `Constant_mem_optimized` compile time defines used, the code should pursue a simple(r) approach which doesn't use shared memory or constant memory. Later you will incorporate shared memory (`Shared_mem_optimized` defined), then constant memory along with shared memory (`Shared_mem_optimized` and `Constant_mem_optimized` defined) in the kernel code using conditional compile.

Matrix convolution is primarily used in image processing for tasks such as image enhancing, encoding etc. A standard image convolution formula for a 5x5 convolution kernel A with matrix B is
```
C(i,j) = sum (m = 0 to 4) {
	 sum(n = 0 to 4) { 
	 	A[m][n] * B[i+m-2][j+n-2] 
	 } 
}
```
 where 0 <= i < B.height and 0 <= j < B.width. For this assignment you can assume that the elements that are "outside" the matrix B, are treated as if they had value zero. You can assume the kernel size 5 x 5 for this assignment but write your code to work for general odd dimnesion kernel sizes.

### Programming Part (80 Points)

All the timing, and plots should be taken from running the code in the Cloud Machine. DONOT produce analysis on personal machines.

Your submission should contain 3 files.

1. Project Report    : E4750.2022Fall.(uni).assignment3.report.PDF   : In PDF format containing information presented at [Homework-Reports.md](https://github.com/eecse4750/e4750_2022Fall_students_repo/wiki/Homework-Reports) , the plots, print and profiling results, and the answers for theory questions. I recommend using A3 Page template since it gives more space to organize code, plots and print results better.
2. PyCUDA solution   : E4750.2022Fall.(uni).assignment3.PyCUDA.py    : In .py format containing the methods and kernel codes, along with comments.
3. PyOpenCL solution : E4750.2022Fall.(uni).assignment3.PyOpenCL.py  : In .py format containing the methods and kernel codes, along with comments.

Replace (uni) with your uni ID. An example report would be titled E4750.2022Fall.zk2172.assignment3.report.PDF 

### Problem set up

Follow the templates to create methods and the kernel code to perform the following

(10 CUDA + 10 OpenCL = 20 Points) 1. Write a Kernel function to perform the above convolution operation without using shared memory or constant memory and name it `conv_gpu`. Define a python method `conv_gpu_naive` to call this kernel without incorporating the modular compile time defines (calling kernel stored in `self.module_naive_gpu` in the template code)

(10 CUDA + 10 OpenCL = 20 Points) 2. Extend this `conv_gpu` function to incorporate shared memory optimization when `Shared_mem_optimized` is defined. Define a python method `conv_gpu_shared_mem` to call this kernel incorporating `Shared_mem_optimized` define (calling kernel stored in `self.module_shared_mem_optimized` in the template code).

(5 CUDA + 5 OpenCL = 10 Points) 3. Further extend this `conv_gpu` function to incorporate shared memory optimization and constant memory optimization when `Shared_mem_optimized` and `Constant_mem_optimized` are defined. Define a python method `conv_gpu_shared_and_constant_mem` to call this kernel incorporating `Shared_mem_optimized` and `Constant_mem_optimized` defines (calling kernel stored in `self.module_const_mem_optimized` in the template code).

(5 CUDA + 5 OpenCL = 10 Points) 4. Write test cases to verify your output with the scipy.signal.convolve2d function from scipy module in python. Name this test function test_conv_pycuda. Write at least one working test case for each function.

(10 CUDA + 10 OpenCL = 20 Points) 5. Record the time taken to execute convolution, including memory transfer operations for the following matrix dimensions: 16 x 16, 64 x 64, 256 x 256, 1024 x 1024, 4096 x 4096. Run each case multiple times and record the average of the time.

### Theory Problems(20 points) 
(4 points) 1. Compare the recorded times against the serial implementation for all the above cases (the three methods). Which approach is faster in PyCuda? Which approach is faster in PyOpenCL? Why is that particular method better than the other.  

(4 points) 2. Profile the three methods in PyCUDA and support your answer for Theory Problem 1 using appropriate screenshots and written inferences.

(4 points) 3. Can this approach be scaled for very big kernel functions? In both cases explain why?  

(4 points) 4. Assuming M > N:

    ```
    Code 1: for(int i=0; i<M; i++) for(int j=0; j<N;j++) val = A[i][j];

    Code 2: for(int j=0; j<N; j++) for(int i=0; i<M;i++) val = A[i][j];
    ```

Will the above two codes give the same performance? Why/Why not?

(4 points) 5. For a tiled 2D convolution, if each output tile is a square with 12 elements on each side and the mask is a square with 5 elements on each side, how many elements are in each input tile?**

a. 12*12 = 144
b. 5*5 = 25
c. (12+2) * (12+2) = 194
d. (12+4) * (12+4) = 256

### Templates

#### PyCuda

```python

import relevant.libraries

class Convolution:
    def __init__(self):
		# Use this space to define the thread dimensions if required, or it can be incorporated into main function
		# You can also define a lambda function to compute grid dimensions if required.
        self.getSourceModule()

    def getSourceModule(self):
        kernel_enable_shared_mem_optimizations = """
        #define Shared_mem_optimized
        """

        kernel_enable_constant_mem_optimizations = """
        #define Constant_mem_optimized
        """

		# STUDENTS SHOULD NOT MODIFY kernel_enable_shared_mem_optimizations, kernel_enable_constant_mem_optimizations, self.module_naive_gpu, self.module_shared_mem_optimized and self.module_const_mem_optimized

		# STUDENTS ARE FREE TO MODIFY ANY PART OF THE CODE INSIDE kernelwrapper as long as the tasks mentioned in the Programming Part are satisfied. The examples shown below are just for reference.
        
        kernelwrapper = r"""
		[TODO: Students to write entire kernel code. An example of using the ifdef and ifndef is shown below. The example can be modified if necessary]
        #ifndef Constant_mem_optimized
        __global__ void conv_gpu(float *a, float *b, float *c, int in_matrix_num_rows, int in_matrix_num_cols, int in_mask_num_rows, int in_mask_num_cols)
        #endif
        #ifdef Constant_mem_optimized
        __global__ void conv_gpu(float *a, float *c, int in_matrix_num_rows, int in_matrix_num_cols, int in_mask_num_rows, int in_mask_num_cols)
        #endif
        {
			[TODO: Perform required tasks, likely some variable declaration, and index calculation, maybe more]

            #ifdef Shared_mem_optimized

			[TODO: Perform some part of Shared memory optimization routine, maybe more]

            #endif

			[TODO: Perform required tasks, mostly relating to the computation part. More #ifdef and #ifndef can be added as necessary]
        }
        """

        self.module_naive_gpu = SourceModule(kernelwrapper)
        self.module_shared_mem_optimized = SourceModule(kernel_enable_shared_mem_optimizations + kernelwrapper)
        self.module_const_mem_optimized = SourceModule(kernel_enable_shared_mem_optimizations + kernel_enable_constant_mem_optimizations + kernelwrapper)

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
```

#### PyOpenCL

```python
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



