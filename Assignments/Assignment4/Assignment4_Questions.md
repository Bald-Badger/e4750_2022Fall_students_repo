# EECS E4750: Heterogeneous Computing for Signal and Data Processing (Fall 2022)

## Assignment-4: Scan

Total points: 100

### References to help you with assignment
* [PyCuda Examples](https://github.com/inducer/pycuda/tree/main/examples)
* [PyOpenCL Examples](https://github.com/inducer/pyopencl/tree/main/examples)
* [NVidia Blog](https://developer.nvidia.com/blog/tag/cuda/)
* [L2 norm](https://www.digitalocean.com/community/tutorials/norm-of-vector-python)

### Primer
For this assignment you will be working on inclusive parallel scan on a 1D list (That is to implement parallel algorithm for all prefix sum). The scan operator will be the addition (plus) operator. There are many uses for all-prefix-sums, including, but not limited to sorting, lexical analysis, string comparison, polynomial evaluation, stream compaction, and building histograms and data structures (graphs, trees, etc.) in parallel. Your kernel should be able to handle input lists of arbitrary length. To simplify the lab, the input list will be at most of length 134215680 (which can be split up as $2048 × 65535$) elements. This means that the computation can be performed using only one kernel launch. The boundary condition can be handled by filling “identity value (0 for sum)” into the shared memory of the last block when the length is not a multiple of the thread block size. Example: All-prefix-sums operation on the array [3 1 7 0 4 1 6 3], would return [3 4 11 11 15 16 22 25].

### Programming Part (70 points)

### Programming Task 1. 1-D Scan - Naive Python algorithm
1. Implement scan on a 1D list (prefix sum) in Python. (5 points)
2. Write 2 test cases (pair of input and expected output values) using some fixed values to confirm that your code is working correctly. Length of the input in the test case should be 5 elements. (You can also use the first 5 elements of the above example in primer as one test case) (5 points)

### Programming Task 2. 1-D Scan - Programing in PyCuda and PyOpenCL 
1.  Implement a naive parallel scan algorithm using both PyOpenCL and PyCUDA. The input and output are the same as those of the serial one. ** Analyze the time complexity.** Hint: Check the course materials for naive scan algorithm. (10 points) 

2. Implement a work efficient parallel scan algorithm using both PyOpenCL and PyCUDA. The input and output remain the same. Analyze the time complexity. Hint: Check the course materials for 'work efficient'. (10 CUDA + 10 OpenCL = 20 points)

3. Write test cases to verify your output with naive python algorithm. Input cases of lengths 128, 2048, 262144 (equals 128 X 2048), 4194304 (equals $2048^2$), 134215680 (equals 65535 X 2048). (5 CUDA + 5 OpenCL = 10 points)

4. For the input cases of length 128, 2048, 262144, 4194304, 134215680 record the time taken(including memory transfer) for the three functions (naive python, naive parallel, and work efficient parallel). Provide a graph of your time observations in the report and compare the performance of the algorithms, compare both space and time complexity. (10 CUDA + 10 OpenCL = 20 points)

### Theory Problems (30 points) 
1. For the work efficient scan kernel based on reduction trees and inverse reduction trees, assume that we have 2048 elements (each block has BLOCK_SIZE=1024 threads) in each section and warp size is 32, how many warps in each block will have control divergence during the reduction tree phase iteration where stride is 16? For your convenience, the relevant code fragment from the kernel is given below: (5 points)

```
for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride = stride*2) {
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index < 2*BLOCK_SIZE) {XY[index] += XY[index-stride];}
    __syncthreads();
}
```
(A) 0
(B) 1
(C) 16
(D) 32

2. Consider that NVIDIA GPUs execute warps of 32 parallel threads using SIMT. What's the difference between SIMD and SIMT? What is the worst choice as the number of threads per block to chose in this case among the following and why? (5 points)
(A) 1  
(B) 16  
(C) 32  
(D) 64  

3. What is a bank conflict? Give an example for bank conflict. (5 points) 

4. For the following basic reduction kernel code fragment, if the block size is 1024 and warp size is 32, how many warps in a block will have divergence during the iteration where stride is equal to 1? (5 points)

```
unsigned int t = threadIdx.x;
Unsigned unsigned int start = 2*blockIdx.x*blockDim.x;
partialSum[t] = input[start + t];
partialSum[blockDim.x+t] = input[start+ blockDim.x+t];
for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
{
    __syncthreads();
    if (t % stride == 0) {partialSum[2*t]+= partialSum[2*t+stride];}
}
```
(A) 0
(B) 1
(C) 16
(D) 32

5. Consider the following code for finding the sum of all elements in a vector. The following code doesn't always work correctly explain why? Also suggest how to fix this? (Hint: use atomics.) (5 points) 
```
__global__ void vectSum(int* d_vect,size_t size, int* result){
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < size){
        *result+=d_vect[tid];
        tid+=blockDim.x * gridDim.x;
    }
}
```

6. Consider the work-efficient parallel sum-scan algorithm you implemented. If we would like to compute the L2 norm (square root of sum of squares of all elements) of the vector along with the sum-scan results, using the same kernel, what would be the optimal way to implement it? (Assume the time to compute square root in the end is negligible, focus on getting sum of squares of all elements) (5 points)

### Templates for PyCuda and PyOpenCL
A common template for PyCuda and PyOpenCL is shown below.

```python
import relevant.libraries

class PrefixSum:
  def __init__(self):
	pass

  def prefix_sum_python(self):
	# implement this, note you can change the function signature (arguments and return type)
	pass

  def prefix_sum_gpu_naive(self):
	# implement this, note you can change the function signature (arguments and return type)
	pass

  def prefix_sum_gpu_work_efficient(self):
	# implement this, note you can change the function signature (arguments and return type)
	pass

  def test_prefix_sum_python(self):
	# implement this, note you can change the function signature (arguments and return type)
	pass

  def test_prefix_sum_gpu_naive(self):
	# implement this, note you can change the function signature (arguments and return type)
	pass

  def test_prefix_sum_gpu_work_efficient(self):
	# implement this, note you can change the function signature (arguments and return type)
	pass

```