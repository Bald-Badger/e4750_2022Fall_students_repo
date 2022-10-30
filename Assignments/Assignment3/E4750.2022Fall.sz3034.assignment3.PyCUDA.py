# reference apart from links listed in course wiki:
# https://stackoverflow.com/questions/40247760/scipy-convolve2d-outputs-wrong-values
# https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf


import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.driver
from pycuda.compiler import SourceModule
import time
import matplotlib.pyplot as plt
from scipy import signal
import pickle


class Convolution:
    def __init__(self):
		# Use this space to define the thread dimensions if required, or it can be incorporated into main function
		# You can also define a lambda function to compute grid dimensions if required.
        self.getSourceModule()
        self.threads_per_block_x = 32 # max kernel size is 1024 = 32 *
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
        #ifdef Constant_mem_optimized
        // use all the constant memory available, supports up to 128*128 kernel
        __constant__ float b[16384];
        __global__ void conv_gpu(
            float *a,               float *c, 
            int in_matrix_num_rows, int in_matrix_num_cols, 
            int in_mask_num_rows,   int in_mask_num_cols,
            int pad_row,            int pad_col
        )
        #else
        __global__ void conv_gpu(
            float *a,               float *b,               float *c, 
            int in_matrix_num_rows, int in_matrix_num_cols, 
            int in_mask_num_rows,   int in_mask_num_cols,
            int pad_row,            int pad_col
        )
        #endif
        {
			// [TODO: Perform required tasks, likely some variable declaration, and index calculation, maybe more]
            // dimension out output matrix
            int c_rows = in_matrix_num_rows - in_mask_num_rows + 2 * pad_row + 1;
            int c_cols = in_matrix_num_cols - in_mask_num_cols + 2 * pad_col + 1;

            // thread ID, block ID
            // I have many grids, each grid have many blocks, a block have many threads
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int bx = blockIdx.x;
            int by = blockIdx.y;
            
            // row number and col number current thread is working on
            int row = by*blockDim.y + ty;
            int col = bx*blockDim.x + tx;

            float sum;
            int mat_col_index, mat_row_index;
            float mat_value, ker_value;
            __syncthreads();
            #ifdef Shared_mem_optimized
                #ifndef Constant_mem_optimized
                    // [TODO: Perform some part of Shared memory optimization routine, maybe more]
                    __shared__ float b_shared[256];
                    int index = (tx + 1) * (ty + 1) - 1;
                    if (index < (in_mask_num_rows * in_mask_num_rows)) {
                        b_shared[index] = b[index];
                        // printf("placed index %d with %f, tx:%d, ty:%d\n", index, b_shared[index], tx, ty);
                    }
                    __syncthreads();

                    /*
                        int blockId = blockIdx.x + blockIdx.y * gridDim.x;
                        int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
                    */
                #endif
            #endif

			// [TODO: Perform required tasks, mostly relating to the computation part. More #ifdef and #ifndef can be added as necessary]
            if (row < c_rows && col < c_cols) {
                sum = 0;
                for (int i = row; i < row + in_mask_num_rows; i++) {
                    for (int j = col; j < col + in_mask_num_cols; j++) {
                        mat_row_index = i - pad_row;
                        mat_col_index = j - pad_col;
                        if (
                            mat_col_index < 0 || 
                            mat_row_index < 0 || 
                            mat_col_index >= in_matrix_num_cols || 
                            mat_row_index >= in_matrix_num_rows
                        ) {
                            mat_value = 0;
                        } else {
                            int index = mat_row_index * in_matrix_num_cols + mat_col_index;
                            mat_value = a[index];
                        }
                        #ifdef Shared_mem_optimized
                            #ifdef Constant_mem_optimized
                                ker_value = b[(i - row) * in_mask_num_cols + j - col];
                            #else
                                ker_value = b_shared[(i - row) * in_mask_num_cols + j - col];
                            #endif
                        #else
                            ker_value = b[(i - row) * in_mask_num_cols + j - col];
                        #endif
                        sum += (mat_value * ker_value);
                        if (row == 0 && col == 1) {
                            // printf("partial: %f x %f\n", mat_value, ker_value);
                        }
                    }
                }
                c[row * c_cols + col] = sum;
            }
        }
        """

        self.module_naive_gpu = SourceModule(kernelwrapper)
        self.module_shared_mem_optimized = SourceModule(kernel_enable_shared_mem_optimizations + kernelwrapper)
        self.module_const_mem_optimized = SourceModule(kernel_enable_shared_mem_optimizations + kernel_enable_constant_mem_optimizations + kernelwrapper)

        # If you wish, you can also include additional compiled kernels and compile-time defines that 
        # you may use for debugging without modifying the above three compiled kernel.


    # warpper function for all 3 modes
    def __conv_gpu(self,
                 inputmatrix,           inputmask, 
                 input_matrix_numrows,  input_matrix_numcolumns, 
                 input_mask_numrows,    input_mask_numcolumns, 
                 pad_row=None,          pad_col=None, 
                 mode=None,             reverse_kernel=True,
                 debug=False
                ):

        mode_list = [
            'conv_gpu_naive',
            'conv_gpu_shared_mem',
            'conv_gpu_shared_and_constant_mem'
        ]
        
        if reverse_kernel:
            inputmask = inputmask[::-1, ::-1]
            input_mask_numrows = inputmask.shape[0]
            input_mask_numcols = inputmask.shape[0]

        if pad_row is None:
            pad_row = input_mask_numrows - 1
        if pad_col is None:
            pad_col = input_mask_numcolumns - 1
        
        start = cuda.Event()
        compute = cuda.Event()
        finish = cuda.Event()

        a = inputmatrix.astype(np.float32)
        b = inputmask.astype(np.float32)
        inputmatrix_shape = inputmatrix.shape
        inputmask_shape = inputmask.shape

        # sanity check
        assert mode in mode_list
        
        assert inputmatrix_shape[0] == input_matrix_numrows
        assert inputmatrix_shape[1] == input_matrix_numcolumns
        assert inputmask_shape[0]   == input_mask_numrows
        assert inputmask_shape[1]   == input_mask_numcolumns

        c_rows = input_matrix_numrows - input_mask_numrows + 2 * pad_row + 1
        c_cols = input_matrix_numcolumns - input_mask_numcolumns + 2 * pad_col + 1

        c = np.zeros((c_rows, c_cols), dtype=np.float32)

        start.record()
        
        a_gpu = cuda.mem_alloc(a.shape[0] * a.shape[1] * a.dtype.itemsize)
        c_gpu = cuda.mem_alloc(c.shape[0] * c.shape[1] * c.dtype.itemsize)
        
        if mode == 'conv_gpu_shared_and_constant_mem':
            b_gpu = self.module_const_mem_optimized.get_global("b")[0]
        else:
            b_gpu = cuda.mem_alloc(b.shape[0] * b.shape[1] * b.dtype.itemsize)
        
        cuda.memcpy_htod(a_gpu, a)
        cuda.memcpy_htod(b_gpu, b)

        blockDim  = (
            self.threads_per_block_x, 
            self.threads_per_block_y, 
            self.threads_per_block_z
        )

        gridDim   = (
            c.shape[0] // self.threads_per_block_x + 1, 
            c.shape[1] // self.threads_per_block_y + 1, 
            1
        )
        
        if debug:
            print("matrix shape: {}".format(a.shape))
            print("kernel shape: {}".format(b.shape))
            print("expect result shape: {}".format(c.shape))
            print("blok dim: {}".format(blockDim))
            print("grid dim: {}".format(gridDim))
        
        if mode == mode_list[0]:    # conv_gpu_naive
            func = self.module_naive_gpu.get_function("conv_gpu")
        elif mode == mode_list[1]:  # conv_gpu_shared_mem
            func = self.module_shared_mem_optimized.get_function("conv_gpu")
        elif mode == mode_list[2]:  # conv_gpu_shared_and_constant_mem
            func = self.module_const_mem_optimized.get_function("conv_gpu")
            
        if mode == mode_list[2]:    # conv_gpu_shared_and_constant_mem
            func(   
                a_gpu, 
                c_gpu, 
                np.int32(input_matrix_numrows), 
                np.int32(input_matrix_numcolumns), 
                np.int32(input_mask_numrows), 
                np.int32(input_mask_numcolumns), 
                np.int32(pad_row), 
                np.int32(pad_col), 
                block = blockDim,  
                grid  = gridDim
            )
        else:
            func(   
                a_gpu, 
                b_gpu, 
                c_gpu, 
                np.int32(input_matrix_numrows), 
                np.int32(input_matrix_numcolumns), 
                np.int32(input_mask_numrows), 
                np.int32(input_mask_numcolumns), 
                np.int32(pad_row), 
                np.int32(pad_col), 
                block = blockDim,  
                grid  = gridDim
            )

        compute.record()
        compute.synchronize()

        cuda.memcpy_dtoh(c, c_gpu)
        finish.record()
        finish.synchronize()

        return (c, start.time_till(finish)*1000) # in us


    def conv_gpu_naive(self, inputmatrix, inputmask, input_matrix_numrows, input_matrix_numcolumns, input_mask_numrows, input_mask_numcolumns, pad_row=None, pad_col=None):
		# Write methods to call self.module_naive_gpu for computing convolution and return the results and time taken. 
        # The above input variable names like inputmask, input_matrix_numrows, etc can be changed as per student requirements.
        return self.__conv_gpu(
            inputmatrix,            inputmask, 
            input_matrix_numrows,   input_matrix_numcolumns, 
            input_mask_numrows,     input_mask_numcolumns, 
            pad_row,                pad_col, 
            "conv_gpu_naive"
        )


    def conv_gpu_shared_mem(self, inputmatrix, inputmask, input_matrix_numrows, input_matrix_numcolumns, input_mask_numrows, input_mask_numcolumns, pad_row=None, pad_col=None):
        # Write methods to call self.module_shared_mem_optimized for computing convolution and return the results and time taken. 
        # The above input variable names like inputmask, input_matrix_numrows, etc can be changed as per student requirements.
        return self.__conv_gpu(
            inputmatrix,            inputmask, 
            input_matrix_numrows,   input_matrix_numcolumns, 
            input_mask_numrows,     input_mask_numcolumns, 
            pad_row,                pad_col, 
            "conv_gpu_shared_mem"
        )


    def conv_gpu_shared_and_constant_mem(self, inputmatrix, inputmask, input_matrix_numrows, input_matrix_numcolumns, input_mask_numrows, input_mask_numcolumns, pad_row=None, pad_col=None):
        # Write methods to call self.module_const_mem_optimized for computing convolution and return the results and time taken. 
        # The above input variable names like inputmask, input_matrix_numrows, etc can be changed as per student requirements.
        return self.__conv_gpu(
            inputmatrix,            inputmask, 
            input_matrix_numrows,   input_matrix_numcolumns, 
            input_mask_numrows,     input_mask_numcolumns, 
            pad_row,                pad_col, 
            "conv_gpu_shared_and_constant_mem"
        )


    def test_conv_pycuda(self, inputmatrix, inputmask):
        # Write methods to perform convolution on the same dataset using scipy's convolution methods running on CPU and return the results and time. 
        # Students are free to experiment with different variable names in place of inputmatrix and inputmask.
        start = time.time()
        result = signal.convolve2d(inputmatrix, inputmask)
        end = time.time()
        return (result, (end - start) * 1000000) #n us


def test_conv_pycuda(mrows = 4096, mcols = 4096, krows = 5, kcols = 5):
    M = np.random.rand(mrows, mcols).astype(np.float32)
    K = np.random.rand(krows, kcols).astype(np.float32)
    computer = Convolution()
    cr, tr = computer.test_conv_pycuda(M, K)
    c0, t0 = computer.conv_gpu_naive(M, K, mrows, mcols, krows, kcols)
    c1, t1 = computer.conv_gpu_shared_mem(M, K, mrows, mcols, krows, kcols)
    c2, t2 = computer.conv_gpu_shared_and_constant_mem(M, K, mrows, mcols, krows, kcols)
    
    r0 = np.isclose(c0, cr).flatten()
    r1 = np.isclose(c1, cr).flatten()
    r2 = np.isclose(c2, cr).flatten()
    
    accuracy = [0, 0, 0]
    for r in r0:
        if r == True:
            accuracy[0] += 1
    for r in r1:
        if r == True:
            accuracy[1] += 1
    for r in r2:
        if r == True:
            accuracy[2] += 1
    accuracy[0] /= len(r0)
    accuracy[1] /= len(r1)
    accuracy[2] /= len(r2)
    accuracy[0] *= 100
    accuracy[1] *= 100
    accuracy[2] *= 100
    
    print(
        "testing conv2d with matrix size of {mrows} x {mcols}, kernel size of {krows} x {kcols}".format(
            mrows=mrows, mcols=mcols, krows=krows, kcols=kcols
        )
    )
    
    print(
        "the accuracy of the 3 kernel compared with scipy method are: {:.2f}%, {:.2f}%, {:.2f}%".format(
            accuracy[0], accuracy[1], accuracy[2]
        )
    )
    
    print(
        "the time it takes for scipy and each kernel to finish in μs are: {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
            tr, t0, t1, t2
        )
    )


def record():
    matrix_size_list = [16, 64, 256, 1024, 4096]
    kernel_size = 5
    iters = 10 # change to high value in final run
    
    refer_times = np.array([])
    naive_times = np.array([])
    share_times = np.array([])
    const_times = np.array([])
    
    computer = Convolution()
    
    for msize in matrix_size_list:
        refer_time = 0
        naive_time = 0
        share_time = 0
        const_time = 0
        for i in range(iters):
            M = np.random.rand(msize, msize).astype(np.float32)
            K = np.random.rand(kernel_size, kernel_size).astype(np.float32)
            
            cr, tr = computer.test_conv_pycuda                  (M, K)
            c0, t0 = computer.conv_gpu_naive                    (M, K, msize, msize, kernel_size, kernel_size)
            c1, t1 = computer.conv_gpu_shared_mem               (M, K, msize, msize, kernel_size, kernel_size)
            c2, t2 = computer.conv_gpu_shared_and_constant_mem  (M, K, msize, msize, kernel_size, kernel_size)
            
            refer_time += tr
            naive_time += t0
            share_time += t1
            const_time += t2
            
            refer_time /= iters
            naive_time /= iters
            share_time /= iters
            const_time /= iters
            
        refer_times = np.append(refer_times, refer_time)
        naive_times = np.append(naive_times, naive_time)
        share_times = np.append(share_times, share_time)
        const_times = np.append(const_times, const_time)
        filename = 'cuda_data.pkl'
        f = open (filename,'wb')
        pickle.dump([
            matrix_size_list, 
            refer_times, 
            naive_times, 
            share_times, 
            const_times
        ], f)
        f.close()


def plot():
    f = open('cuda_data.pkl','rb')
    time_list   = pickle.load(f)
    
    # log2 normalization
    time_list = np.log2(time_list)
    
    size_list   = time_list[0]
    refer_times = time_list[1]
    naive_times = time_list[2]
    share_times = time_list[3]
    const_times = time_list[4]
    
    
    
    plt.figure(0)
    plt.plot(size_list, refer_times, label="cpu")
    plt.plot(size_list, naive_times, label="naive")
    plt.plot(size_list, share_times, label="shared")
    plt.plot(size_list, const_times, label="constant")
    plt.legend()
    plt.xlabel("matrix width in 2 log scale (2^x)")
    plt.ylabel("time takes in 2 log scale (2^y μs)")
    plt.title("PyCUDA: time it takes for each workload in 2 log scale")
    plt.savefig("cuda_plot_with_cpu.png")
    
    plt.figure(1)
    plt.plot(size_list, naive_times, label="naive")
    plt.plot(size_list, share_times, label="shared")
    plt.plot(size_list, const_times, label="constant")
    plt.legend()
    plt.xlabel("matrix width in 2 log scale (2^x)")
    plt.ylabel("time takes in 2 log scale (2^y μs)")
    plt.title("PyCUDA: time it takes for each workload in 2 log scale without CPU")
    plt.savefig("cuda_plot_without_cpu.png")
    

def main():
    test_conv_pycuda()
    record()
    plot()


if __name__ == "__main__":
    # Main code
    # Write methods to perform the computations, get the timings for all the tasks mentioned in programming sections
    # and also comparing results and mentioning if there is a sum mismatch. Students can experiment with numpy.math.isclose function.
    main()
