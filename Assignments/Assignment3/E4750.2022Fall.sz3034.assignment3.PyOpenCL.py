import numpy as np
import pyopencl as cl
import pyopencl.array as pycl_array
import time
import matplotlib.pyplot as plt
import pickle
from scipy import signal


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

		# STUDENTS SHOULD NOT MODIFY kernel_enable_shared_mem_optimizations, 
        # kernel_enable_constant_mem_optimizations, self.module_naive_gpu, 
        # self.module_shared_mem_optimized and self.module_const_mem_optimized

		# STUDENTS ARE FREE TO MODIFY ANY PART OF THE CODE INSIDE kernelwrapper
        # as long as the tasks mentioned in the Programming Part are satisfied. 
        # The examples shown below are just for reference.

        kernel_code = r"""
            /*
            [TODO: Students to write entire kernel code. 
            An example of using the ifdef and ifndef is shown below. 
            The example can be modified if necessary]
            */

            #if defined(Constant_mem_optimized)
            __kernel void conv_gpu (
                __global float* a,                      __constant float* b,                    __global float* c, 
                const unsigned int in_matrix_num_rows,  const unsigned int in_matrix_num_cols, 
                const unsigned int in_mask_num_rows,    const unsigned int in_mask_num_cols,
                const unsigned int pad_row,             const unsigned int pad_col
            )
            #elif defined(Shared_mem_optimized)
            __kernel void conv_gpu (
                __global float* a,                      __local float* b,                      __global float* c, 
                const unsigned int in_matrix_num_rows,  const unsigned int in_matrix_num_cols, 
                const unsigned int in_mask_num_rows,    const unsigned int in_mask_num_cols,
                const unsigned int pad_row,             const unsigned int pad_col
            )
            #else
            __kernel void conv_gpu (
                __global float* a,                      __global float* b,                      __global float* c, 
                const unsigned int in_matrix_num_rows,  const unsigned int in_matrix_num_cols, 
                const unsigned int in_mask_num_rows,    const unsigned int in_mask_num_cols,
                const unsigned int pad_row,             const unsigned int pad_col
            )
            #endif
            {
                // [TODO: Perform required tasks, likely some variable declaration, and index calculation, maybe more]
                // dimension out output matrix
                int c_rows = in_matrix_num_rows - in_mask_num_rows + 2 * pad_row + 1;
                int c_cols = in_matrix_num_cols - in_mask_num_cols + 2 * pad_col + 1;
                
                int row = get_global_id(0);
                int col = get_global_id(1);
                
                float sum = 0;
                int mat_row_index, mat_col_index;
                float mat_value, ker_value;
                
                int probe_row = 0;
                int probe_col = 0;

                // printf("Hello from global id %d x %d\n", row, col);

                /*
                [TODO: Perform required tasks, mostly relating to the computation part. 
                More #ifdef and #ifndef can be added as necessary]
                */
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
                        ker_value = b[(i - row) * in_mask_num_cols + j - col];
                        sum += (mat_value * ker_value);
                        if (row == probe_row && col == probe_col) {
                            //printf("partial sum: %f x %f = %f\n", mat_value, ker_value, mat_value * ker_value);
                        }
                    }
                }
                
                c[row * c_cols + col] = sum;
                __syncthreads();
            }
        """

        self.module_naive_gpu = cl.Program(
            self.ctx, 
            kernel_code
        ).build()
        self.module_shared_mem_optimized = cl.Program(
            self.ctx, 
            kernel_enable_shared_mem_optimizations + kernel_code
        ).build()
        self.module_const_mem_optimized = self.prg = cl.Program(
            self.ctx, 
            kernel_enable_shared_mem_optimizations + kernel_enable_constant_mem_optimizations + kernel_code
        ).build()

        # If you wish, you can also include additional compiled kernels and compile-time defines that 
        # you may use for debugging without modifying the above three compiled kernel.

    def __conv_gpu (
                    self,
                    inputmatrix,                inputmask, 
                    input_matrix_numrows=None,  input_matrix_numcolumns=None, 
                    input_mask_numrows=None,    input_mask_numcolumns=None, 
                    pad_row=None,               pad_col=None, 
                    mode=None,                  reverse_kernel=True,
                    debug=False
                ):

        mode_list = [
            'conv_gpu_naive',
            'conv_gpu_shared_mem',
            'conv_gpu_shared_and_constant_mem'
        ]
        
        # sanity check
        assert mode in mode_list
        
        if reverse_kernel:
            inputmask = inputmask[::-1, ::-1]
        if debug:
            print("actual kernel:")
            print(inputmask)
            
        if input_matrix_numrows is None:
            input_matrix_numrows = inputmatrix.shape[0]
        if input_matrix_numcolumns is None:
            input_matrix_numcolumns = inputmatrix.shape[1]
        if input_mask_numrows is None:
            input_mask_numrows = inputmask.shape[0]
        if input_mask_numcolumns is None:
            input_mask_numcolumns = inputmask.shape[1]
            
        if pad_row is None:
            pad_row = input_mask_numrows - 1
        if pad_col is None:
            pad_col = input_mask_numcolumns - 1
            
        a = inputmatrix.astype(np.float32)
        b = inputmask.astype(np.float32)
        
        c_rows = input_matrix_numrows - input_mask_numrows + 2 * pad_row + 1
        c_cols = input_matrix_numcolumns - input_mask_numcolumns + 2 * pad_col + 1
        
        start = time.time()
        
        a_gpu = pycl_array.to_device(self.queue, a)
        b_gpu = pycl_array.to_device(self.queue, b)
        c_gpu = pycl_array.zeros(self.queue, shape=(c_rows, c_cols), dtype=np.float32)
        
        if debug:
            print("matrix shape: {}"        .format(a_gpu.shape))
            print("kernel shape: {}"        .format(b_gpu.shape))
            print("expect result shape: {}" .format(c_gpu.shape))
            
        kernel = self.module_naive_gpu.conv_gpu
        
        kernel (
            self.queue,                     (c_rows,c_cols),                    None, 
            a_gpu.data,                     b_gpu.data,                         c_gpu.data,
            np.int32(input_matrix_numrows), np.int32(input_matrix_numcolumns),
            np.int32(input_mask_numrows),   np.int32(input_mask_numcolumns),
            np.int32(pad_row),              np.int32(pad_col)
        )
        c = c_gpu.get()
        end = time.time()
        return c, (end - start) * 1e6 # in us


    def conv_gpu_naive (
                        self, 
                        inputmatrix,                inputmask, 
                        input_matrix_numrows=None,  input_matrix_numcolumns=None, 
                        input_mask_numrows=None,    input_mask_numcolumns=None, 
                        pad_row=None,               pad_col=None
                    ):
		# Write methods to call self.module_naive_gpu for computing convolution and
        # return the results and time taken. 
        # The above input variable names like inputmask, input_matrix_numrows, etc 
        # can be changed as per student requirements.

        return self.__conv_gpu(
            inputmatrix,            inputmask, 
            input_matrix_numrows,   input_matrix_numcolumns, 
            input_mask_numrows,     input_mask_numcolumns, 
            pad_row,                pad_col, 
            "conv_gpu_naive",       debug=True
        )


    def conv_gpu_shared_mem (
                                self, 
                                inputmatrix,                inputmask, 
                                input_matrix_numrows=None,  input_matrix_numcolumns=None, 
                                input_mask_numrows=None,    input_mask_numcolumns=None, 
                                pad_row=None,               pad_col=None
                            ):
        # Write methods to call self.module_shared_mem_optimized for computing convolution 
        # and return the results and time taken. 
        # The above input variable names like inputmask, input_matrix_numrows, etc 
        # can be changed as per student requirements.
        return self.__conv_gpu(
            inputmatrix,            inputmask, 
            input_matrix_numrows,   input_matrix_numcolumns, 
            input_mask_numrows,     input_mask_numcolumns, 
            pad_row,                pad_col, 
            "conv_gpu_shared_mem"
        )


    def conv_gpu_shared_and_constant_mem (
                                            self, 
                                            inputmatrix,                inputmask, 
                                            input_matrix_numrows=None,  input_matrix_numcolumns=None, 
                                            input_mask_numrows=None,    input_mask_numcolumns=None, 
                                            pad_row=None,               pad_col=None
                                        ):
        # Write methods to call self.module_const_mem_optimized for 
        # computing convolution and return the results and time taken. 
        # The above input variable names like inputmask, input_matrix_numrows, etc 
        # can be changed as per student requirements.
        return self.__conv_gpu(
            inputmatrix,            inputmask, 
            input_matrix_numrows,   input_matrix_numcolumns, 
            input_mask_numrows,     input_mask_numcolumns, 
            pad_row,                pad_col, 
            "conv_gpu_shared_and_constant_mem"
        )


    def conv_cpu(self, inputmatrix, inputmask):
        start = time.time()
        result = signal.convolve2d(inputmatrix, inputmask)
        end = time.time()
        return (result, (end - start) * 1000000) #in us


def test_conv_pycuda(mrows = 4096, mcols = 4096, krows = 5, kcols = 5):
    # Write methods to perform convolution on the same dataset using 
    # scipy's convolution methods running on CPU and return the results and time. 
    # Students are free to experiment with different 
    # variable names in place of inputmatrix and inputmask.
    M = np.random.rand(mrows, mcols).astype(np.float32)
    K = np.random.rand(krows, kcols).astype(np.float32)
    computer = Convolution()
    cr, tr = computer.conv_cpu                          (M, K)
    c0, t0 = computer.conv_gpu_naive                    (M, K, mrows, mcols, krows, kcols)
    c1, t1 = computer.conv_gpu_shared_mem               (M, K, mrows, mcols, krows, kcols)
    c2, t2 = computer.conv_gpu_shared_and_constant_mem  (M, K, mrows, mcols, krows, kcols)
    
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



def myTest():
    M = np.array(
        [
            [ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11],
            [12, 13, 14, 15]
        ], dtype=np.float32
    )
    
    K = np.array(
        [
            [1, 2],
            [3, 4]
        ], dtype=np.float32
    )

    computer = Convolution()
    
    ref, tr = computer.conv_cpu(M, K)
    sol, ts = computer.conv_gpu_naive(M, K)
    print(M)
    print(K)
    print(ref)
    print(sol)
    print(np.isclose(ref, sol))


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
            
            cr, tr = computer.conv_cpu                          (M, K)
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
        filename = 'pycl_data.pkl'
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
    plt.title("PyOpenCL: time it takes for each workload in 2 log scale")
    plt.savefig("pycl_plot_with_cpu.png")
    
    plt.figure(1)
    plt.plot(size_list, naive_times, label="naive")
    plt.plot(size_list, share_times, label="shared")
    plt.plot(size_list, const_times, label="constant")
    plt.legend()
    plt.xlabel("matrix width in 2 log scale (2^x)")
    plt.ylabel("time takes in 2 log scale (2^y μs)")
    plt.title("PyOpenCL: time it takes for each workload in 2 log scale without CPU")
    plt.savefig("pycl_plot_without_cpu.png")


def main():
    #myTest()
    test_conv_pycuda()
    record()
    plot()


if __name__ == "__main__":
     # Main code
    # Write methods to perform the computations, 
    # get the timings for all the tasks mentioned in programming sections
    # and also comparing results and mentioning if there is a sum mismatch. 
    # Students can experiment with numpy.math.isclose function.
    main()
