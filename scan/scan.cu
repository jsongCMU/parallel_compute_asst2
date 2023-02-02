#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"


extern float toBW(int bytes, float sec);


/* Helper function to round up to a power of 2.
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

const int num_print = 128; // DEBUGGING
void print_host_data(int *data, int size, int num_print)
{
    // Prints data on host
    num_print = (num_print > size) ? size : num_print;
    std::cout << "host data: ";
    for(int i = 0; i < num_print; i++){
        std::cout << data[i] << ", ";
    }
    if(num_print < size)
        std::cout << "...";
    std::cout << "\n";
}

void print_device_data(int *device_data, int size, int num_print)
{
    // Prints data on device
    // print_device_data(device_data+N-num_print, num_print, num_print);
    if(num_print > size){
      // Print whole thing
      int* inarray = new int[size];
      cudaMemcpy(inarray, device_data, size*sizeof(int), cudaMemcpyDeviceToHost);
      std::cout << "(device) ";
      for(int i = 0; i < num_print; i++){
          std::cout << inarray[i] << ", ";
      }
      std::cout << "\n";
    }
    else
    {
      // Print first and last data
      int num_print1 = num_print/2;
      int num_print2 = num_print - num_print1;
      int* inarray1 = new int[num_print1];
      int* inarray2 = new int[num_print2];
      cudaMemcpy(inarray1, device_data, num_print1*sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(inarray2, device_data+size-num_print2, num_print2*sizeof(int), cudaMemcpyDeviceToHost);
      std::cout << "(device) ";
      for(int i = 0; i < num_print1; i++){
          std::cout << inarray1[i] << ", ";
      }
      std::cout << "...\n... ";
      for(int i = 0; i < num_print2; i++){
          std::cout << inarray2[i] << ", ";
      }
      std::cout << "\n";
    }
}

void exclusive_scan_iterative(int* data, int length)
{
    int N = length;
    // upsweep phase.
    std::cout << "Before upsweep: ";
    print_host_data(data, length, num_print);
    for (int twod = 1; twod < N; twod*=2)
    {
        int twod1 = twod*2;
        for(int i = 0; i < N; i += twod1)
            data[i+twod1-1] += data[i+twod-1];
    }
    std::cout << "After upsweep: ";
    print_host_data(data, length, num_print);
    data[N-1] = 0;
    // downsweep phase.
    for (int twod = N/2; twod >= 1; twod /= 2)
    {
        int twod1 = twod*2;
        for(int i = 0; i < N; i += twod1)
        {
            int t = data[i+twod-1];
            data[i+twod-1] = data[i+twod1-1];
            // change twod1 below to twod to reverse prefix sum.
            data[i+twod1-1] += t;
        }
    }
    std::cout << "After downsweep: ";
    print_host_data(data, length, num_print);
    std::cout << "\n";
}

__global__ void upsweep_kernel(int *device_data, int N, int twod)
{
    int twod1 = twod*2;
    long index = (blockIdx.x * blockDim.x + threadIdx.x) * (long)twod1;
    if ((index+twod1-1) < N)
        device_data[index+twod1-1] += device_data[index+twod-1];
}

__global__ void downsweep_kernel(int *device_data, int N, int twod)
{
    int twod1 = twod*2;
    long index = (blockIdx.x * blockDim.x + threadIdx.x) * (long)twod1;
    if((index+twod1-1) < N)
    {
        int t = device_data[index+twod-1];
        device_data[index+twod-1] = device_data[index+twod1-1];
        // change twod1 below to twod to reverse prefix sum.
        device_data[index+twod1-1] += t;
    }
}

void exclusive_scan(int* device_data, int length)
{
    /* TODO
     * Fill in this function with your exclusive scan implementation.
     * You are passed the locations of the data in device memory
     * The data are initialized to the inputs.  Your code should
     * do an in-place scan, generating the results in the same array.
     * This is host code -- you will need to declare one or more CUDA
     * kernels (with the __global__ decorator) in order to actually run code
     * in parallel on the GPU.
     * Note you are given the real length of the array, but may assume that
     * both the data array is sized to accommodate the next
     * power of 2 larger than the input.
     */
    const int N = nextPow2(length);
    const int threadsPerBlock = 512;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    // {
    //   // DEBUGGING
    //   printf("threads = %d, blocks = %d, prod = %d\n", threadsPerBlock, blocks, threadsPerBlock*blocks);
    //   long max_index = (threadsPerBlock*blocks-1)*65536*2;
    //   std::cout << "Max index  = " << max_index << "\n";
    //   printf("Sizeof(long) = %ld\n", sizeof(long long));
    // }
    // upsweep phase.
    // {
    //     // DEBUGGING
    //     cudaMemset(device_data+length, 0, (N-length)*sizeof(int));
    //     std::cout << "\tBefore upsweep: ";
    //     print_device_data(device_data, N, num_print); 
    // }
    for (int twod = 1; twod < N; twod*=2)
    {
        upsweep_kernel<<<blocks, threadsPerBlock>>>(device_data, N, twod);
        // {
        //     // DEBUGGING
        //     printf("\tDuring upsweep (twod = %d): ", twod);
        //     print_device_data(device_data, N, num_print);
        // }
    }
    // {
    //     // DEBUGGING
    //     std::cout << "\tAfter upsweep: ";
    //     print_device_data(device_data, N, num_print);
    // }
    // Zero unused memory
    cudaMemset(device_data+length-1, 0, (N-length+1)*sizeof(int));
    // {
    //     // DEBUGGING
    //     std::cout << "\tAfter zeroing: ";
    //     print_device_data(device_data, N, num_print);
    // }

    // downsweep phase.
    for (int twod = N/2; twod >= 1; twod /= 2)
    {
        downsweep_kernel<<<blocks, threadsPerBlock>>>(device_data, N, twod);
        // {
        //     // DEBUGGING
        //     printf("\tDuring downsweep (twod = %d): ", twod);
        //     print_device_data(device_data, N, num_print);
        // }
    }
    // {
    //     // DEBUGGING
    //     std::cout << "\tAfter downsweep: ";
    //     print_device_data(device_data, N, num_print);
    //     std::cout << "\n";
    // }

}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int* inarray, int* end, int* resultarray)
{
    // {
    //     // DEBUGGING
    //     int length = end-inarray;
    //     int* test_array = new int[length];
    //     memcpy(test_array, inarray, length * sizeof(int));
    //     exclusive_scan_iterative(test_array, length);
    // }

    int* device_data;
    // We round the array size up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness.
    // You may have an easier time in your implementation if you assume the
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    cudaMalloc((void **)&device_data, sizeof(int) * rounded_length);

    cudaMemcpy(device_data, inarray, (end - inarray) * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_data, end - inarray);

    // Wait for any work left over to be completed.
    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    cudaMemcpy(resultarray, device_data, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);
    return overallDuration;
}

/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}



int find_peaks(int *device_input, int length, int *device_output) {
    /* TODO:
     * Finds all elements in the list that are greater than the elements before and after,
     * storing the index of the element into device_result.
     * Returns the number of peak elements found.
     * By definition, neither element 0 nor element length-1 is a peak.
     *
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if
     * it requires that. However, you must ensure that the results of
     * find_peaks are correct given the original length.
     */
    return 0;
}



/* Timing wrapper around find_peaks. You should not modify this function.
 */
double cudaFindPeaks(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    int result = find_peaks(device_input, length, device_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}


void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
