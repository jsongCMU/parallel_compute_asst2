#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>



#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

#include <thrust/scan.h>
#include <thrust/device_ptr.h>

////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

// Divide screen into segments
#define SCREEN_X_SEGMENTS (10)
#define SCREEN_Y_SEGMENTS (SCREEN_X_SEGMENTS)
// Divide circles into batches
#define CIRCLE_BATCH_SIZE (1024)
// Blocking
#define PIECES_PER_SEGMENT (23)
#define ROWS_PER_PIECE (5)

// This stores the global constants
struct GlobalConstants {

    SceneName sceneName;

    int numberOfCircles;

    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
    int* circleSegmentMatrix;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// Read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// Color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// Include parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update positions of fireworks
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = M_PI;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // Determine the firework center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // Update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // Firework sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // Compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // Compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // Random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // Travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // Place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the position of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numberOfCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// Move the snowflake animation forward one time step.  Update circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // Load from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // Hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // Add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // Drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // Update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // Update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // If the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // Restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // Store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// Given a pixel and a circle, determine the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(float2 pixelCenter, float3 p, float rad, float3 color, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;
    float maxDist = rad * rad;

    // Circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // There is a non-zero contribution.  Now compute the shading value

    // Suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks, etc., to implement the conditional.  It
    // would be wise to perform this logic outside of the loops in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // Simple: each circle has an assigned color
        rgb = color;
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // Global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

__global__ void kernelRenderPixel(int* rel_circles_idxs, int* num_rel_circles,  int batch_idx, int batch_size) {
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    // Compute segment specific pointers
    int seg_x_idx = blockIdx.y;
    int seg_y_idx = blockIdx.x;
    int piece_idx = blockIdx.z;
    int seg_x_dim = imageWidth/SCREEN_X_SEGMENTS;
    int seg_y_dim = imageHeight/SCREEN_Y_SEGMENTS;
    int* cur_seg_rel_circles = rel_circles_idxs + CIRCLE_BATCH_SIZE * (seg_y_idx*SCREEN_X_SEGMENTS+seg_x_idx);
    int cur_seg_rel_circles_num = num_rel_circles[seg_y_idx*SCREEN_X_SEGMENTS+seg_x_idx];
    // Load segment specific circle info into shared
    __shared__ float shared_radii[CIRCLE_BATCH_SIZE];
    __shared__ float3 shared_coord[CIRCLE_BATCH_SIZE];
    __shared__ float3 shared_color[CIRCLE_BATCH_SIZE];
    for(int i = threadIdx.x; i < cur_seg_rel_circles_num; i += blockDim.x){
        int circle_idx = cur_seg_rel_circles[i] + batch_idx * CIRCLE_BATCH_SIZE;
        int circle_idx3 = circle_idx*3;
        shared_radii[i] = cuConstRendererParams.radius[circle_idx];
        shared_coord[i] = *(float3*)(&cuConstRendererParams.position[circle_idx3]);
        shared_color[i] = *(float3*)&(cuConstRendererParams.color[circle_idx3]);
    }
    __syncthreads();
    // Blend colors for pixel
    int pixelX = seg_x_idx * seg_x_dim + (threadIdx.x%seg_x_dim);
    int pixelY = seg_y_idx * seg_y_dim + piece_idx*ROWS_PER_PIECE + (threadIdx.x/seg_x_dim);
    float4 pixel_color = *(float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
    for (int i = 0; i < cur_seg_rel_circles_num; i++)
    {
        float3 p = shared_coord[i];
        float rad = shared_radii[i];
        float3 circle_color = shared_color[i];
        float invWidth = 1.f / imageWidth;
        float invHeight = 1.f / imageHeight;
        float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                            invHeight * (static_cast<float>(pixelY) + 0.5f));
        shadePixel(pixelCenterNorm, p, rad, circle_color, &pixel_color);
    }

    cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)] = pixel_color.x;
    cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX) + 1] = pixel_color.y;
    cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX) + 2] = pixel_color.z;
    cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX) + 3] = pixel_color.w;
}

__device__ __inline__ int
circleInBox(
    float circleX, float circleY, float circleRadius,
    float boxL, float boxR, float boxT, float boxB)
{

    // clamp circle center to box (finds the closest point on the box)
    float closestX = (circleX > boxL) ? ((circleX < boxR) ? circleX : boxR) : boxL;
    float closestY = (circleY > boxB) ? ((circleY < boxT) ? circleY : boxT) : boxB;

    // is circle radius less than the distance to the closest point on
    // the box?
    float distX = closestX - circleX;
    float distY = closestY - circleY;

    if ( ((distX*distX) + (distY*distY)) <= (circleRadius*circleRadius) ) {
        return 1;
    } else {
        return 0;
    }
}

// For each circle in batch, determine if it impacts segment
__global__ void kernelCircleInSegment(int batch_idx, int batch_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size)
        return;
    int global_index = index + batch_idx * CIRCLE_BATCH_SIZE;
    int index3 = 3 * global_index;

    // Read circle properties
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    p.x *= imageWidth; p.y *= imageHeight;
    float  rad = cuConstRendererParams.radius[global_index]*imageWidth+1;

    // Compute if circle touches each segment
    short segmentWidth = imageWidth / SCREEN_X_SEGMENTS;
    short segmentHeight = imageHeight / SCREEN_Y_SEGMENTS;
    for(int i = 0; i < SCREEN_Y_SEGMENTS; i++){
        short segmentMinY = i*segmentHeight;
        short segmentMaxY = (i+1)*segmentHeight-1;
        for(int j = 0; j < SCREEN_X_SEGMENTS; j++){
            short segmentMinX = j*segmentWidth;
            short segmentMaxX = (j+1)*segmentWidth-1;
            cuConstRendererParams.circleSegmentMatrix[CIRCLE_BATCH_SIZE*(i*SCREEN_X_SEGMENTS+j)+index] = 
                circleInBox(p.x, p.y, rad, segmentMinX, segmentMaxX, segmentMaxY, segmentMinY);
        }
    }
    
}

// Given CircleSegmentMatrix and scan, compress into cudaRelCircles
__global__ void kernelIndexCompress(int* flags, int* scan, int* relevant_circles, int* relevant_circles_num, int batch_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // Determine segment
    int seg_idx = index / CIRCLE_BATCH_SIZE;
    int seg_start = seg_idx * CIRCLE_BATCH_SIZE;
    if (index >= seg_start + batch_size)
        return;
    // Determine value
    int cur_val = scan[index] - scan[seg_start];
    if(index == seg_start + batch_size-1)
    {
        // Contains last scan, which gives size
        relevant_circles_num[seg_idx] = flags[index] ? cur_val+1 : cur_val;
    }
    if(flags[index]){
        relevant_circles[seg_start + cur_val] = index - seg_start;
    }
    
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numberOfCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
    cudaScanResult = NULL;
    cudaRelCircles = NULL;
    cudaRelCirclesNum = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
        cudaFree(cudaCircleImpactsSegment);
        cudaFree(cudaScanResult);
        cudaFree(cudaRelCircles);
        cudaFree(cudaRelCirclesNum);
    }
}

const Image*
CudaRenderer::getImage() {

    // Need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numberOfCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("GeForce RTX 2080") == 0)
        {
            isFastGPU = true;
        }

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    if (!isFastGPU)
    {
        printf("WARNING: "
               "You're not running on a fast GPU, please consider using "
               "NVIDIA RTX 2080.\n");
        printf("---------------------------------------------------------\n");
    }
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numberOfCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numberOfCircles, cudaMemcpyHostToDevice);

    cudaError_t mallocErr1 = cudaMalloc(&cudaCircleImpactsSegment, sizeof(int) * SCREEN_X_SEGMENTS * SCREEN_Y_SEGMENTS * CIRCLE_BATCH_SIZE);
    printf("numCircles: %d\n", numberOfCircles);
    cudaError_t mallocErr2 = cudaMalloc(&cudaScanResult, sizeof(int) * SCREEN_X_SEGMENTS * SCREEN_Y_SEGMENTS * CIRCLE_BATCH_SIZE);
    cudaError_t mallocErr3 = cudaMalloc(&cudaRelCircles, sizeof(int) * SCREEN_X_SEGMENTS * SCREEN_Y_SEGMENTS * CIRCLE_BATCH_SIZE);
    cudaError_t mallocErr4 = cudaMalloc(&cudaRelCirclesNum, sizeof(int) * SCREEN_X_SEGMENTS * SCREEN_Y_SEGMENTS);
    if((mallocErr1 == cudaSuccess) && (mallocErr2 == cudaSuccess) && (mallocErr3 == cudaSuccess) && (mallocErr4 == cudaSuccess))
    {
        printf("***SUC: malloc has succeeded\n\n\n");
    }
    else
    {
        printf("***ERR: malloc has failed***\n\n\n");
    }

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numberOfCircles = numberOfCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;
    params.circleSegmentMatrix = cudaCircleImpactsSegment;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // Also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // Copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numberOfCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

// DBG
void print_device_data(int *device_data, int size, int num_print)
{
    // Prints data on device
    if(num_print >= size){
      // Print whole thing
      int* inarray = new int[size];
      cudaMemcpy(inarray, device_data, size*sizeof(int), cudaMemcpyDeviceToHost);
      printf("(dev): ");
      for(int i = 0; i < size; i++){
          printf("%d, ", inarray[i]);
      }
      printf("\n");
      delete inarray;
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
      delete inarray1;
      delete inarray2;
    }
}

void
CudaRenderer::render() {
    // Compute circles in batches
    int num_batches = (numberOfCircles+CIRCLE_BATCH_SIZE-1)/CIRCLE_BATCH_SIZE;
    for(int batch_idx = 0; batch_idx < num_batches; batch_idx++)
    {
        int batch_size = (CIRCLE_BATCH_SIZE*(batch_idx+1) > numberOfCircles) ? (numberOfCircles - CIRCLE_BATCH_SIZE*batch_idx) : CIRCLE_BATCH_SIZE;
        // Determine which circles impact which segments
        {
            dim3 blockDim(256, 1);
            dim3 gridDim((CIRCLE_BATCH_SIZE + blockDim.x - 1) / blockDim.x);
            kernelCircleInSegment<<<gridDim, blockDim>>>(batch_idx, batch_size);
        }
        cudaDeviceSynchronize();
        // Compress vector of flags to vector of idxs using scan
        {
            dim3 blockDim(256, 1);
            dim3 gridDim((SCREEN_X_SEGMENTS * SCREEN_Y_SEGMENTS * CIRCLE_BATCH_SIZE + blockDim.x - 1) / blockDim.x);
            // Do scan [0,1,1,0,1] -> [0,0,1,2,2]
            thrust::device_ptr<int> scan_start = thrust::device_pointer_cast(cudaCircleImpactsSegment);
            thrust::device_ptr<int> scan_result = thrust::device_pointer_cast(cudaScanResult);
            thrust::exclusive_scan(scan_start, scan_start + SCREEN_X_SEGMENTS * SCREEN_Y_SEGMENTS * CIRCLE_BATCH_SIZE, scan_result);
            // Compress [0,1,1,0,1] -> [1,2,4], 3
            kernelIndexCompress<<<gridDim, blockDim>>>(cudaCircleImpactsSegment, cudaScanResult, cudaRelCircles, cudaRelCirclesNum, batch_size);
        }
        cudaDeviceSynchronize();
        // Render
        {
            dim3 blockDim(115*ROWS_PER_PIECE);
            dim3 gridDim(SCREEN_Y_SEGMENTS, SCREEN_X_SEGMENTS, 115/ROWS_PER_PIECE);
            kernelRenderPixel<<<gridDim, blockDim>>>(cudaRelCircles, cudaRelCirclesNum, batch_idx, batch_size);
        }
        cudaDeviceSynchronize();
    }
}
