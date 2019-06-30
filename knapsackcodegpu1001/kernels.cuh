#include <cuda.h>
#include <cuda_runtime.h>
#include "knapsack.cuh"

#define GPU_SWITCH_THRESH 192
#define BRANCH_BLOCK_SIZE 128


__constant__ int d_weights[MAX_KNAPSACK_SIZE + 1];
__constant__ int d_values[MAX_KNAPSACK_SIZE + 1];
__constant__ int d_cap[1];
__constant__ int d_knapsackSize[1];

cudaError_t transfer_knapsack(KnapsackInstance* inst);


__global__ void branch_kernel(int* beforeSlackWeights, int* beforeSlackValues, int* slackItems, int* ubValues, bool* solutionVectors, int k, int qSize);