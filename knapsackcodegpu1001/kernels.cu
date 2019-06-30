#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "kernels.cuh"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


cudaError_t transfer_knapsack(KnapsackInstance* m_inst)
{
	auto err = cudaMemcpyToSymbol(d_weights, m_inst->weight_ptr(), sizeof(int)*(m_inst->GetItemCnt() + 1));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to copy knapsack weights" << std::endl;
		return err;
	}

	err = cudaMemcpyToSymbol(d_values, m_inst->value_ptr(), sizeof(int)*(m_inst->GetItemCnt() + 1));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to copy knapsack values" << std::endl;
		return err;
	}

	int cap = m_inst->GetCapacity();
	int knapsackSize = m_inst->GetItemCnt();
	err = cudaMemcpyToSymbol(d_cap, &cap, sizeof(int));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to copy knapsack capacity" << std::endl;
		return err;
	}
	err = cudaMemcpyToSymbol(d_knapsackSize, &knapsackSize, sizeof(int));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to copy knapsack Size" << std::endl;
	}

	return err;
}
__global__ void branch_kernel(int* d_beforeSlackWeights, int* d_beforeSlackValues, int* d_slackItems, int* d_ubValues, bool* d_solutionVectors, int k, int qSize)
{
	int id = blockIdx.x *blockDim.x + threadIdx.x;// ************XXXX
	if (id >= qSize)
	{
		return;
	}

	int slackItem = d_slackItems[id];
	int beforeSlackWeight = d_beforeSlackWeights[id];
	int beforeSlackValue = d_beforeSlackValues[id];
	int ubValue = d_ubValues[id];

	// Copy solution vector into child node (dont take node)
	for (int i = 1; i <= d_knapsackSize[0]; ++i)
	{
		int fromSolIndex = id * (d_knapsackSize[0] + 1) + i;
		int toSolIndex = (id + qSize) * (d_knapsackSize[0] + 1) + i;
		d_solutionVectors[toSolIndex] = d_solutionVectors[fromSolIndex];
	}

	d_solutionVectors[(id + qSize) * (d_knapsackSize[0] + 1) + k] = false;
	__syncthreads();

	if (k < slackItem)
	{
		d_solutionVectors[id * (d_knapsackSize[0] + 1) + k] = true;
		beforeSlackWeight -= d_weights[k];
		beforeSlackValue -= d_values[k];
	}
	else
	{
		slackItem = slackItem + 1;
		ubValue = 0;
	}

	atomicExch(&d_beforeSlackWeights[id + qSize], beforeSlackWeight);
	atomicExch(&d_beforeSlackValues[id + qSize], beforeSlackValue);
	atomicExch(&d_slackItems[id + qSize], slackItem);
	atomicExch(&d_ubValues[id], ubValue);
}
