#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/timeb.h>
#include <string.h>
#include <string> 
#include <deque>
#include <vector>
#include <algorithm>
#include "knapsack.cuh" 
#include <iostream>
#include <thrust/device_vector.h>
#include "kernels.cuh"
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <cassert>



#define TIMEB struct timeb
#define FTIME ftime
#define UDT_TIME long
#define MAX_SIZE_TO_PRINT 10

UDT_TIME gRefTime = 0;

UDT_TIME GetMilliSecondTime(TIMEB timeBuf);
void SetTime(void);
UDT_TIME GetTime(void);

int main(int argc, char* argv[])
{
	UDT_TIME time;
	int itemCnt;
	KnapsackInstance* inst; //a Knapsack instance object
	KnapsackDPSolver DPSolver;
	KnapsackBFSolver BFSolver; //brute-force solver
	KnapsackSolution *BFSoln;

	if (argc != 2) {
		printf("Invalid Number of command-line arguments\n");
		exit(1);
	}
	itemCnt = atoi(argv[1]);
	if (itemCnt < 1) {
		printf("Invalid number of items\n");
		exit(1);
	}

	// Creating Knapsack and solution object
	inst = new KnapsackInstance(itemCnt);
	BFSoln = new KnapsackSolution(inst);

	// Item Generation inside knapsack
	inst->Generate();

	// Print knapsack items
	inst->Print();


	SetTime();
	DPSolver.Solve(inst, BFSoln);
	time = GetTime();
	printf("\n\nSolved using dynamic programming (DP) in %ld ms. Optimal value = %d", time, BFSoln->getDPValue());
	if (itemCnt <= MAX_SIZE_TO_PRINT)
		BFSoln->Print("Dynamic Programming Solution");

	SetTime();
	BFSolver.Solve(inst, BFSoln);

	printf("\n\nSolved using brute-force enumeration (BF) in %ld ms. Optimal value = %d", time, BFSoln->getlbValue());
	if (itemCnt <= MAX_SIZE_TO_PRINT)
		BFSoln->Print("Brute-Force Solution");



	delete inst;
	delete BFSoln;

	printf("\n\nProgram Completed Successfully\n");

	return 0;
}
/********************************************************************/
UDT_TIME GetCurrentTime(void)
{
	UDT_TIME crntTime = 0;

	TIMEB timeBuf;
	FTIME(&timeBuf);
	crntTime = GetMilliSecondTime(timeBuf);

	return crntTime;
}
/********************************************************************/

void SetTime(void)
{
	gRefTime = GetCurrentTime();
}
/********************************************************************/

UDT_TIME GetTime(void)
{
	UDT_TIME crntTime = GetCurrentTime();

	return (crntTime - gRefTime);
}
/********************************************************************/

UDT_TIME GetMilliSecondTime(TIMEB timeBuf)
{
	UDT_TIME mliScndTime;

	mliScndTime = timeBuf.time;
	mliScndTime *= 1000;
	mliScndTime += timeBuf.millitm;
	return mliScndTime;
}

class KnapsackItem {

public:
	int weight;
	int value;
	float valueperweight;
	KnapsackItem(int wght, int val) {
		weight = wght;
		value = val;
		valueperweight = (float)val / (float)wght;
	}

};

void KnapsackDPSolver::Solve(KnapsackInstance* inst_, KnapsackSolution* soln_)
{
	inst = inst_;
	soln = soln_;
	int n = inst->GetItemCnt();
	int c = inst->GetCapacity();

	int** a;
	a = new int*[n + 1];
	for (int i = 0; i < n + 1; i++) {
		a[i] = new int[c + 1];
	}


	for (int j = 0; j <= c; j++)
		a[0][j] = 0;

	for (int i = 1; i <= n; i++)
	{
		for (int j = 0; j <= c; j++)
		{
			if (inst->GetItemWeight(i) > j)
				a[i][j] = a[i - 1][j];
			else
				a[i][j] = std::max(inst->GetItemValue(i) + a[i - 1][j - inst->GetItemWeight(i)], a[i - 1][j]);
		}

	}
	soln->setDPValue(a[n][c]);

	int j = c;
	for (int i = n; i >= 1; i--)
	{
		if (a[i][j] > a[i - 1][j])
		{
			printf("%d ", i);
			j -= inst->GetItemWeight(i);
		}
	}
	delete a;
}

void KnapsackSolution::setDPValue(int val)
{
	dpvalue = val;
}

int KnapsackSolution::getDPValue()
{
	return dpvalue;
}

KnapsackInstance::KnapsackInstance(int itemCnt_)
{
	itemCnt = itemCnt_;

	weights = new int[itemCnt + 1];
	values = new int[itemCnt + 1];
	cap = 0;
}
/********************************************************************/

KnapsackInstance::~KnapsackInstance()
{
	delete[] weights;
	delete[] values;
}
/********************************************************************/

void KnapsackInstance::Generate()
{
	int i, wghtSum = 0;
	std::vector<KnapsackItem> items;

	weights[0] = 0;
	values[0] = 0;

	for (i = 1; i <= itemCnt; i++)
	{
		weights[i] = rand() % 100 + 1;
		values[i] = weights[i] + 10;
		items.push_back(KnapsackItem(weights[i], values[i]));
		wghtSum += weights[i];
	}

	std::sort(items.begin(), items.end(), [](KnapsackItem const & a, KnapsackItem const & b) -> bool
	{ return a.valueperweight > b.valueperweight; });


	printf("Number of items = %d, Capacity = %d\n", itemCnt, cap);
	printf("Weights: ");
	for (i = 1; i <= itemCnt; i++)
	{
		printf("%d ", weights[i]);
	}
	printf("\nValues: ");
	for (i = 1; i <= itemCnt; i++)
	{
		printf("%d ", values[i]);
	}
	printf("\n");

	i = 1;
	for (auto itm = items.begin(); itm != items.end(); itm++, i++) {
		weights[i] = itm->weight;
		values[i] = itm->value;
		printf("%7.2f ", itm->valueperweight);

	}

	cap = wghtSum / 2;

}

void KnapsackInstance::GenerateManually()
{
	int i, wghtSum;

	/*weights[0] = 0;
	values[0] = 0;
	weights[1] = 5;
	values[1] = 55;
	weights[2] = 2;
	values[2] = 20;
	weights[3] = 7;
	values[3] = 63;
	weights[4] = 8;
	values[4] = 64;
	weights[5] = 13;
	values[5] = 91;
	weights[6] = 14;
	values[6] = 84;
	weights[7] = 25;
	values[7] = 125;
	weights[8] = 20;
	values[8] = 80;
	weights[9] = 2;
	values[9] = 6;
	weights[10] = 19;
	values[10] = 38;*/

	wghtSum = 0;
	for (i = 1; i <= itemCnt; i++)
	{
		/*weights[i] = rand() % 100 + 1;
		values[i] = weights[i] + 10;*/
		wghtSum += weights[i];
	}
	cap = wghtSum / 2;
	//cap = 10;
}
/********************************************************************/

int KnapsackInstance::GetItemCnt()
{
	return itemCnt;
}
/********************************************************************/

int KnapsackInstance::GetItemWeight(int itemNum)
{
	return weights[itemNum];
}
/********************************************************************/

int KnapsackInstance::GetItemValue(int itemNum)
{
	return values[itemNum];
}
/********************************************************************/

int KnapsackInstance::GetCapacity()
{
	return cap;
}
/********************************************************************/

const int* KnapsackInstance::weight_ptr()
{
	return weights;
}

const int* KnapsackInstance::value_ptr()
{
	return values;
}


void KnapsackInstance::Print()
{
	int i;
	printf("\nAfter Sorting\n");
	printf("Number of items = %d, Capacity = %d\n", itemCnt, cap);
	printf("Weights: ");
	for (i = 1; i <= itemCnt; i++)
	{
		printf("%d ", weights[i]);
	}
	printf("\nValues: ");
	for (i = 1; i <= itemCnt; i++)
	{
		printf("%d ", values[i]);
	}
	printf("\n");
}
/*****************************************************************************/

KnapsackSolution::KnapsackSolution(KnapsackInstance* inst_)
{
	int i, itemCnt = inst_->GetItemCnt();
	beforeSlackWeight = 0;
	beforeSlackValue = 0;
	lbValue = 0;
	slackItem = 1;
	inst = inst_;
	isTaken = new bool[itemCnt + 1];


	for (i = 1; i <= itemCnt; i++)
	{
		isTaken[i] = false;
	}
}
/********************************************************************/

KnapsackSolution::~KnapsackSolution()
{
	//delete [] isTaken;
}
/********************************************************************/

//bool KnapsackSolution::operator == (KnapsackSolution& otherSoln)
//{
//	return value == otherSoln.value;
//}
/********************************************************************/

void KnapsackSolution::Take(int itemNum)
{
	if (slackItem > itemNum)
	{
		isTaken[itemNum] = true;
	}
}


void KnapsackSolution::DontTake(int itemNum)
{
	isTaken[itemNum] = false;
	if (slackItem > itemNum)
	{
		beforeSlackWeight -= inst->GetItemWeight(itemNum);
		beforeSlackValue -= inst->GetItemValue(itemNum);
	}
	else
	{
		slackItem++;
	}
}

int KnapsackSolution::getbeforeSlackWeight()
{
	return beforeSlackWeight;
}
int KnapsackSolution::getbeforeSlackValue()
{
	return beforeSlackValue;
}
int KnapsackSolution::getslackItem()
{
	return slackItem;
}
int KnapsackSolution::getlbValue()
{
	return lbValue;
}



void KnapsackSolution::Copy(KnapsackSolution* otherSoln)
{
	int i, itemCnt = inst->GetItemCnt();

	for (i = 1; i <= itemCnt; i++)
	{
		isTaken[i] = otherSoln->isTaken[i];
	}
	//value = otherSoln->value;
	beforeSlackValue = otherSoln->getbeforeSlackValue();
	beforeSlackWeight = otherSoln->getbeforeSlackWeight();
	slackItem = otherSoln->getslackItem();
	lbValue = otherSoln->getlbValue();
}

/********************************************************************/

void KnapsackSolution::Print(std::string title)
{
	int i, itemCnt = inst->GetItemCnt();

	printf("\n%s: ", title.c_str());
	for (i = 1; i <= itemCnt; i++)
	{
		if (isTaken[i] == true)
			printf("%d ", i);
	}
	//printf("\nValue = %d\n",value);

}
/*****************************************************************************/

KnapsackBFSolver::KnapsackBFSolver()
{
	crntSoln = NULL;
}
/********************************************************************/

KnapsackBFSolver::~KnapsackBFSolver()
{
	if (crntSoln != NULL)
		delete crntSoln;
}
/********************************************************************/

void KnapsackBFSolver::Solve(KnapsackInstance* inst_, KnapsackSolution* soln_)
{
	//saumya
	inst = inst_;
	bestSoln = soln_;
	
	int level = 0;
	std::deque <KnapsackSolution> solq;

	crntSoln = new KnapsackSolution(inst);
	crntSoln->update_bounds();
	solq.push_back(*crntSoln);

	// Transferring memory to GPU
	transfer_knapsack(inst);
	initialize_memory_pool();
	
	while (++level <= inst->GetItemCnt() && (!solq.empty() || md_vectorSizes > 0))
	{
		if (solq.size() > GPU_SWITCH_THRESH && md_vectorSizes == 0)
		{
#ifdef DEBUG_MODE_GPU
			std::cout << "Branching on GPU, Level: " << k << std::endl;
#endif
			transfer_to_gpu(solq);
			*mh_bestLbValue = bestSoln->getlbValue();
		}

		if (md_vectorSizes > 0) {
#ifdef DEBUG_MODE_GPU
			std::cout << "-----------------------" << "Now on level: " << k << "--------------------" << std::endl;
#endif
			branch_gpu(level);
			//bound_gpu(k);
			//historyBasedApproach(k);
			//int bestIndex = find_best_lb_index_gpu(k);
			//label_nodes_gpu();
			//concatenate_lists_cpu_gpu(bestIndex);
		}
		else
		{
			branch(level, solq);
			bound(level, solq, *bestSoln);
			prune(level, solq, *bestSoln);
		}

		// Transfer from the GPU only if we have nodes on the GPU and qsize is 0 and we're under the threshold
		if (md_vectorSizes > 0 && md_vectorSizes < GPU_SWITCH_THRESH && solq.size() == 0)
		{
#ifdef DEBUG_MODE_GPU
			std::cout << "Transfering from GPU, Level: " << k << std::endl;
#endif
			//transfer_from_gpu(q, best, k);
		}
	}

	// This is if the initial solution is the best solution...
	if (level < bestSoln->getslackItem())
	{
		// Means the solution was not finished
		for (int i = level; i < bestSoln->getslackItem(); ++i)
		{
			bestSoln->Take(i);
		}

		// Take any other items that might fit after slack items
		int w = inst->GetCapacity() - bestSoln->getbeforeSlackWeight();
		for (int i = bestSoln->getslackItem(); i <inst->GetItemCnt(); ++i)
		{
			if (w >= inst->GetItemWeight(i))
			{
				bestSoln->Take(i);
				w -= inst->GetItemWeight(i);
			}
		}
	}

	//auto soln = best->solution();
	//assert(soln->value() == best->lb_value());

	//return soln;


	/*while (++level <= inst->GetItemCnt()) {
		branch(level, solq);
		bound(level, solq, *bestSoln);
		prune(level, solq, *bestSoln);
	}
	int i = 0;
	while (i < solq.size())
	{
		auto sol = std::move(solq.at(i));
		i++;
		delete sol.isTaken;
	}*/
}
/********************************************************************/

cudaError_t KnapsackBFSolver::transfer_to_gpu(std::deque<KnapsackSolution>& q)
{
	md_vectorSizes = q.size() * 2;

	// Create solutions vectors as one flat array with a stride of knapsack size between levels
	md_solutionVectorSize = q.size() * (inst->GetItemCnt() + 1) * 2;

	if (md_vectorSizes > md_nodeLimit)
	{
		std::cerr << "Not enough device memory... Aborting...";
		throw std::runtime_error("Out of memory, GPU");
	}

	// Transfer values into host vectors
	size_t qSize = q.size();

	// Push back solution vectors now
	for (int i = 0; i < qSize; ++i)
	{
		auto node = std::move(q.front());
		q.pop_front();

		mh_beforeSlackValues[i] = node.getbeforeSlackValue();
		mh_beforeSlackWeights[i] = node.getbeforeSlackWeight();
		mh_lbValues[i] = node.getlbValue();
		mh_ubValues[i] = node.getubValue();
		mh_slackItems[i] = node.getslackItem();

		for (int j = 1; j <= inst->GetItemCnt(); ++j) {
			mh_solutionVectors[i * (inst->GetItemCnt() + 1) + j] = node.isTaken[j];
		}
	}

	auto err = cudaMemcpy(&md_ubValues[0], &mh_ubValues[0], sizeof(int) * md_vectorSizes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		std::cerr << " Failed to Copy UB Values from Host to Device" << std::endl;
		goto Error;
	}
	err = cudaMemcpy(&md_slackItems[0], &mh_slackItems[0], sizeof(int) * md_vectorSizes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		std::cerr << " Failed to Copy Slack Items from Host to Device" << std::endl;
		goto Error;
	}
	err = cudaMemcpy(&md_beforeSlackValues[0], &mh_beforeSlackValues[0], sizeof(int) * md_vectorSizes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		std::cerr << " Failed to Copy Before Slack Values from Host to Device" << std::endl;
		goto Error;
	}
	err = cudaMemcpy(&md_beforeSlackWeights[0], &mh_beforeSlackWeights[0], sizeof(int) * md_vectorSizes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		std::cerr << " Failed to Copy Before Slack Weights from Host to Device" << std::endl;
		goto Error;
	}
	err = cudaMemcpy(&md_solutionVectors[0], &mh_solutionVectors[0], sizeof(bool) * md_solutionVectorSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		std::cerr << " Failed to Copy Solution Vector from Host to Device" << std::endl;
		goto Error;
	}
	err = cudaMemcpy(&md_lbValues[0], &mh_lbValues[0], sizeof(int) * md_vectorSizes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		std::cerr << " Failed to Copy LB Values from Host to Device" << std::endl;
		goto Error;
	}

Error:
	if (err != cudaSuccess)
	{
		// Now see what memory got allocated and dealloc that
		free_gpu_memory();
		throw std::runtime_error("Failed to copy host memory");
	}

	return err;
}

void KnapsackBFSolver::branch_gpu(int k)
{
	// Create grid for branch kernel, depends on size being halved so that it can split between the two different sections of memory
	int gridX = ((md_vectorSizes / 2 - 1) / BRANCH_BLOCK_SIZE) + 1;
	dim3 gridSize(gridX, 1, 1);
	dim3 blockSize(BRANCH_BLOCK_SIZE, 1, 1);

	// Launch branch kernel
	branch_kernel <<<gridSize, blockSize >> > (md_beforeSlackWeights, md_beforeSlackValues, md_slackItems, md_ubValues, md_solutionVectors, k, md_vectorSizes / 2);
}

void KnapsackBFSolver::free_gpu_memory()
{
	if (md_rawData != nullptr) {
		cudaFree(md_rawData);
	}
	if (mh_rawData != nullptr)
	{
		cudaFreeHost(mh_rawData);
	}

	// Set all pointers to nullptr
	md_rawData = nullptr;
	mh_rawData = nullptr;
	md_beforeSlackValues = nullptr;
	mh_beforeSlackValues = nullptr;
	md_beforeSlackWeights = nullptr;
	mh_beforeSlackWeights = nullptr;
	md_slackItems = nullptr;
	mh_slackItems = nullptr;
	md_lbValues = nullptr;
	mh_lbValues = nullptr;
	md_ubValues = nullptr;
	mh_ubValues = nullptr;
	md_labelTable = nullptr;
	mh_labelTable = nullptr;
	md_solutionVectors = nullptr;
	mh_solutionVectors = nullptr;

	md_vectorSizes = 0;
	md_solutionVectorSize = 0;
}
void KnapsackSolution::update_bounds()
{
	// If the node has been labeled as non-promising, dont update bounds
	if (beforeSlackValue < 0)
	{
		return;
	}

	while (slackItem <= inst->GetItemCnt())
	{
		int itemWght = inst->GetItemWeight(slackItem);
		int itemVal = inst->GetItemValue(slackItem);

		if (beforeSlackWeight + itemWght <= inst->GetCapacity())
		{
			// We can fit this item in completely
			beforeSlackWeight += itemWght;
			beforeSlackValue += itemVal;

			// See if we can try to fit any more items from the knapsack
			slackItem++;
		}
		else
		{
			break;
		}
	}

	int lbItemIndex = slackItem + 1;
	lbValue = beforeSlackValue;
	int lbWeight = beforeSlackWeight;

	while (lbWeight < inst->GetCapacity() && lbItemIndex <= inst->GetItemCnt())
	{
		// See if we can fit the item into the lower bound solution
		if (inst->GetItemWeight(lbItemIndex) <= (inst->GetCapacity() - lbWeight))
		{
			lbWeight += inst->GetItemWeight(lbItemIndex);
			lbValue += inst->GetItemValue(lbItemIndex);
		}

		lbItemIndex++;
	}
}

int KnapsackSolution::getubValue() const
{
	// This means that we have taken upto and not including slack item, so we
	// Dont need to search down this subtree anymore, prune this node
	if (beforeSlackValue < 0)
	{
		return INVALID_VALUE;
	}

	if (slackItem <= inst->GetItemCnt()) {
		int residualCapacity = inst->GetCapacity() - beforeSlackWeight;
		float p = static_cast<float>(residualCapacity)*static_cast<float>(inst->GetItemValue(slackItem)) / static_cast<float>(inst->GetItemWeight(slackItem));
		return beforeSlackValue + p;
	}

	return static_cast<int>(beforeSlackValue);
}


void KnapsackBFSolver::branch(int k, std::deque<KnapsackSolution>& q)
{
	//saving queue size since queue will be updated during computation
	int currentqsize = q.size(), i = 0;

	// taking and untaking items for each node in queue
	while (i < currentqsize)
	{
		KnapsackSolution solutionTakeitem = std::move(q.front());
		q.pop_front();

		KnapsackSolution solutionUntakeitem = KnapsackSolution(inst);
		solutionUntakeitem.Copy(&solutionTakeitem);

		solutionTakeitem.Take(k);
		solutionUntakeitem.DontTake(k);

		q.push_back(std::move(solutionTakeitem));
		q.push_back(std::move(solutionUntakeitem));
		++i;
	}

}

void KnapsackBFSolver::bound(int k, std::deque<KnapsackSolution>& q, KnapsackSolution& best)
{
	int qSize = q.size();

	for (int i = 0; i < qSize; ++i)
	{
		auto soln = std::move(q.front());
		q.pop_front();

		// Update bounds
		soln.update_bounds();

		// See if it is better than the best, if it is, update best
		if (soln.getlbValue() > best.getlbValue())
		{
			best.Copy(&soln);
		}
		q.push_back(std::move(soln));
	}

	if (k < best.getslackItem())
	{
		// Means the solution was not finished
		for (int i = k + 1; i < best.getslackItem(); ++i)
		{
			best.Take(i);
		}

		// Take any other items that might fit after slack items
		int w = inst->GetCapacity() - best.getbeforeSlackWeight();
		for (int i = best.getslackItem() + 1; i < inst->GetItemCnt(); ++i)
		{
			if (w >= inst->GetItemWeight(i))
			{
				best.Take(i);
				w -= inst->GetItemWeight(i);
			}
		}
	}
}

void KnapsackBFSolver::prune(int k, std::deque<KnapsackSolution>& q, KnapsackSolution& best)
{
	int qSize = q.size();
	std::deque<KnapsackSolution> historyTable;


	for (int i = 0; i < qSize; ++i)
	{
		auto soln = std::move(q.front());
		q.pop_front();
		int soln_ub_value = soln.getubValue();
		int soln_before_slack_value = soln.getbeforeSlackValue();
		int soln_before_slack_weight = soln.getbeforeSlackWeight();

		if (soln_ub_value > best.getlbValue())/*If it is a valid solution*/
		{
			if (historyTable.size() == 0)
			{
				historyTable.push_back(std::move(soln));

			}
			else
			{
				/*comparing with the last element and placing at the last*/
				auto last = historyTable.at(historyTable.size() - 1);
				if (last.getbeforeSlackWeight() < soln_before_slack_weight)
				{
					if (last.getbeforeSlackValue() >= soln_before_slack_value)
					{
						continue;
					}
					else
					{
						historyTable.push_back(std::move(soln));
					}
				}
				else
				{ /*checking if we can place at the beginning of the table*/
					auto first = historyTable.at(0);
					if (first.getbeforeSlackWeight() > soln_before_slack_weight)
					{
						if (soln_before_slack_value < first.getbeforeSlackValue())
						{
							historyTable.push_front(std::move(soln));
						}
						else
						{
							int index = 0;
							while (index < historyTable.size() && historyTable.at(index).getbeforeSlackValue() <= soln_before_slack_value)
							{
								index++;
							}
							historyTable.erase(historyTable.begin(), historyTable.begin() + index - 1);

							historyTable.push_front(std::move(soln));
						}
					}
					else {
						int n = node_insert(historyTable, soln_before_slack_weight);
						if (historyTable.at(n).getbeforeSlackWeight() == soln_before_slack_weight)
						{
							auto equal = std::move(historyTable.at(n));
							auto equal_bsv = equal.getbeforeSlackValue();

							if (soln_before_slack_value <= equal_bsv)
							{
								std::swap(equal, historyTable.at(n));
								continue;
							}

							else
							{
								/*std::unique_ptr<bfs_solution> m;
								std::swap(m, historyTable.at(mid));*/
								equal.Copy(&soln);
								std::swap(equal, historyTable.at(n));
								int index = n + 1;

								while (index < historyTable.size() && historyTable.at(index).getbeforeSlackValue() <= soln_before_slack_value)
								{
									index++;
								}
								//historyTable.erase(historyTable.begin() + n + 1, historyTable.begin() + index - 1);
								if ((index - n) == 1)
								{

								}
								else
								{
									historyTable.erase(historyTable.begin() + n + 1, historyTable.begin() + index - 1);
								}
							}

						}
						else
						{
							auto less = std::move(historyTable.at(n - 1));
							auto less_bsv = less.getbeforeSlackValue();
							if (soln_before_slack_value <= less_bsv)
							{
								std::swap(less, historyTable.at(n - 1));
								continue;
							}
							else
							{
								auto high = std::move(historyTable.at(n));
								auto high_bsv = high.getbeforeSlackValue();
								if (soln_before_slack_value >= high_bsv)
								{
									high.Copy(&soln);
									std::swap(high, historyTable.at(n));
									int index = n + 1;
									while (index < historyTable.size() && historyTable.at(index).getbeforeSlackValue() <= soln_before_slack_value)
									{
										index++;
									}
									//historyTable.erase(historyTable.begin() + n, historyTable.begin() + index - 1);
									if ((index - n) == 1)
									{

									}
									else
									{
										historyTable.erase(historyTable.begin() + n + 1, historyTable.begin() + index - 1);
									}

								}
								else
								{
									historyTable.insert(historyTable.begin() + n, std::move(soln));
								}
							}


						}


					}
				}
			}
		}
	}
	q = historyTable;

}


int KnapsackBFSolver::node_insert(std::deque<KnapsackSolution>& historyTable, int weight)
{
	int lowerBound = 0;
	int upperBound = historyTable.size() - 1;
	int curIn = 0;
	while (true) {
		curIn = (upperBound + lowerBound) / 2;
		if (historyTable.at(curIn).getbeforeSlackWeight() == weight) {
			return curIn;
		}
		else if (historyTable.at(curIn).getbeforeSlackWeight() < weight) {
			lowerBound = curIn + 1; // its in the upper
			if (lowerBound > upperBound)
				return curIn + 1;
		}
		else {
			upperBound = curIn - 1; // its in the lower
			if (lowerBound > upperBound)
				return curIn;
		}
	}
}

cudaError_t KnapsackBFSolver::initialize_memory_pool()
{
	// Determine how much free space we have and allocate as much as possible
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	auto size = static_cast<size_t>(static_cast<double>(free - 1)*.985 + 1) / sizeof(int);
	auto intSize = sizeof(int);

#ifdef DEBUG_MODE_GPU
	std::cout << "Free: " << free << ", Total: " << total << std::endl;
	std::cout << "Allocating " << size << " ints" << std::endl;
#endif

	auto err = cudaMalloc(&md_rawData, size * sizeof(int));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device memory pool for algorithm, aborting...";
		goto Error;
	}

	// Allocate same pool on the host
	err = cudaMallocHost(&mh_rawData, size * sizeof(int));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate host memory pool for algorithm, aborting...";
		goto Error;
	}

	// Calculate how much memory each part will take
	size_t ksSize = inst->GetItemCnt();
	size_t solVectorSize = ((ksSize + 1) * sizeof(bool) / sizeof(int) + 1) * sizeof(int);
	size_t perNodeValueSize = 6 * sizeof(int);
	size_t perNodeSize = solVectorSize + perNodeValueSize;

	// Split memory by percentage of per space size
	auto valuePercent = static_cast<double>(perNodeValueSize) / static_cast<double>(perNodeSize);
	auto solVectorPercent = static_cast<double>(solVectorSize) / static_cast<double>(perNodeSize);
	size_t bytesForValues = valuePercent * (size - 2) * sizeof(int);
	size_t bytesForSol = solVectorPercent * (size - 2) * sizeof(int);

	// How many nodes can we store in each?
	size_t nodesForValues = bytesForValues / (6 * sizeof(int));
	size_t nodesForSol = bytesForSol / solVectorSize;
	size_t intsPerValue = bytesForValues / 6 / sizeof(int);

	// Initialze device sizes
	md_vectorSizes = 0;
	md_solutionVectorSize = 0;
	md_nodeLimit = (nodesForValues < nodesForSol ? nodesForValues : nodesForSol);

	// Initailze pointers for device
	md_beforeSlackValues = static_cast<int*>(md_rawData);
	md_beforeSlackWeights = static_cast<int*>(md_rawData) + intsPerValue;
	md_slackItems = static_cast<int*>(md_rawData) + intsPerValue * 2;
	md_lbValues = static_cast<int*>(md_rawData) + intsPerValue * 3;
	md_ubValues = static_cast<int*>(md_rawData) + intsPerValue * 4;
	md_labelTable = static_cast<int*>(md_rawData) + intsPerValue * 5;
	md_solutionVectors = reinterpret_cast<bool*>(static_cast<int*>(md_rawData) + intsPerValue * 6);
	md_bestLbValue = static_cast<int*>(static_cast<int*>(md_rawData) + (size - 1));
	md_bestIndex = static_cast<int*>(static_cast<int*>(md_rawData) + (size - 2));

#ifdef DEBUG_MODE_GPU
	md_endBeforeSlackValues = static_cast<int*>(md_rawData) + intsPerValue;
	md_endBeforeSlackWeights = static_cast<int*>(md_rawData) + intsPerValue * 2;
	md_endSlackItems = static_cast<int*>(md_rawData) + intsPerValue * 3;
	md_endLbValues = static_cast<int*>(md_rawData) + intsPerValue * 4;
	md_endUbValues = static_cast<int*>(md_rawData) + intsPerValue * 5;
	md_endLabelTable = static_cast<int*>(md_rawData) + intsPerValue * 6;
	md_endSolutionVectors = reinterpret_cast<bool*>(static_cast<int*>(md_rawData) + intsPerValue * 6 + (bytesForSol / sizeof(int) - 2));

	assert(static_cast<void*>(md_endSolutionVectors) <= md_bestIndex);
#endif

	// Test values
	auto endSolVector = reinterpret_cast<int*>(md_solutionVectors) + bytesForSol;
	auto endRawData = static_cast<int*>(md_rawData) + size;


	// Initialize pointers for host
	mh_beforeSlackValues = static_cast<int*>(mh_rawData);
	mh_beforeSlackWeights = static_cast<int*>(mh_rawData) + intsPerValue;
	mh_slackItems = static_cast<int*>(mh_rawData) + intsPerValue * 2;
	mh_lbValues = static_cast<int*>(mh_rawData) + intsPerValue * 3;
	mh_ubValues = static_cast<int*>(mh_rawData) + intsPerValue * 4;
	mh_labelTable = static_cast<int*>(mh_rawData) + intsPerValue * 5;
	mh_solutionVectors = reinterpret_cast<bool*>(static_cast<int*>(mh_rawData) + intsPerValue * 6);
	mh_bestLbValue = static_cast<int*>(static_cast<int*>(mh_rawData) + (size - 1));
	mh_bestIndex = static_cast<int*>(static_cast<int*>(mh_rawData) + (size - 1));
#ifdef DEBUG_MODE_GPU
	std::cout << "Nodes for values: " << nodesForValues << " Nodes for Solution Vectors: " << nodesForSol << std::endl;
#endif
Error:
	if (err != cudaSuccess)
	{
		if (md_rawData != nullptr)
		{
			cudaFree(md_rawData);
		}

		if (mh_rawData != nullptr)
		{
			cudaFreeHost(mh_rawData);
		}

		throw std::runtime_error("Failed to allocate memory pool");
	}

	return err;
}



/*****************************************************************************/