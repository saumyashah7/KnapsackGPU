
//#define KNAPSACK_DEBUG

#define INVALID_VALUE -1
#include <deque>
#define MAX_KNAPSACK_SIZE 511

class KnapsackInstance
{
private:
	int itemCnt; //Number of items
	int cap; //The capacity
	int* weights;//An array of weights
	int* values;//An array of values

public:
	KnapsackInstance(int itemCnt_);
	~KnapsackInstance();

	void Generate();
	void GenerateManually();

	int GetItemCnt();
	int GetItemWeight(int itemNum);
	int GetItemValue(int itemNum);
	int GetCapacity();
	void Print();

	const int* weight_ptr();
	const int* value_ptr();
};

class KnapsackSolution
{
private:


	int beforeSlackWeight;
	int beforeSlackValue;
	int lbValue;
	int slackItem;
	int dpvalue;

public:
	KnapsackSolution(KnapsackInstance* inst);
	~KnapsackSolution();
	bool* isTaken;
	KnapsackInstance* inst;

	void setDPValue(int val);
	int getDPValue();

	void Take(int itemNum);
	void DontTake(int itemNum);


	int getbeforeSlackWeight();
	int getbeforeSlackValue();
	int getslackItem();
	int getlbValue();
	int getubValue() const;


	void Print(std::string str);
	void Copy(KnapsackSolution* otherSoln);
	void update_bounds();

};

// Dynamic programming solver
class KnapsackDPSolver
{
private:
	KnapsackInstance* inst;
	KnapsackSolution* soln;

public:
	KnapsackDPSolver();
	~KnapsackDPSolver();
	void Solve(KnapsackInstance* inst, KnapsackSolution* soln);
};


// Brute-force solver
class KnapsackBFSolver
{
private:
	// All of the data needed to run the BFS B&B alg on the gpu, device local
	int* md_beforeSlackWeights = nullptr;
	int* md_beforeSlackValues = nullptr;
	int* md_ubValues = nullptr;
	int* md_lbValues = nullptr;
	int* md_slackItems = nullptr;
	int* md_labelTable = nullptr;
	bool* md_solutionVectors = nullptr;
	int* md_bestLbValue = nullptr;
	int* md_bestIndex = nullptr;

	int* mh_beforeSlackWeights = nullptr;
	int* mh_beforeSlackValues = nullptr;
	int* mh_ubValues = nullptr;
	int* mh_lbValues = nullptr;
	int* mh_slackItems = nullptr;
	int* mh_labelTable = nullptr;
	bool* mh_solutionVectors = nullptr;
	int* mh_bestLbValue = nullptr;
	int* mh_bestIndex = nullptr;

	void* md_rawData = nullptr;
	void* mh_rawData = nullptr;

	// We need to store how big each of the arrays passed to the gpu is.
	size_t md_vectorSizes = 0;
	size_t md_solutionVectorSize = 0;
	size_t md_nodeLimit = 0;

#ifdef DEBUG_MODE_GPU
	int* md_endBeforeSlackWeights = nullptr;
	int* md_endBeforeSlackValues = nullptr;
	int* md_endUbValues = nullptr;
	int* md_endLbValues = nullptr;
	int* md_endSlackItems = nullptr;
	int* md_endLabelTable = nullptr;
	bool* md_endSolutionVectors = nullptr;
#endif



protected:
	KnapsackInstance* inst;
	KnapsackSolution* crntSoln;
	KnapsackSolution* bestSoln;


public:
	KnapsackBFSolver();
	~KnapsackBFSolver();
	virtual void Solve(KnapsackInstance* inst, KnapsackSolution* soln);
	virtual void branch(int k, std::deque<KnapsackSolution>& q);
	virtual void bound(int k, std::deque<KnapsackSolution>& q, KnapsackSolution& best);
	virtual void prune(int k, std::deque<KnapsackSolution>& q, KnapsackSolution& best);
	virtual int node_insert(std::deque<KnapsackSolution>& historyTable, int weight);


	cudaError_t transfer_to_gpu(std::deque<KnapsackSolution>& q);
	cudaError_t transfer_from_gpu(std::deque<KnapsackSolution>& q, KnapsackSolution& best, int k);
	cudaError_t initialize_memory_pool();
	void free_gpu_memory();
	void branch_gpu(int k);
};





