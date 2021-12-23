#include "universal_functions.cuh"
#include <chrono>

using namespace std;

void cpuImplementation(int* hInput_grids, int amount);
void gpuImplementation(int* hInput_grids, int amount);

int main(void) {
    Difficulty sudokuDifficulty = Difficulty::EXTREME;
    int amount = 1000;

    //Twice, to 'preheat' the kernels for the first time
    gpuImplementation(Board::get(sudokuDifficulty, amount), amount);
    cpuImplementation(Board::get(sudokuDifficulty, amount), amount);

    gpuImplementation(Board::get(sudokuDifficulty, amount), amount);
    cpuImplementation(Board::get(sudokuDifficulty, amount), amount);

    return 0;
}

void gpuImplementation(int* hInput_grids, int amount) {

    int* dInput_grids = nullptr, * dOutput_grids = nullptr;
    int* dPossibilities = nullptr, * dSum = nullptr, * dBest_cell = nullptr;
    int* dMin = nullptr, * dIndices = nullptr;

    int input_grids_count = amount;
    int output_grids_count = 1;

    cudaMalloc((void**)&dInput_grids, sizeof(int) * NN * input_grids_count);
    cudaMemcpy(dInput_grids, hInput_grids, sizeof(int) * NN * input_grids_count, cudaMemcpyHostToDevice);

    // display input
    printNSmall(hInput_grids,1);

    auto start = chrono::high_resolution_clock::now();

    // each loop iteration should replace one '0' (empty cell) with a 'perfect number'
    // therefore after count(0) oterations solution will be obtained
    for (int zeros_count = countZeros(hInput_grids, input_grids_count); zeros_count > 0; zeros_count--) {

        // allocate and clear memory for all required memory objects
        allocateMemory(input_grids_count, &dPossibilities, &dSum, &dMin, &dIndices, &dBest_cell);

        // for each cell in every board make a  9-sized possibility map, and count how many possibilites are available
        findSudokuPossibleValues << <input_grids_count, NN >> > (dInput_grids, dPossibilities, dSum);

        // for each board find the lowest possibilities count
        findLowestAssumptionsCount << <input_grids_count, 1 >> > (dSum, dMin);

        // for each board find first best cell index - first cell which has the lowest possibilities count
        findFirstCellWithLowestAssumptions << <input_grids_count, 1 >> > (dSum, dMin, dBest_cell);

        // make an indices array that will be used to locate new generated board's location in memory (sum scan)
        findIndicesForNewBoardsSeqential << <1, 1 >> > (dMin, dIndices, input_grids_count);

        // compute the number of new boards that will be generated
        sumReduce(input_grids_count, dMin);

        // allocate memory for new boards
        cudaMemcpy(&output_grids_count, dMin, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMalloc((void**)&dOutput_grids, sizeof(int) * NN * output_grids_count);

        // if there are no new boards, the puzzle is unsolvable
        if (output_grids_count == 0) break;

        // generate new boards from all inforamtion available
        generateNewBoards << <input_grids_count, 1 >> > (dInput_grids, dPossibilities, dBest_cell, dIndices, dOutput_grids);

        // free old memory
        freeMemory(dInput_grids, dPossibilities, dSum, dMin, dBest_cell, dIndices);

        // set output grids as new input grids
        input_grids_count = output_grids_count;
        swapDevicePointers(dInput_grids, dOutput_grids);
    }
    auto stop = chrono::high_resolution_clock::now();

    // display the final result
    cudaMemcpy(hInput_grids, dInput_grids, sizeof(int) * NN, cudaMemcpyDeviceToHost);
    printNSmall(hInput_grids,1);
    cout << (countZerosMin(hInput_grids, input_grids_count) > 0 ? "No solution" : "Solved!") << endl<<endl;

    cudaFree(dInput_grids);
    free(hInput_grids);

    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "GPU implementaiton took: " << duration.count() << " milliseconds" << endl << endl;
}

void cpuImplementation(int* hInput_grids, int amount) {

    int* dInput_grids = nullptr, * dOutput_grids = nullptr;
    int* dPossibilities = nullptr, * dSum = nullptr, * dBest_cell = nullptr;
    int* dMin = nullptr, * dIndices = nullptr;

    int input_grids_count = amount;
    int output_grids_count = 1;

    dInput_grids = new int[NN*input_grids_count];
    memcpy(dInput_grids, hInput_grids, sizeof(int) * NN*input_grids_count);

    printNSmall(hInput_grids,1);

    auto start = chrono::high_resolution_clock::now();
    for (int zeros_count = countZeros(hInput_grids, input_grids_count); zeros_count > 0; zeros_count--) {

        // allocate and clear memory for all required memory objects
        allocateMemoryCPU(input_grids_count, &dPossibilities, &dSum, &dMin, &dIndices, &dBest_cell);

        // for each cell in every board make a  9-sized possibility map, and count how many possibilites are available
        findSudokuPossibleValuesCPU(dInput_grids, dPossibilities, dSum, input_grids_count);
        
        // for each board find the lowest possibilities count
        findLowestAssumptionsCountCPU(dSum, dMin, input_grids_count);

        // for each board find first best cell index - first cell which has the lowest possibilities count
        findFirstCellWithLowestAssumptionsCPU(dSum, dMin, dBest_cell, input_grids_count);

        // make an indices array that will be used to locate new generated board's location in memory (sum scan)
        findIndicesForNewBoardsSeqentialCPU(dMin, dIndices, input_grids_count);

        // compute the number of new boards that will be generated
        sumReduceCPU(input_grids_count, dMin);

        // allocate memory for new boards
        output_grids_count = dMin[0];
        dOutput_grids = new int[NN * output_grids_count];

        // if there are no new boards, the puzzle is unsolvable
        if (output_grids_count == 0) break;

        // generate new boards from all inforamtion available
        generateNewBoardsCPU (dInput_grids, dPossibilities, dBest_cell, dIndices, dOutput_grids, input_grids_count);

        // free old memory
        freeMemoryCPU(dInput_grids, dPossibilities, dSum, dMin, dBest_cell, dIndices);

        // set output grids as new input grids
        input_grids_count = output_grids_count;
        swapDevicePointers(dInput_grids, dOutput_grids);
    }
    auto stop = chrono::high_resolution_clock::now();

    // display the final result
    memcpy(hInput_grids, dInput_grids, sizeof(int) * NN);
    printNSmall(hInput_grids,1);
    cout << (countZerosMin(hInput_grids, input_grids_count) > 0 ? "No solution" : "Solved!") << endl<<endl;

    free(dInput_grids);
    free(hInput_grids);

    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "CPU implementaiton took: " << duration.count() << " milliseconds" << endl << endl;;

}