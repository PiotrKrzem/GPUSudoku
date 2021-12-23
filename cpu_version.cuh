#include "gpu_version.cuh"

using namespace std;

void findSudokuPossibleValuesCPU(int* grids, int* possibilities, int* sum, int grids_count) {

    for (int grid_idx = 0; grid_idx < grids_count; grid_idx++) {
        for (int thread_idx = 0; thread_idx < NN; thread_idx++) {
            
            int x = thread_idx % N;
            int y = (thread_idx - x) / N;
            int grid_cut = NN * grid_idx;
            int cell_cut = grid_cut + N * (int)y + (int)x;
            int poss_cut = cell_cut * N;

            if (grids[cell_cut]) {
                sum[cell_cut] = 10;
                continue;
            }
            else {
                int sumPossibilities = 0;

                int y1 = y;
                int y2 = y % 3 == 0 ? y + 2 : y - 1;
                int y3 = y % 3 == 2 ? y - 2 : y + 1;
                int yr1 = y < 3 ? 3 : 0;
                int yr2 = y > 5 ? 3 : 6;

                int x1 = x;
                int x2 = x % 3 == 0 ? x + 2 : x - 1;
                int x3 = x % 3 == 2 ? x - 2 : x + 1;
                int xr1 = x < 3 ? 3 : 0;
                int xr2 = x > 5 ? 3 : 6;


                for (int i = 1; i < N + 1; i++) {

                    int possibility =
                        // cube vals
                        grids[grid_cut + N * y1 + x2] != (int)i &&
                        grids[grid_cut + N * y1 + x3] != (int)i &&
                        grids[grid_cut + N * y2 + x1] != (int)i &&
                        grids[grid_cut + N * y2 + x2] != (int)i &&
                        grids[grid_cut + N * y2 + x3] != (int)i &&
                        grids[grid_cut + N * y3 + x1] != (int)i &&
                        grids[grid_cut + N * y3 + x2] != (int)i &&
                        grids[grid_cut + N * y3 + x3] != (int)i &&
                        // column vals
                        grids[grid_cut + N * (yr1 + 0) + x] != (int)i &&
                        grids[grid_cut + N * (yr1 + 1) + x] != (int)i &&
                        grids[grid_cut + N * (yr1 + 2) + x] != (int)i &&
                        grids[grid_cut + N * (yr2 + 0) + x] != (int)i &&
                        grids[grid_cut + N * (yr2 + 1) + x] != (int)i &&
                        grids[grid_cut + N * (yr2 + 2) + x] != (int)i &&
                        // row vals
                        grids[grid_cut + N * y + xr1 + 0] != (int)i &&
                        grids[grid_cut + N * y + xr1 + 1] != (int)i &&
                        grids[grid_cut + N * y + xr1 + 2] != (int)i &&
                        grids[grid_cut + N * y + xr2 + 0] != (int)i &&
                        grids[grid_cut + N * y + xr2 + 1] != (int)i &&
                        grids[grid_cut + N * y + xr2 + 2] != (int)i;

                    possibilities[poss_cut + i - 1] = (int)possibility;
                    sumPossibilities += (int)possibility;
                }
                sum[cell_cut] = sumPossibilities;
            }
        }
    }
    
}

// ok
void findLowestAssumptionsCountCPU(int* sum, int* min, int grids_count) {
    for (int blockIdx = 0; blockIdx < grids_count; blockIdx++) {
        int sum_cut = blockIdx * NN;
        int min_cut = blockIdx;

        int m = (int)sum[sum_cut];
        for (int i = 1; i < NN; i++) {
            m = ((int)sum[sum_cut + i] < m) ? (int)sum[sum_cut + i] : m;
        }
        min[min_cut] = m;
    }
}

// ok
void findFirstCellWithLowestAssumptionsCPU(int* sum, int* min, int* best_cell, int grids_count) {
    for (int blockIdx = 0; blockIdx < grids_count; blockIdx++) {
        int grid_idx = blockIdx;
        int cell_cut = grid_idx * NN;

        int target_value = min[grid_idx];
        int x_l = 0;
        for (int i = NN - 1; i >= 0; i--) {
            x_l = target_value == (int)sum[cell_cut + i] ? (int)i : x_l;
        }
        best_cell[grid_idx] = x_l;
    } 
}


void sumReduceCPU(int input_grids_count, int* dMin) {
    int sum = 0;
    for (int i = 0; i < input_grids_count; i++) {
        sum += dMin[i];
    }
    if (input_grids_count > 0) dMin[0] = sum;   
}

// ok
void findIndicesForNewBoardsSeqentialCPU(int* min, int* idx, int max) {
    int sum = 0;
    idx[0] = 0;
    for (int i = 1; i < max; i++) {
        sum += min[i - 1];
        idx[i] = sum;
    }
}

//ok
 void generateNewBoardsCPU(int* grids, int* possibilities, int* best_cell, int* idx, int* output_grids, int grids_count) {
     for (int blockIdx = 0; blockIdx < grids_count; blockIdx++) {
         int grid_idx = blockIdx;
         int grid_cut = NN * grid_idx;
         int cell_cut = best_cell[grid_idx];
         int poss_cut = NN * N * grid_idx + cell_cut * N;


         int offset = 0;
         for (int i = 0; i < N; i++) {
             if (possibilities[poss_cut + i]) {
                 for (int j = 0; j < NN; j++) {
                     output_grids[NN * (idx[grid_idx] + offset) + j] = grids[grid_cut + j];
                 }
                 output_grids[NN * (idx[grid_idx] + offset) + cell_cut] = i + 1;
                 offset++;
             }
         }
     }
}

 void allocateMemoryCPU(int input_grids_count, int** dPossibilities, int** dSum, int** dMin, int** dIndices, int** dBest_cell) {
    
    *dPossibilities = new int[input_grids_count * NN * N];
    *dSum = new int[input_grids_count * NN];
    *dMin = new int[input_grids_count];
    *dIndices = new int[input_grids_count];
    *dBest_cell = new int[input_grids_count];
    memset(*dPossibilities, 0, sizeof(int) * input_grids_count * NN * N);
}


 void freeMemoryCPU(int* dInput_grids, int* dPossibilities, int* dSum, int* dMin, int* dBest_cell, int* dIndices) {
    free(dInput_grids);
    free(dPossibilities);
    free(dSum);
    free(dMin);
    free(dBest_cell);
    free(dIndices);
}