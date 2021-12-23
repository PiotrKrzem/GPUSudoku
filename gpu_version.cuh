
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "boards.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void findSudokuPossibleValues(int* grids, int* possibilities, int* sum) {
    int gridIdx = blockIdx.x;
    int x = threadIdx.x % N;
    int y = (threadIdx.x - x) / N;

    int grid_cut = NN * gridIdx;
    int cell_cut = grid_cut + N * (int)y + (int)x;
    int poss_cut = cell_cut * N;

    if (grids[cell_cut]) {
        sum[cell_cut] = 10;
        return;
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

// ok
__global__ void findLowestAssumptionsCount(int* sum, int* min) {
    int sum_cut = blockIdx.x * NN;
    int min_cut = blockIdx.x;

    int m = (int)sum[sum_cut];
    for (int i = 1; i < NN; i++) {
        m = ((int)sum[sum_cut + i] < m) ? (int)sum[sum_cut + i] : m;
    }
    min[min_cut] = m;
}

// ok
__global__ void findFirstCellWithLowestAssumptions(int* sum, int* min, int* best_cell) {
    int gridIdx = blockIdx.x;
    int cell_cut = gridIdx * NN;

    int target_value = min[gridIdx];
    int x_l = 0;
    for (int i = NN - 1; i >= 0; i--) {
        x_l = target_value == (int)sum[cell_cut + i] ? (int)i : x_l;
    }
    best_cell[gridIdx] = x_l;
}

// ok
__global__ void sumReduceEven(int* min) {
    int idx = blockIdx.x;
    int offset = gridDim.x;
    min[idx] += min[idx + offset];
}

// ok
__global__ void sumReduceOdd(int* min) {
    int idx = blockIdx.x + 1;
    int offset = gridDim.x;
    min[idx] += min[idx + offset];
}

__host__ void sumReduce(int input_grids_count, int* dMin) {
    int reduce_grids_kernels;
    int reduce_grids_count = input_grids_count;
    while (reduce_grids_count > 1) {

        reduce_grids_kernels = reduce_grids_count / 2;
        if (reduce_grids_count % 2 == 0)
            sumReduceEven << <reduce_grids_kernels, 1 >> > (dMin);
        else
            sumReduceOdd << <reduce_grids_kernels, 1 >> > (dMin);
        reduce_grids_count -= reduce_grids_kernels;
    }
}

// ok
__global__ void findIndicesForNewBoardsSeqential(int* min, int* idx, int max) {
    int sum = 0;
    idx[0] = 0;
    for (int i = 1; i < max; i++) {
        sum += min[i - 1];
        idx[i] = sum;
    }
}

//ok
__global__ void generateNewBoards(int* grids, int* possibilities, int* best_cell, int* idx, int* output_grids) {
    int gridIdx = blockIdx.x;
    int grid_cut = NN * gridIdx;
    int cell_cut = best_cell[gridIdx];
    int poss_cut = NN * N * gridIdx + cell_cut * N;


    int offset = 0;
    for (int i = 0; i < N; i++) {
        if (possibilities[poss_cut + i]) {
            for (int j = 0; j < NN; j++) {
                output_grids[NN * (idx[gridIdx] + offset) + j] = grids[grid_cut + j];
            }
            output_grids[NN * (idx[gridIdx] + offset) + cell_cut] = i + 1;
            offset++;
        }
    }
}

__host__ void allocateMemory(int input_grids_count, int** dPossibilities, int** dSum, int** dMin, int** dIndices, int** dBest_cell) {
    cudaMalloc((void**)dPossibilities, sizeof(int) * input_grids_count * NN * N);
    cudaMalloc((void**)dSum, sizeof(int) * input_grids_count * NN);
    cudaMalloc((void**)dMin, sizeof(int) * input_grids_count);
    cudaMalloc((void**)dIndices, sizeof(int) * input_grids_count);
    cudaMalloc((void**)dBest_cell, sizeof(int) * input_grids_count);
    cudaMemset(*dPossibilities, 0, sizeof(int) * input_grids_count * NN * N);
}


__host__ void freeMemory(int* dInput_grids, int* dPossibilities, int* dSum, int* dMin, int* dBest_cell, int* dIndices) {
    cudaFree(dInput_grids);
    cudaFree(dPossibilities);
    cudaFree(dSum);
    cudaFree(dMin);
    cudaFree(dBest_cell);
    cudaFree(dIndices);
}