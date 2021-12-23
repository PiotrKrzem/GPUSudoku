#include "cpu_version.cuh"

using namespace std;

static int id = 0;

void safeCall(cudaError_t error) {
	id++;
	if (error != cudaSuccess) {
		cout << "!!!!!!!!! <" << id << "> " << error <<
			" " << cudaGetErrorString(error) << endl;
	}
	exit(1);
}

void safeCheck() {
	cudaError_t error = cudaGetLastError();
	id++;
	if (error != cudaSuccess) {
		cout << "!!!!!!!!! <" << id << "> " << error <<
			" " << cudaGetErrorString(error) << endl;
	}
	exit(-1);
}

int countZeros(int* input_grid, int grids) {
	int max_count = 0;
	for (int g = 0; g < grids; g++) {
		int count = 0;
		for (int i = 0; i < NN; i++) {
			if ((int)input_grid[i+g*NN] == 0) count++;
		}
		if (count > max_count) {
			max_count = count;
		}
	}
    
    return max_count;
}

int countZerosMin(int* input_grid, int grids) {
	int min_count = NN;
	for (int g = 0; g < grids; g++) {
		int count = 0;
		for (int i = 0; i < NN; i++) {
			if ((int)input_grid[i + g * NN] == 0) count++;
		}
		if (count < min_count) {
			min_count = count;
		}
	}

	return min_count;
}

void swapDevicePointers(int*& a, int*& b) {
    int* temp = a;
    a = b;
    b = temp;
}

void print(int* grid) {

	cout << "#     #     #     #\n";
	for (int i = 0; i < 9; i++) {
		if (!(i % 3) && i) {
			cout << "#     #     #     #\n";
			cout << "######$#####$######\n";
			cout << "#     #     #     #\n";
		}
		cout << "# " << grid[9 * i + 0] << grid[9 * i + 1] << grid[9 * i + 2] << " #";
		cout << " "  << grid[9 * i + 3] << grid[9 * i + 4] << grid[9 * i + 5] << " #";
		cout << " "  << grid[9 * i + 6] << grid[9 * i + 7] << grid[9 * i + 8] << " #\n";
	}
	cout << "#     #     #     #\n";
}

void printSmall(int* grid) {
	for (int i = 0; i < 9; i++) {
		if (!(i % 3) && i) {
			cout << "###$###$###\n";
		}
		cout << grid[9 * i + 0];
		cout << grid[9 * i + 1];
		cout << grid[9 * i + 2];
		cout << "#";
		cout << grid[9 * i + 3];
		cout << grid[9 * i + 4];
		cout << grid[9 * i + 5];
		cout << "#";
		cout << grid[9 * i + 6];
		cout << grid[9 * i + 7];
		cout << grid[9 * i + 8];
		cout << "\n";
	}
	cout << "\n";
}

void printNSmall(int* grids, int count) {
	for (int i = 0; i < count; i++) {
		printSmall(&grids[i * NN]);
	}
}

void printPossibilities(int* poss) {
	for (int i = 0; i < NN; i++) {
		if (i % 9 == 0 && i != 0) cout << endl;
		for (int j = 0; j < 9; j++) {
			cout << poss[i * 9 + j];
		}
		cout << " ";
	}
}

void printFirstNPointerValues(int* val, int howMuch) {

	int* stuff = new int[howMuch];
	cudaMemcpy(stuff, val, sizeof(int) * howMuch, cudaMemcpyDeviceToHost);
	for (int i = 0; i < howMuch; i++) {
		cout << stuff[i] << " ";
	}

	delete[](stuff);
}

void printFirstNPointerValues(int* val, int howMuch, void (*func)(int* v)) {

	int* stuff = new int[howMuch];
	cudaMemcpy(stuff, val, sizeof(int) * howMuch, cudaMemcpyDeviceToHost);
	func(stuff);
	delete[](stuff);
}
