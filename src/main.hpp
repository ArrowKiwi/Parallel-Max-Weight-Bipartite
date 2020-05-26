#include <mpi.h>
#include <iomanip>
#include <limits>
#include <chrono>
#include <random>
#include <iostream>
#include <set>
#include <algorithm> 
#include <unistd.h>
#include <unordered_set>

//functions used for testing
void createMatrix(std::vector<int>& matrix, int n, int m);
void displayMatrix(std::vector<int>& matrix, int rows, int cols, bool result, std::string prefix);

//used for preprocessing the data
void generateTranspose(std::vector<int>& matrix, std::vector<int>& transpose, int n, int m);

//the actual implementation of the algorithm
void SequentialMatching(std::vector<int>& matrix, std::vector<int>& matrix_excess_node, std::vector<int>& transpose, std::vector<int>& transpose_excess_node, std::vector<int>& output, int n, int p, int rank);
void FindBestMatch(std::unordered_set<int>& D, std::vector<int>& C, std::vector<int>& M, std::vector<int>& matrix, std::vector<int>& matrix_excess_node, std::vector<int>& transpose, std::vector<int>& transpose_excess_node, std::vector<int>& sortedMatrixNeighbors, std::vector<int>& sortedTransposeNeighbors, std::vector<int>& sortedMatrixExcessNeighbors, std::vector<int>& sortedTransposeExcessNeighbors, std::vector<int>& indexOfSortedNeighborIndexes, int n, int p, int rank);
void ProcessNeighborsOfMatched(std::unordered_set<int>& D, std::vector<int>& C, std::vector<int>& M, std::vector<int>& matrix, std::vector<int>& matrix_excess_node, std::vector<int>& transpose, std::vector<int>& transpose_excess_node, std::vector<int>& sortedMatrixNeighbors, std::vector<int>& sortedTransposeNeighbors, std::vector<int>& sortedMatrixExcessNeighbors, std::vector<int>& sortedTransposeExcessNeighbors, std::vector<int>& indexOfSortedNeighborIndexes, int n, int p, int rank);
int MaxIndex(std::vector<int>& matrix, int start, int end, std::vector<int>& M, int shift);
int MaxIndex(std::vector<int>& neighbors, std::vector<int>& indexOfSortedNeighborIndexes, std::vector<int>& M, int local_vertex_num, int n, int p, int rank);
void SortNeighbors(std::vector<int>& matrix, std::vector<int>& neighbors, int start, int end);