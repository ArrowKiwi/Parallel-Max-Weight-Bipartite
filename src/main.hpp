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

/*
void createMatrices(int* matrix, int* transpose, int n);
void displayMatrix(int* matrix, int rows, int cols, bool result, std::string prefix);
void displayMatrix(std::vector<int> matrix, int rows, int cols, bool result, std::string prefix);
void SequentialMatching(int* matrix, int* transpose, int vertex_count, int rank, int n, int p);
int MaxIndex(int* matrix, int start, int end, std::vector<int> Edges, int shift);
void FindBestMatch(std::unordered_set<int>& D, std::vector<int>& C, std::vector<int>& M, int* matrix, int* transpose, int n, int vertex_count, int rank);
void ProcessNeighborsOfMatched(std::unordered_set<int>& D, std::vector<int>& C, std::vector<int>& M, int* matrix, int* transpose, int vertex_count, int rank, int n, int p);
*/

void createMatrices(std::vector<int>& matrix, std::vector<int>& transpose, int n);
void displayMatrix(std::vector<int>& matrix, int rows, int cols, bool result, std::string prefix);
void displayMatrix(std::vector<int>& matrix, int rows, int cols, bool result, std::string prefix);
void SequentialMatching(std::vector<int>& matrix, std::vector<int>& transpose, int vertex_count, int rank, int n, int p);
int MaxIndex(std::vector<int>& matrix, int start, int end, std::vector<int>& Edges, int shift);
void FindBestMatch(std::unordered_set<int>& D, std::vector<int>& C, std::vector<int>& M, std::vector<int>& matrix, std::vector<int>& transpose, int n, int vertex_count, int rank);
void ProcessNeighborsOfMatched(std::unordered_set<int>& D, std::vector<int>& C, std::vector<int>& M, std::vector<int>& matrix, std::vector<int>& transpose, int vertex_count, int rank, int n, int p);