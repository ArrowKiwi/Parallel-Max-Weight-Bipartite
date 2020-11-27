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

/*
//the actual implementation of the algorithm
void SequentialMatching(std::vector<int>& matrix, std::vector<int>& matrix_excess_node, std::vector<int>& transpose, std::vector<int>& transpose_excess_node, std::vector<int>& output, int n, int m, int p, int rank);
void FindBestMatch(std::unordered_set<int>& D, std::vector<int>& C, std::vector<int>& M, std::vector<int>& matrix, std::vector<int>& matrix_excess_node, std::vector<int>& transpose, std::vector<int>& transpose_excess_node, int n, int m, int p, int rank);
void ProcessNeighborsOfMatched(std::unordered_set<int>& D, std::vector<int>& C, std::vector<int>& M, std::vector<int>& matrix, std::vector<int>& matrix_excess_node, std::vector<int>& transpose, std::vector<int>& transpose_excess_node, int n, int m, int p, int rank);
int MaxIndex(std::vector<int>& matrix, int start, int end, std::vector<int>& M, int shift);
void SetUpdates(std::unordered_set<int>& D, std::vector<int>& C, std::vector<int>& M, int index, std::vector<int>& matrix, int start, int end, int shift, int modulus, int additive);
*/

class SequentialMatchingClass
{
    private:
        int n, m, p, rank;
        std::unordered_multiset<int> D;
        std::vector<int> C, M;

        std::vector<int> matrix, matrix_excess_node, transpose, transpose_excess_node;
        

        int vertex_count_n, vertex_count_m;
        bool excess_vertex_n, excess_vertex_m;
        //std::vector<int> finishedProcesses_n, finishedProcesses_m;

        void FindBestMatch();
        void ProcessNeighborsOfMatched();
        int MaxIndex(std::vector<int>& matrix, int start, int end, int shift);
        void SetUpdates(int index, std::vector<int>& matrix, int start, int end, int shift, int modulus, int additive);
        bool PairEvaluation(int localIndex, int potentialIndexPair);
        
        bool IsLocalIndex(int index);
        bool IsMatrixIndex(int index, bool global=false);
        bool IsTransposeIndex(int index, bool global=false);
        bool IsExcessMatrixIndex(int index, bool global=false);
        bool IsExcessTransposeIndex(int index, bool global=false);

    public:
        SequentialMatchingClass(std::vector<int>& matrix, std::vector<int>& matrix_excess_node, std::vector<int>& transpose, std::vector<int>& transpose_excess_node, int n, int m, int p, int rank)
        {
            this->matrix = matrix;
            this->matrix_excess_node = matrix_excess_node;
            this->transpose = transpose;
            this->transpose_excess_node = transpose_excess_node;
            
            this->n = n;
            this->m = m;
            this->p = p;
            this->rank = rank;

            this->C.resize(n+m, -1);
            this->M.resize(n+m, -1);

            this->vertex_count_n = n/p;
            this->vertex_count_m = m/p;

            this->excess_vertex_n = vertex_count_n*p != n && rank < n - vertex_count_n*p;
            this->excess_vertex_m = vertex_count_m*p != m && rank < m - vertex_count_m*p;
        }

        void GenerateMatching(std::vector<int>& output);
};

/*
class SubMatrix
{
    private:
        int rows, cols;

    public:
        
}
*/