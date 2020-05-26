#include "./main.hpp"

#define PRINT_DEBUG_STATEMENTS 1
#define SHOW_MATRICES_AND_RESULTS 0
#define TESTING 0
#define SHOW_COMPLETIONS 0
#define SHOW_INDIVIDUAL_RESULTS 0
#define SHOW_INTERMEDIATE_RESULTS 0
#define INCLUDE_SORTING 0

int main(int argc, char** argv)
{
    //used for measuring performance, just declaring not using current values
    auto start = std::chrono::high_resolution_clock::now();

    //same and sent
    int p; //number of processes
    int n; //number of random numbers generated, assuming that it is divisible by p

    //same and computed
    int vertex_count; //also the vertex count per vertex disjoint set; ideally the same
    int output_count;

    //unique and sent
    int rank; //process id, 0 <= rank < p
    std::vector<int> matrix;
    std::vector<int> transpose;
    std::vector<int> matrix_excess_node;
    std::vector<int> transpose_excess_node;

    //unique and computed
    std::vector<int> output;
    std::vector<int> combinedResults; //only used for process with rank == 0
    std::vector<int> filteredResults; //only used for process with rank == 0

    if(argc ==  2)
    {
        //*************************START OF MPI SETUP*************************
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &p);
        //*************************END OF MPI SETUP*************************

        //*************************START OF DATA DISTRIBUTION*************************
        if(rank == 0)
        {
            n = atoi(argv[1]); 
            createMatrix(matrix, n, n);
            if(SHOW_MATRICES_AND_RESULTS) displayMatrix(matrix, n, n, true, "");
            if(SHOW_MATRICES_AND_RESULTS) std::cout << "--------" << std::endl;
        }
        

        if(rank == 0)
            start = std::chrono::high_resolution_clock::now();

        //send out the size of the matrix
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        vertex_count = n/p;
        output_count = 2*vertex_count;
        if(vertex_count*p != n)
            output_count += 2;
        output.resize(output_count, -1); //this holds the final edges, comes in pairs of two
        
        if(rank == 0)
        {                
            generateTranspose(matrix, transpose, n, n);
            if(SHOW_MATRICES_AND_RESULTS) displayMatrix(transpose, n, n, true, "");

            std::vector<MPI_Request> requests(p-1);
            
            //send out the matrices
            for(int i = 1; i < p; i++)
                MPI_Isend(&matrix[vertex_count*i*n], vertex_count*n, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[i-1]);

            //wait until everyone received their transpose segments
            MPI_Waitall(p-1, &requests[0], MPI_STATUS_IGNORE);

            //sending remaining vertices in the matrix that aren't indexed between 0 and p*floor(n/p), known as excess matrix vertices in comments
            if(vertex_count*p != n)
            {
                for(int i = 1; i < n - vertex_count*p; i++)
                    MPI_Isend(&matrix[(vertex_count*p + i)*n], n, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[i-1]);
                matrix_excess_node.resize(n);
                std::copy(matrix.begin() + vertex_count*p*n, matrix.begin() + vertex_count*p*n + n, matrix_excess_node.begin());
            }

            //wait until the excess matrix vertices have been distributed
            MPI_Waitall(std::max(n-vertex_count*p - 1, 0), &requests[0], MPI_STATUS_IGNORE);

            //send out the transposes
            for(int i = 1; i < p; i++)
                MPI_Isend(&transpose[vertex_count*i*n], vertex_count*n, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[i-1]);

            //wait until everyone received their transpose segments
            MPI_Waitall(p-1, &requests[0], MPI_STATUS_IGNORE);

            //sending remaining vertices in the transpose that aren't indexed between 0 and p*floor(n/p), known as excess transpose vertices in comments
            if(vertex_count*p != n)
            {
                for(int i = 1; i < n - vertex_count*p; i++)
                    MPI_Isend(&transpose[(vertex_count*p + i)*n], n, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[i-1]);
                transpose_excess_node.resize(n);
                std::copy(transpose.begin() + vertex_count*p*n, transpose.begin() + vertex_count*p*n + n, transpose_excess_node.begin());
            }

            //wait until the excess matrix vertices have been distributed
            MPI_Waitall(std::max(n-vertex_count*p - 1, 0), &requests[0], MPI_STATUS_IGNORE);
        }
        else
        {
            matrix.resize(vertex_count*n);
            transpose.resize(vertex_count*n);

            //receive portion of the matrix
            MPI_Recv(&matrix[0], vertex_count*n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            //receiving excess matrix vertices
            if(vertex_count*p != n && rank < n-vertex_count*p)
            {
                matrix_excess_node.resize(n);
                MPI_Recv(&matrix_excess_node[0], n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
                
            //receive portion of the transpose
            MPI_Recv(&transpose[0], vertex_count*n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if(vertex_count*p != n && rank < n-vertex_count*p)
            {
                transpose_excess_node.resize(n);
                MPI_Recv(&transpose_excess_node[0], n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        //*************************END OF DATA DISTRIBUTION*************************

        if(PRINT_DEBUG_STATEMENTS) std::cout << "Hello from Rank " << rank << std::endl;

        //the algorithm implementation
        SequentialMatching(matrix, matrix_excess_node, transpose, transpose_excess_node, output, n, p, rank);
        
        //*************************START OF OUTPUT RETRIEVAL*************************
        if(p != 1)
        {
            if(rank == 0)
                combinedResults.resize(p*output_count);
            MPI_Gather(&output[0], output_count, MPI_INT, &combinedResults[0], output_count, MPI_INT, 0, MPI_COMM_WORLD);
        }
        else
            combinedResults = output;
        //*************************END OF OUTPUT RETRIEVAL*************************

        MPI_Finalize();

        if(rank == 0)
        {
            filteredResults.resize(n, -1);

            //filtering the results
            for(int i = 0; i < 2*n; i = i + 2)
                if(combinedResults[i] != -1 && combinedResults[i+1] != -1)
                    filteredResults[combinedResults[i]] = combinedResults[i+1] - n;
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
            if(TESTING)
                std::cout << duration.count() << std::endl; 
            else
            {
                if(SHOW_MATRICES_AND_RESULTS) 
                {
                    //displayMatrix(filteredResults, 1, filteredResults.size(), true, "Final Result:");
                    for(int i = 0; i < filteredResults.size(); i++)
                        std::cout << i << " " << filteredResults[i] << " " << matrix[i*n + filteredResults[i]] << std::endl;
                }
                std::cout << "Total Execution Time: " << duration.count() << " microseconds." << std::endl; 
            }
        }
    }
    else
        if(rank == 0)
            std::cout << "You did not provide enough command line arguments, please enter a number of test integers and a number of nodes." << std::endl;

    return 0;
}

//note that is is assumed that the indexes of a row are shifted by vertex_count
void SequentialMatching(std::vector<int>& matrix, std::vector<int>& matrix_excess_node, std::vector<int>& transpose, std::vector<int>& transpose_excess_node, std::vector<int>& output, int n, int p, int rank)
{
    int vertex_count = n/p;
    std::vector<int> C(2*n, -1); //contains the current most ideal vertex pair
    std::vector<int> M(2*n, -1); //this holds the final edges, comes in pairs of two
    std::unordered_set<int> D; //all the matched vertices that have successfully been matched

    std::vector<int> indexOfSortedNeighborIndexes;
    if(vertex_count*p != n && rank < n - vertex_count*p)
        indexOfSortedNeighborIndexes.resize((vertex_count+1)*2, 0);
    else
        indexOfSortedNeighborIndexes.resize(vertex_count*2, 0);

    std::vector<int> unsortedIndexes(n); //used as a reference to copy from
    std::vector<int> sortedMatrixNeighbors(vertex_count*n);
    std::vector<int> sortedTransposeNeighbors(vertex_count*n);
    std::vector<int> sortedMatrixExcessNeighbors;
    std::vector<int> sortedTransposeExcessNeighbors;

    //determining if we need to do local computations or continue all together
    int localFinished = false;

    std::vector<int> finishedProcesses(p, false);
    int tag;

    //*************************THIS SHOULD BE THE START OF THE ALGORITHM*************************

    //beginning of sorting the edges/neighbors of the vertices local to each process in the decending order of weight--------------------------------------------
    #if INCLUDE_SORTING
    if(PRINT_DEBUG_STATEMENTS) std::cout << "Sorting Neighbors at Rank " << rank << std::endl;
    for(int i = 0; i < n; i++)
        unsortedIndexes[i] = i;

    if(vertex_count*p != n && rank < n - vertex_count*p)
    {
        sortedMatrixExcessNeighbors.resize(n);
        std::copy(unsortedIndexes.begin(), unsortedIndexes.end(), sortedMatrixExcessNeighbors.begin());
        SortNeighbors(matrix_excess_node, sortedMatrixExcessNeighbors, 0, n);
        displayMatrix(sortedMatrixExcessNeighbors, 1, n, false, "Rank " + std::to_string(rank) + " EXCESS MATRIX    = ");

        sortedTransposeExcessNeighbors.resize(n);
        std::copy(unsortedIndexes.begin(), unsortedIndexes.end(), sortedTransposeExcessNeighbors.begin());
        SortNeighbors(transpose_excess_node, sortedTransposeExcessNeighbors, 0, n);
        displayMatrix(sortedTransposeExcessNeighbors, 1, n, false, "Rank " + std::to_string(rank) + " EXCESS TRANSPOSE = ");
    }

    //this is where openmp could be included to parallelize this on each node
    for(int i = 0; i < vertex_count; i++)
    {
        std::copy(unsortedIndexes.begin(), unsortedIndexes.end(), sortedMatrixNeighbors.begin() + i*n);
        SortNeighbors(matrix, sortedMatrixNeighbors, i*n, (i+1)*n);
    }

    //this is where openmp could be included to parallelize this on each node
    for(int i = 0; i < vertex_count; i++)
    {
        std::copy(unsortedIndexes.begin(), unsortedIndexes.end(), sortedTransposeNeighbors.begin() + i*n);
        SortNeighbors(transpose, sortedTransposeNeighbors, i*n, (i+1)*n);
    }
    #endif
    //end of sorting the edges/neighbors of the vertices local to each process in the decending order of weight--------------------------------------------

    while(true)
    {
        //sending request sizes structures
        std::vector<int> vertexSendSizeRequests(p,0);
        std::vector<MPI_Request> vertexSendSizeRequestsMessages;
        vertexSendSizeRequestsMessages.reserve(p);

        //receiving request size structures (pause after using these)
        std::vector<int> vertexRecvSizeRequests(p,0);
        std::vector<MPI_Request> vertexRecvSizeRequestsMessages;
        vertexRecvSizeRequestsMessages.reserve(p);
        
        //sending request structures
        std::vector<std::vector<int>> vertexSendRequests(p);
        std::vector<MPI_Request> vertexSendRequestsMessages;
        vertexSendRequestsMessages.reserve(p);

        //receiving requests (pause after using these)
        std::vector<std::vector<int>> vertexRecvRequests(p);
        std::vector<MPI_Request> vertexRecvRequestsMessages;
        vertexRecvRequestsMessages.reserve(p);

        //sending responses (assumming that the size will be equal to some constant times the request size)
        std::vector<std::vector<int>> vertexSendResponses(p);
        std::vector<MPI_Request> vertexSendResponsesMessages;
        vertexSendResponsesMessages.reserve(p);

        //receiving responses (using previous assumption) (pause after using these)
        std::vector<std::vector<int>> vertexRecvResponses(p);
        std::vector<MPI_Request> vertexRecvResponsesMessages;
        vertexRecvResponsesMessages.reserve(p);

        //recent completed processes
        std::vector<int> recentFinishedProcesses(p, false);

        //beginning of determining if locally complete, adjusting to others that are locally complete and determining if the graph is globally complete--------------------------------------------
        int globalChangeOccured = true;
        if(PRINT_DEBUG_STATEMENTS) std::cout << "Determining Local And Global Changes at Rank " << rank << std::endl;
        while(globalChangeOccured)
        {
            int localChangeOccured;
            if(!localFinished)
            {
                std::cout << "Rank " << rank << " here 1" << std::endl;
                FindBestMatch(D, C, M, matrix, matrix_excess_node, transpose, transpose_excess_node, sortedMatrixNeighbors, sortedTransposeNeighbors, sortedMatrixExcessNeighbors, sortedTransposeExcessNeighbors, indexOfSortedNeighborIndexes, n, p, rank);
                std::cout << "Rank " << rank << " here 2" << std::endl;
                ProcessNeighborsOfMatched(D, C, M, matrix, matrix_excess_node, transpose, transpose_excess_node, sortedMatrixNeighbors, sortedTransposeNeighbors, sortedMatrixExcessNeighbors, sortedTransposeExcessNeighbors, indexOfSortedNeighborIndexes, n, p, rank);
                std::cout << "Rank " << rank << " here 3" << std::endl;

                localFinished = true;
                int index;

                //checking if all the excess vertices are matched
                if(vertex_count*p != n && rank < n - vertex_count*p)
                {
                    index = vertex_count*p + rank;
                    localFinished = localFinished && M[index] != -1;

                    index = vertex_count*p + rank + n;
                    localFinished = localFinished && M[index] != -1;
                }

                //checking if all the "normal" vertices are matched
                for(int i = 0; i < vertex_count && localFinished; i++)
                {
                    index = vertex_count*rank + i;
                    localFinished = localFinished && M[index] != -1;

                    index = vertex_count*rank + i + n;
                    localFinished = localFinished && M[index] != -1;
                }
                localChangeOccured = localFinished;
                if(localFinished && SHOW_COMPLETIONS)
                    std::cout << "*******Rank " << rank << " is done.******** " << std::endl;
            }
            else
                localChangeOccured = false;

            if(SHOW_INTERMEDIATE_RESULTS) displayMatrix(C, 1, C.size(), false, "Rank " + std::to_string(rank) + "[" + std::to_string(localFinished) + "] C = ");
            if(SHOW_INTERMEDIATE_RESULTS) displayMatrix(M, 1, M.size(), false, "Rank " + std::to_string(rank) + "[" + std::to_string(localFinished) + "] M = ");
            if(SHOW_INTERMEDIATE_RESULTS) displayMatrix(finishedProcesses, 1, finishedProcesses.size(), false, "Rank " + std::to_string(rank) + "[" + std::to_string(localFinished) + "] F = ");
            
            MPI_Allgather(&localFinished, 1, MPI_INT, &recentFinishedProcesses[0], 1, MPI_INT, MPI_COMM_WORLD);

            for(int i = 0; i < p && !localFinished; i++)
            {
                if(i != rank && recentFinishedProcesses[i] && !finishedProcesses[i])
                {
                    int index;

                    //removing all the excess vertices that are associated with rank i
                    if(vertex_count*p != n && i < n - vertex_count*p)
                    {
                        index = vertex_count*p + i;
                        M[index] = 2*n;

                        index = vertex_count*p + i + n;
                        M[index] = 2*n;
                    }

                    //removing all the "normal" vertices that are associated with rank i
                    for(int j = 0; j < vertex_count; j++)
                    {
                        index = vertex_count*i + j;
                        M[index] = 2*n;

                        index = vertex_count*i + j + n;
                        M[index] = 2*n;
                    }
                }
            }
            finishedProcesses = recentFinishedProcesses;

            MPI_Allreduce(&localChangeOccured, &globalChangeOccured, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        }
        //end of determining if locally complete, adjusting to others that are locally complete and determining if the graph is globally complete--------------------------------------------

        //beginning of determining whether the algorithm should end--------------------------------------------
        if(PRINT_DEBUG_STATEMENTS) std::cout << "Determining Algorithm Termination at Rank " << rank << std::endl;
        int globalFinished = true;
        for(int i = 0; i < p && globalFinished; i++) 
            globalFinished = globalFinished && finishedProcesses[i];

        if(globalFinished)
            break;
        //end of determining whether the algorithm should end--------------------------------------------

        //beginning of generating requests--------------------------------------------
        if(PRINT_DEBUG_STATEMENTS && !localFinished) std::cout << "Generating Requests at Rank " << rank << std::endl;

        //generating the matches from the excess vertices 
        if(vertex_count*p != n && rank < n - vertex_count*p)
        {
            int index = vertex_count*p + rank;
            int goal = (C[index] - (C[index] < n ? 0 : n)); //not the complete goal yes
            if(goal < vertex_count*p)
                goal = goal / vertex_count; //requesting a vertex with 0 <= vertex_id < vertex_count*p
            else
                goal = goal - vertex_count*p; //requesting a vertex with vertex_count*p <= vertex_id < n

            if(M[index] == -1 && goal != rank && !finishedProcesses[goal])
            {
                vertexSendRequests[goal].push_back(C[index]);
                vertexSendRequests[goal].push_back(index);
            }

            index = vertex_count*p + rank + n;
            goal = (C[index] - (C[index] < n ? 0 : n)); //not the complete goal yes
            if(goal < vertex_count*p)
                goal = goal / vertex_count;
            else
                goal = goal - vertex_count*p;
            if(M[index] == -1 && goal != rank && !finishedProcesses[goal])
            {
                vertexSendRequests[goal].push_back(C[index]);
                vertexSendRequests[goal].push_back(index);
            }
        }

        for(int i = 0; i < vertex_count && !localFinished; i++)
        {
            int index = vertex_count*rank + i;
            int goal = (C[index] - (C[index] < n ? 0 : n)); //not the complete goal yes
            if(goal < vertex_count*p)
                goal = goal / vertex_count; //requesting a vertex with 0 <= vertex_id < vertex_count*p
            else
                goal = goal - vertex_count*p; //requesting a vertex with vertex_count*p <= vertex_id < n

            if(M[index] == -1 && goal != rank && !finishedProcesses[goal])
            {
                vertexSendRequests[goal].push_back(C[index]);
                vertexSendRequests[goal].push_back(index);
            }
            
            index = vertex_count*rank + i + n;
            goal = (C[index] - (C[index] < n ? 0 : n)); //not the complete goal yes
            if(goal < vertex_count*p)
                goal = goal / vertex_count; //requesting a vertex with 0 <= vertex_id < vertex_count*p
            else
                goal = goal - vertex_count*p; //requesting a vertex with vertex_count*p <= vertex_id < n
            if(M[index] == -1 && goal != rank && !finishedProcesses[goal])
            {
                vertexSendRequests[goal].push_back(C[index]);
                vertexSendRequests[goal].push_back(index);
            }
        }
        //end of generating requests--------------------------------------------

        //beginning of sending request sizes--------------------------------------------
        if(PRINT_DEBUG_STATEMENTS && !localFinished) std::cout << "Sending Request Sizes at Rank " << rank << std::endl;
        tag = 0;
        for(int i = 0; i < p && !localFinished; i++)
        {
            if(i != rank && !finishedProcesses[i])
            {
                MPI_Request req;
                vertexSendRequests[i].shrink_to_fit();
                vertexSendSizeRequests[i] = vertexSendRequests[i].size();
                MPI_Isend(&vertexSendSizeRequests[i], 1, MPI_INT, i, tag, MPI_COMM_WORLD, &req);
                vertexSendSizeRequestsMessages.push_back(req);
            }
        }
        //end of sending request sizes--------------------------------------------

        //beginning of receiving request sizes--------------------------------------------
        if(PRINT_DEBUG_STATEMENTS && !localFinished) std::cout << "Receiving Request Sizes at Rank " << rank << std::endl;
        tag = 0;
        for(int i = 0; i < p && !localFinished; i++)
        {
            if(i != rank && !finishedProcesses[i])
            {
                MPI_Request req;
                MPI_Irecv(&vertexRecvSizeRequests[i], 1, MPI_INT, i, tag, MPI_COMM_WORLD, &req);
                vertexRecvSizeRequestsMessages.push_back(req);
            }
        }
        //end of receiving request sizes--------------------------------------------

        vertexRecvSizeRequestsMessages.shrink_to_fit();
        MPI_Waitall(vertexRecvSizeRequestsMessages.size(), &vertexRecvSizeRequestsMessages[0],  MPI_STATUSES_IGNORE);
        vertexSendSizeRequestsMessages.shrink_to_fit();
        MPI_Waitall(vertexSendSizeRequestsMessages.size(), &vertexSendSizeRequestsMessages[0],  MPI_STATUSES_IGNORE);

        //beginning of sending requests--------------------------------------------
        if(PRINT_DEBUG_STATEMENTS && !localFinished) std::cout << "Sending Requests at Rank " << rank << std::endl;
        tag = 1;
        for(int i = 0; i < p && !localFinished; i++)
        {
            if(vertexSendSizeRequests[i] != 0 && i != rank && !finishedProcesses[i])
            {
                MPI_Request req;
                MPI_Isend(&vertexSendRequests[i][0], vertexSendSizeRequests[i], MPI_INT, i, tag, MPI_COMM_WORLD, &req);
                vertexSendRequestsMessages.push_back(req);
            }
        }
        //end of sending requests--------------------------------------------

        //beginning of receiving requests--------------------------------------------
        if(PRINT_DEBUG_STATEMENTS && !localFinished) std::cout << "Receiving Requests at Rank " << rank << std::endl;
        tag = 1;
        for(int i = 0; i < p && !localFinished; i++)
        {
            if(vertexRecvSizeRequests[i] != 0 && rank != i && !finishedProcesses[i])
            {
                MPI_Request req;
                vertexRecvRequests[i].resize(vertexRecvSizeRequests[i], 0);
                MPI_Irecv(&vertexRecvRequests[i][0], vertexRecvSizeRequests[i], MPI_INT, i, tag, MPI_COMM_WORLD, &req);
                vertexRecvRequestsMessages.push_back(req);
            }
        }
        //end of receiving requests--------------------------------------------

        vertexRecvRequestsMessages.shrink_to_fit();
        MPI_Waitall(vertexRecvRequestsMessages.size(), &vertexRecvRequestsMessages[0],  MPI_STATUSES_IGNORE);
        vertexSendRequestsMessages.shrink_to_fit();
        MPI_Waitall(vertexSendRequestsMessages.size(), &vertexSendRequestsMessages[0],  MPI_STATUSES_IGNORE);
        
        //beginning of processing requests and generating responses--------------------------------------------
        if(PRINT_DEBUG_STATEMENTS && !localFinished) std::cout << "Processing Requests and Generating Responses at Rank " << rank << std::endl;
        for(int i  = 0; i < p && !localFinished; i++)
        {
            if(vertexRecvRequests[i].size() != 0 && i != rank && !finishedProcesses[i])
            {
                vertexSendResponses[i].resize(vertexRecvSizeRequests[i]/2, false);
                for(int j = 0; j < vertexRecvSizeRequests[i]; j = j + 2)
                {
                    vertexSendResponses[i][j/2] = (C[vertexRecvRequests[i][j]] == vertexRecvRequests[i][j + 1] ? 1 : 0);
                    if(M[vertexRecvRequests[i][j]] != -1)
                        vertexSendResponses[i][j/2] = -1;

                    if(vertexSendResponses[i][j/2] == 1)
                    {
                        M[C[vertexRecvRequests[i][j]]] = vertexRecvRequests[i][j + 1];
                        M[vertexRecvRequests[i][j + 1]] = C[vertexRecvRequests[i][j]];
                        D.insert(vertexRecvRequests[i][j]);
                    }
                }
            }
        }
        //end of processing requests and generating responses--------------------------------------------
		
        //beginning of sending responses--------------------------------------------
        if(PRINT_DEBUG_STATEMENTS && !localFinished) std::cout << "Sending Responses at Rank " << rank << std::endl;
        tag = 2;
        for(int i = 0; i < p && !localFinished; i++)
        {
            if(vertexRecvSizeRequests[i] != 0 && i != rank && !finishedProcesses[i])
            {
                MPI_Request req;
                MPI_Isend(&vertexSendResponses[i][0], vertexRecvSizeRequests[i]/2, MPI_INT, i, tag, MPI_COMM_WORLD, &req);
                vertexSendResponsesMessages.push_back(req);
            }
        }
        //end of sending responses--------------------------------------------

        //beginning of receiving responses--------------------------------------------
        if(PRINT_DEBUG_STATEMENTS && !localFinished) std::cout << "Receiving Responses at Rank " << rank << std::endl;
        tag = 2;
        for(int i = 0; i < p && !localFinished; i++)
        {
            if(vertexSendSizeRequests[i] != 0 && i != rank && !finishedProcesses[i])
            {
                MPI_Request req;
                vertexRecvResponses[i].resize(vertexSendSizeRequests[i]/2, 0);
                MPI_Irecv(&vertexRecvResponses[i][0], vertexSendSizeRequests[i]/2, MPI_INT, i, tag, MPI_COMM_WORLD, &req);
                vertexRecvResponsesMessages.push_back(req);
            }
        }
        //end of receiving responses--------------------------------------------

        vertexRecvResponsesMessages.shrink_to_fit();
        MPI_Waitall(vertexRecvResponsesMessages.size(), &vertexRecvResponsesMessages[0], MPI_STATUSES_IGNORE);
        vertexSendResponsesMessages.shrink_to_fit();
        MPI_Waitall(vertexSendResponsesMessages.size(), &vertexSendResponsesMessages[0], MPI_STATUSES_IGNORE);
        
        //beginning of processing responses--------------------------------------------
        if(PRINT_DEBUG_STATEMENTS && !localFinished) std::cout << "Processing Responses at Rank " << rank << std::endl;
        for(int i = 0; i < p && !localFinished; i++)
        {
            //if(vertexRecvResponses[i] != NULL)
            if(vertexSendSizeRequests[i] != 0 && i != rank)
            {  
                for(int j = 0; j < vertexSendSizeRequests[i]/2; j++)
                {
                    if(vertexRecvResponses[i][j] == 1)
                    {
                        M[vertexSendRequests[i][2*j]] = vertexSendRequests[i][2*j + 1];
                        M[vertexSendRequests[i][2*j + 1]] = vertexSendRequests[i][2*j];
                        D.insert(vertexSendRequests[i][2*j + 1]);
                    }
                    else if(vertexRecvResponses[i][j] == -1)
                        M[vertexSendRequests[i][2*j]] = 2*n;
                }
            }
        }
        //ending of processing responses--------------------------------------------
	}

    finishedProcesses.clear();

    int out_index = 0;
    for(int i = 0; i < vertex_count; i++)
    {
        int index = vertex_count*rank + i;
        output[out_index++] = index;
        output[out_index++] = M[index];
    }
    if(vertex_count*p != n)
    {
        if(rank < n-vertex_count*p)
        {
            output[out_index++] = vertex_count*p+rank;
            output[out_index++] = M[vertex_count*p+rank];
        }
        else
        {
            output[out_index++] = -1; 
            output[out_index++] = -1;
        }
    }
    //*************************THIS SHOULD BE THE END OF THE ALGORITHM*************************
    if(rank == 0 && !TESTING) std::cout << "Finished." << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    if(SHOW_INDIVIDUAL_RESULTS) displayMatrix(M, 1, M.size(), false, "Rank " + std::to_string(rank) + ":");

}

void FindBestMatch(std::unordered_set<int>& D, std::vector<int>& C, std::vector<int>& M, std::vector<int>& matrix, std::vector<int>& matrix_excess_node, std::vector<int>& transpose, std::vector<int>& transpose_excess_node, std::vector<int>& sortedMatrixNeighbors, std::vector<int>& sortedTransposeNeighbors, std::vector<int>& sortedMatrixExcessNeighbors, std::vector<int>& sortedTransposeExcessNeighbors, std::vector<int>& indexOfSortedNeighborIndexes, int n, int p, int rank)
{
    int vertex_count = n/p;

    #if INCLUDE_SORTING
    if(vertex_count*p != n && rank < n - vertex_count*p)
    {
        int index = vertex_count*p + rank;
        if(M[index] == -1)
        {
            C[index] = MaxIndex(sortedMatrixExcessNeighbors, indexOfSortedNeighborIndexes, M, vertex_count*2, n, p, rank) + n;
            if(C[C[index]] == index)
            {
                D.insert(index);
                D.insert(C[index]);
                M[C[index]] = index;
                M[index] = C[index];
            }
        }

        index = vertex_count*p + rank + n;
        if(M[index] == -1)
        {
            C[index] = MaxIndex(sortedTransposeExcessNeighbors, indexOfSortedNeighborIndexes, M, vertex_count*2+1, n, p, rank);
            if(C[C[index]] == index)
            {
                D.insert(index);
                D.insert(C[index]);
                M[C[index]] = index;
                M[index] = C[index];
            }
        }
    }

    for(int i = 0; i < vertex_count; i++)
    {
        int index = vertex_count*rank + i;
        C[index] = MaxIndex(sortedMatrixNeighbors, indexOfSortedNeighborIndexes, M, i, n, p, rank) + n;
        if(C[C[index]] == index)
        {
            D.insert(index);
            D.insert(C[index]);
            M[C[index]] = index;
            M[index] = C[index];
        }
    }

    for(int i = 0; i < vertex_count; i++)
    {
        int index = vertex_count*rank + i + n;
        C[index] = MaxIndex(sortedTransposeNeighbors, indexOfSortedNeighborIndexes, M, i+vertex_count, n, p, rank);
        if(C[C[index]] == index)
        {
            D.insert(index);
            D.insert(C[index]);
            M[C[index]] = index;
            M[index] = C[index];
        }
    }
    #else
    //finding the best match for the excess vertices
    if(vertex_count*p != n && rank < n - vertex_count*p)
    {
        int index = vertex_count*p + rank;
        if(M[index] == -1)
        {
            C[index] = MaxIndex(transpose_excess_node, 0, n, M, n) % n + n;

            if(C[C[index]] == index)
            {
                D.insert(index);
                D.insert(C[index]);
                M[C[index]] = index;
                M[index] = C[index];
            }
        }

        index = vertex_count*p + rank + n;
        if(M[index] == -1)
        {
            C[index] = MaxIndex(matrix_excess_node, 0, n, M, 0) % n;

            if(C[C[index]] == index)
            {
                D.insert(index);
                D.insert(C[index]);
                M[C[index]] = index;
                M[index] = C[index];
            }
        }
    }

    //finding the best match for the "normal" vertices in the matrix
    for(int i = 0; i < vertex_count; i++)
    {
        int index = vertex_count*rank + i;
        if(M[index] == -1)
        {
            C[index] = MaxIndex(transpose, i*n, (i+1)*n, M, n) % n + n;

            if(C[C[index]] == index)
            {
                D.insert(index);
                D.insert(C[index]);
                M[C[index]] = index;
                M[index] = C[index];
            }
        }
    }

    //finding the best match for the "normal" vertices in the transpose
    for(int i = 0; i < vertex_count;  i++)
    {
        int index = vertex_count*rank + i + n;
        if(M[index] == -1)
        {
            C[index] = MaxIndex(matrix, i*n, (i+1)*n, M, 0) % n; //rows look for matches in the columns
            
            if(C[C[index]] == index)
            {
                D.insert(index);
                D.insert(C[index]);
                M[C[index]] = index;
                M[index] = C[index];
            }
        }
    }
    #endif
}

void ProcessNeighborsOfMatched(std::unordered_set<int>& D, std::vector<int>& C, std::vector<int>& M, std::vector<int>& matrix, std::vector<int>& matrix_excess_node, std::vector<int>& transpose, std::vector<int>& transpose_excess_node, std::vector<int>& sortedMatrixNeighbors, std::vector<int>& sortedTransposeNeighbors, std::vector<int>& sortedMatrixExcessNeighbors, std::vector<int>& sortedTransposeExcessNeighbors, std::vector<int>& indexOfSortedNeighborIndexes, int n, int p, int rank)
{
    int vertex_count = n/p;
    while(!D.empty())
    {
        //std::vector<int> searchMatrix;
        int v, index;
        v = *(D.begin());
        D.erase(v);

        #if INCLUDE_SORTING
        if(vertex_count*p != n && rank < n - vertex_count*p)
        {
            if(v < n)
            {
                index = vertex_count*p + rank + n;
                //std::cout << "Rank " << rank << " I3 " << index << " " << sortedTransposeExcessNeighbors.size() << std::endl;
                if(M[index] == -1) C[index] = MaxIndex(sortedTransposeExcessNeighbors, indexOfSortedNeighborIndexes, M, vertex_count*2+1, n, p, rank);
                //std::cout << "Rank " << rank << " C3 " << C[index] << std::endl;

            } 
            else
            {
                index = vertex_count*p + rank;
                //std::cout << "Rank " << rank << " I4 " << index << " " << sortedMatrixExcessNeighbors.size() << std::endl;
                if(M[index] == -1) C[index] = MaxIndex(sortedMatrixExcessNeighbors, indexOfSortedNeighborIndexes, M, vertex_count*2, n, p, rank) + n;
                //std::cout << "Rank " << rank << " C4 " << C[index] << std::endl;
            } 
            if(C[C[index]] == index)
            {
                D.insert(index);
                D.insert(C[index]);
                M[index] = C[index];
                M[C[index]] = index;
            }
        }

        for(int x = 0; x < vertex_count; x++)
        {
            if(v < n)
            {
                index = vertex_count*rank + x + n;
                //std::cout << "Rank " << rank << " I1 " << index << " " << sortedTransposeNeighbors.size() << std::endl;
                if(M[index] == -1) C[index] = MaxIndex(sortedTransposeNeighbors, indexOfSortedNeighborIndexes, M, x+vertex_count, n, p, rank);
                //std::cout << "Rank " << rank << " C1 " << C[index] << std::endl;
            }
            else
            {
                index = vertex_count*rank + x;
                //std::cout << "Rank " << rank << " I2 " << index << " " << sortedMatrixNeighbors.size() << std::endl;
                if(M[index] == -1) C[index] = MaxIndex(sortedMatrixNeighbors, indexOfSortedNeighborIndexes, M, x, n, p, rank) + n;
                //std::cout << "Rank " << rank << " C2 " << C[index] << std::endl;
            }
            if(C[C[index]] == index)
            {
                D.insert(index);
                D.insert(C[index]);
                M[index] = C[index];
                M[C[index]] = index;
            }
        }
        #else
        for(int x = 0; x < vertex_count; x++)
        {
            if(v < n)
            {
                index = vertex_count*rank + x + n;
                if(M[index] == -1)
                {
                    //doing computations on the excess vertices in the transpose
                    if(vertex_count*p != n && rank < n - vertex_count*p)
                    {
                        if(M[vertex_count*p + rank + n] == -1)
                        {
                            C[index] = MaxIndex(transpose_excess_node, 0, n, M, 0) % n;
                            if(C[C[index]] == index)
                            {
                                D.insert(index);
                                D.insert(C[index]);
                                M[index] = C[index];
                                M[C[index]] = index;
                            }
                        }
                    }

                    //doing computations on the "normal" vertices in the transpose
                    if(M[index] == -1)
                    {
                        C[index] = MaxIndex(transpose, x*n, (x+1)*n, M, 0) % n;
                        if(C[C[index]] == index)
                        {
                            D.insert(index);
                            D.insert(C[index]);
                            M[index] = C[index];
                            M[C[index]] = index;
                        }
                    }
                }
            }
            else
            {
                index = vertex_count*rank + x;
                if(M[index] == -1)
                {
                    //doing computations on the excess vertices in the matrix
                    if(vertex_count*p != n && rank < n - vertex_count*p)
                    {
                        if(M[vertex_count*p + rank] == -1)
                        {
                            C[index] = MaxIndex(matrix_excess_node, 0, n, M, n) % n + n;
                            if(C[C[index]] == index)
                            {
                                D.insert(index);
                                D.insert(C[index]);
                                M[index] = C[index];
                                M[C[index]] = index;
                            }
                        }
                    }

                    //doing computations on the excess vertices in the matrix
                    if(M[index] == -1)
                    {
                        C[index] = MaxIndex(matrix, x*n, (x+1)*n, M, n) % n + n;
                        if(C[C[index]] == index)
                        {
                            D.insert(index);
                            D.insert(C[index]);
                            M[index] = C[index];
                            M[C[index]] = index;
                        }
                    }
                }
            }
        }
        #endif
    }
}

int MaxIndex(std::vector<int>& matrix, int start, int end, std::vector<int>& M, int shift)
{
    int index, maxIndex;
    maxIndex = start;
    for(int i = start; i < end; i++)
        if(M[i - start + shift] == -1)
        {
            index = maxIndex = i;
            break;
        }
    
    for(int i = index; i < end; i++)
        if(matrix[i] > matrix[maxIndex] && M[i - start + shift] == -1)
            maxIndex = i;
    
    return maxIndex;
}

int MaxIndex(std::vector<int>& neighbors, std::vector<int>& indexOfSortedNeighborIndexes, std::vector<int>& M, int local_vertex_num, int n, int p, int rank)
{
    int currentIndex, neighborShift, shift, vertex_count, maxIndex;
    vertex_count = n/p;

    if(local_vertex_num < vertex_count)
    {
        neighborShift = local_vertex_num*n;
        shift = n;
        //std::cout << "1" << std::endl;
    }
    else if(local_vertex_num < vertex_count*2)
    {
        neighborShift = (local_vertex_num - vertex_count)*n;
        shift = 0;
        //std::cout << "2" << std::endl;
    }
    else if(local_vertex_num == vertex_count*2)
    {
        neighborShift = 0;
        shift = n;
        //std::cout << "3" << std::endl;
    }
    else if(local_vertex_num == (vertex_count*2 + 1))
    {
        neighborShift = 0;
        shift = 0;
    }

    while(true)
    {
        currentIndex = neighborShift + indexOfSortedNeighborIndexes[local_vertex_num];
        maxIndex = neighbors[currentIndex];
        if(M[maxIndex + shift] == -1)
            break;
        indexOfSortedNeighborIndexes[local_vertex_num] = indexOfSortedNeighborIndexes[local_vertex_num] + 1;
    }
    std::cout << "out with " << maxIndex << " for local " << local_vertex_num << std::endl;
    return maxIndex;
}

void SortNeighbors(std::vector<int>& matrix, std::vector<int>& neighbors, int start, int end)
{
    auto cmpr = [&matrix, &neighbors, start](int a, int b){return matrix[a + start] > matrix[b + start];};
    std::sort(neighbors.begin() + start, neighbors.begin() + end, cmpr);
}

void createMatrix(std::vector<int>& matrix, int n, int m)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    int min, max;

    min = -9; //std::numeric_limits<int>::min();
    max = 9; //std::numeric_limits<int>::max();
    if(SHOW_MATRICES_AND_RESULTS)
    {
        min = -9;
        max = 9;
    }
    
    matrix.clear();
    matrix.resize(n*m);
    std::uniform_int_distribution<int> dist(min, max);
    for (unsigned i = 0; i < n; i++)
        for(unsigned j = 0; j < m; j++)
            matrix[i*m + j] = dist(gen);
}

void generateTranspose(std::vector<int>& matrix, std::vector<int>& transpose, int n, int m)
{
    transpose.clear();
    transpose.resize(n*m);
    for(unsigned i = 0; i < n; i++)
        for(unsigned j = 0; j < m; j++)
            transpose[j*n + i] = matrix[i*m + j];
}

void displayMatrix(std::vector<int>& matrix, int rows, int cols, bool result, std::string prefix)
{
    if(result)
    {
        for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
                std::printf("%7d ", matrix[i*cols + j]);
            std::cout << std::endl;
        }
    }
    else
    {
        std::string output = prefix;
        for(int i = 0; i < rows*cols; i++)
            output = output + (i%rows == 0 && i != 0 ? "| " : "") + ((matrix[i] >= 0) ? " " : "") + (abs(matrix[i] > 9) ? "" : " ") + std::to_string(matrix[i]) + " ";
        std::cout << output << std::endl;
    }
}