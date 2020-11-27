#include "./main.hpp"

#define PRINT_DEBUG_STATEMENTS 0
#define SHOW_MATRICES_AND_RESULTS 0
#define TESTING 1
#define SHOW_COMPLETIONS 0
#define SHOW_INDIVIDUAL_RESULTS 0
#define SHOW_INTERMEDIATE_RESULTS 0
#define MIN(a, b) (((a) < (b)) ? (a) : (b)) 

int main(int argc, char** argv)
{
    #if PRINT_DEBUG_STATEMENTS
        std::cout << "main" << std::endl;
    #endif
    
    //used for measuring performance, just declaring not using current values
    auto start = std::chrono::high_resolution_clock::now();

    //same and sent
    int p; //number of processes
    int n; //number of items in the first vertex disjoint set; number of rows in the matrix
    int m; //number of items in the second vertex disjoint set; number of cols in the matrix

    //same and computed
    int vertex_count_n; //also the vertex count per vertex disjoint set
    int vertex_count_m; //also the vertex count per 
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

    if(argc ==  3)
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
            m = atoi(argv[2]);

            if(n <= 0 || m <= 0)
            {
                std::cout << "The dimentions of the matrix must be postive integers." << std::endl;
                exit(0);
            }
            else if(n < p || m < p)
            {
                std::cout << "Right now, the application works under the assumption that both the matrix dimentions (individually) must at least match the number of nodes to be used." << std::endl;
            }

            createMatrix(matrix, n, m);
            #if SHOW_MATRICES_AND_RESULTS
                displayMatrix(matrix, n, m, true, "Rank " + std::to_string(rank) + " Matrix\n");
            #endif
        }
        
        if(rank == 0)
            start = std::chrono::high_resolution_clock::now();

        //send out the size of the matrix
        #if PRINT_DEBUG_STATEMENTS
            std::cout << "Sending/Receiving the Matrix Dimentions at Rank " << rank << std::endl;
        #endif
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

        //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
        vertex_count_n = n/p;
        vertex_count_m = m/p;

        //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
        //each node outputs the least amount of information
        if(n <= m)
        {
            output_count = 2*vertex_count_n;
            if(vertex_count_n*p != n)
                output_count += 2;
        }
        else
        {
            output_count = 2*vertex_count_m;
            if(vertex_count_m*p != m)
                output_count += 2;
        }
        output.resize(output_count, -1); //this holds the final edges, comes in pairs of two
        
        //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
        if(rank == 0)
        {                
            generateTranspose(matrix, transpose, n, m);

            #if SHOW_MATRICES_AND_RESULTS
                displayMatrix(transpose, m, n, true, "Rank " + std::to_string(rank) + " Transpose\n");
            #endif

            std::vector<MPI_Request> requests(p-1);
            
            //send out the matrices
            #if PRINT_DEBUG_STATEMENTS
                std::cout << "Sending the Matrix Portions at Rank " << rank << std::endl;
            #endif

            for(int i = 1; i < p; i++)
                MPI_Isend(&matrix[vertex_count_n*i*m], vertex_count_n*m, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[i-1]);

            //wait until everyone received their transpose segments
            #if PRINT_DEBUG_STATEMENTS
                std::cout << "Waiting Until Everyone Receives Their Matrix Portions at Rank " << rank << std::endl;
            #endif
            MPI_Waitall(p-1, &requests[0], MPI_STATUS_IGNORE);

            //sending remaining vertices in the matrix that aren't indexed between 0 and p*floor(n/p), known as excess matrix vertices in comments
            //wait until everyone received their transpose segments
            #if PRINT_DEBUG_STATEMENTS
                std::cout << "Sending the Excess Matrix Portions at Rank " << rank << std::endl;
            #endif
            if(vertex_count_n*p != n)
            {
                matrix_excess_node.resize(m);
                std::copy(matrix.begin() + vertex_count_n*p*m, matrix.begin() + vertex_count_n*p*m + m, matrix_excess_node.begin());

                for(int i = 1; i < n - vertex_count_n*p; i++)
                    MPI_Isend(&matrix[(vertex_count_n*p + i)*m], m, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[i-1]);
            }

            //wait until the excess matrix vertices have been distributed
            //wait until everyone received their transpose segments
            #if PRINT_DEBUG_STATEMENTS
                std::cout << "Waiting Until Every Excess Matrix Portions Has Been Received at Rank " << rank << std::endl;
            #endif
            MPI_Waitall(std::max(n - vertex_count_n*p - 1, 0), &requests[0], MPI_STATUS_IGNORE);

            //send out the transposes
            #if PRINT_DEBUG_STATEMENTS
                std::cout << "Sending the Transpose Portions at Rank " << rank << std::endl;
            #endif
            for(int i = 1; i < p; i++)
                MPI_Isend(&transpose[vertex_count_m*i*n], vertex_count_m*n, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[i-1]);

            //wait until everyone received their transpose segments
            #if PRINT_DEBUG_STATEMENTS
                std::cout << "Waiting Until Everyone Receives Their Transpose Portions at Rank " << rank << std::endl;
            #endif
            MPI_Waitall(p-1, &requests[0], MPI_STATUS_IGNORE);

            //sending remaining vertices in the transpose that aren't indexed between 0 and p*floor(n/p), known as excess transpose vertices in comments
            #if PRINT_DEBUG_STATEMENTS
                std::cout << "Sending the Excess Transpose Portions at Rank " << rank << std::endl;
            #endif
            if(vertex_count_m*p != m)
            {
                transpose_excess_node.resize(n);
                std::copy(transpose.begin() + vertex_count_m*p*n, transpose.begin() + vertex_count_m*p*n + n, transpose_excess_node.begin());

                for(int i = 1; i < m - vertex_count_m*p; i++)
                    MPI_Isend(&transpose[(vertex_count_m*p + i)*n], n, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[i-1]);
            }

            //wait until the excess transpose vertices have been distributed
            #if PRINT_DEBUG_STATEMENTS
                std::cout << "Waiting Until Every Excess Transpose Portion Has Been Received at Rank " << rank << std::endl;
            #endif
            MPI_Waitall(std::max(m - vertex_count_m*p - 1, 0), &requests[0], MPI_STATUS_IGNORE);
        }
        //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
        else
        {
            matrix.resize(vertex_count_n*m);
            transpose.resize(vertex_count_m*n);

            //receive portion of the matrix
            #if PRINT_DEBUG_STATEMENTS
                std::cout << "Waiting For My Portion of the Matrix at Rank " << rank << std::endl;
            #endif
            MPI_Recv(&matrix[0], vertex_count_n*m, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            //receiving excess matrix vertices
            #if PRINT_DEBUG_STATEMENTS
                if(vertex_count_n*p != n && rank < n-vertex_count_n*p)
                    std::cout << "Waiting For An Excess Matrix Portion at Rank " << rank << std::endl;
                else
                    std::cout << "Not Waiting For An Excess Matrix Portion at Rank " << rank << std::endl;
            #endif
            if(vertex_count_n*p != n && rank < n-vertex_count_n*p)
            {
                matrix_excess_node.resize(m);
                MPI_Recv(&matrix_excess_node[0], m, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
                
            //receive portion of the transpose
            #if PRINT_DEBUG_STATEMENTS
                std::cout << "Waiting For My Portion of the Transpose at Rank " << rank << std::endl;
            #endif
            MPI_Recv(&transpose[0], vertex_count_m*n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            //receiving excess transpose vertices
            #if PRINT_DEBUG_STATEMENTS
                if(vertex_count_m*p != m && rank < m-vertex_count_m*p)
                    std::cout << "Waiting For An Excess Transpose Portion at Rank " << rank << std::endl;
                else
                    std::cout << "Not Waiting For An Excess Transpose Portion at Rank " << rank << std::endl;
            #endif
            if(vertex_count_m*p != m && rank < m-vertex_count_m*p)
            {
                transpose_excess_node.resize(n);
                MPI_Recv(&transpose_excess_node[0], n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        //*************************END OF DATA DISTRIBUTION*************************

        #if PRINT_DEBUG_STATEMENTS
            std::cout << "Hello from Rank " << rank << std::endl;
        #endif

        //the algorithm implementation
        SequentialMatchingClass SequentialMatching(matrix, matrix_excess_node, transpose, transpose_excess_node, n, m, p, rank);
        SequentialMatching.GenerateMatching(output);
        
        //*************************START OF OUTPUT RETRIEVAL AND COMBINING*************************
        if(p != 1)
        {
            if(rank == 0)
                combinedResults.resize(p*output_count);
            MPI_Gather(&output[0], output_count, MPI_INT, &combinedResults[0], output_count, MPI_INT, 0, MPI_COMM_WORLD);
        }
        else
            combinedResults = output;
        //*************************END OF OUTPUT RETRIEVAL AND COMBINING*************************

        MPI_Finalize();

        if(rank == 0)
        {
            filteredResults.resize(MIN(n, m), -1);

            //filtering the results
            for(int i = 0; i < combinedResults.size(); i = i + 2)
                if(combinedResults[i] != -1 && combinedResults[i+1] != -1)
                {
                    if (n <= m)
                        filteredResults[combinedResults[i]] = combinedResults[i+1] - n;
                    else
                        filteredResults[combinedResults[i] - n] = combinedResults[i+1];
                }
                    
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
            
            #if SHOW_MATRICES_AND_RESULTS
                std::cout << "Final Results" << std::endl;
                for(int i = 0; i < filteredResults.size(); i++)
                {
                    if(n <= m)
                        std::cout << i << " " << filteredResults[i] << " " << matrix[i*m + filteredResults[i]] << std::endl;
                    else
                        std::cout << i << " " << filteredResults[i] << " " << matrix[filteredResults[i]*m + i] << std::endl;
                }
            #endif

            #if TESTING
                std::cout << duration.count() << std::endl; 
            #else
                std::cout << "Total Execution Time: " << duration.count() << " microseconds." << std::endl; 
            #endif
        }
    }
    else
        if(rank == 0)
            std::cout << "You did not provide the correct number of command line arguments, please pass three integers. The first two represent the dimentions of the matrix while the last represents the number of nodes/processors that OpenMPI can utilize." << std::endl;
    return 0;
}

//note that is is assumed that the indexes of a row are shifted by vertex_count
void SequentialMatchingClass::GenerateMatching(std::vector<int>& output)
{
    #if PRINT_DEBUG_STATEMENTS
        std::cout << "SequentialMatching at Rank " << rank << std::endl;
    #endif

    //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
    //determining if we need to continue to do local computations or continue all together
    int localFinished_n = false;
    int localFinished_m = false;

    //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
    std::vector<int> finishedProcesses_n(p, false);
    std::vector<int> finishedProcesses_m(p, false);

    int tag;

    //*************************THIS SHOULD BE THE START OF THE ALGORITHM*************************

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

        //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
        //recent completed processes
        std::vector<int> recentFinishedProcesses_n(p, false);
        std::vector<int> recentFinishedProcesses_m(p, false);

        //beginning of determining if locally complete, adjusting to others that are locally complete and determining if the graph is globally complete--------------------------------------------
        int globalChangeOccured = true;

        #if PRINT_DEBUG_STATEMENTS
            std::cout << "Determining Local and Global Changes at Rank " << rank << std::endl;
        #endif
        
        while(globalChangeOccured)
        {
            int localChangeOccured_n = false;
            int localChangeOccured_m = false;
            if(!localFinished_n || !localFinished_m)
            {
                #if PRINT_DEBUG_STATEMENTS
                    std::cout << "Determining Local Changes at Rank " << rank << std::endl;
                #endif
                
                FindBestMatch();
                    
                #if SHOW_INTERMEDIATE_RESULTS
                    for(int current_rank = 0; current_rank < p; current_rank++)
                    {
                        if(rank == current_rank)
                        {
                            std::vector<int> vecD;
                            vecD.reserve(D.size());
                            for (auto val = D.begin(); val != D.end(); ++val)
                                vecD.push_back(*val);
                                

                            displayMatrix(C, 1, C.size(), false, "Rank " + std::to_string(rank) + "[" + std::to_string(localFinished_n) + "][" + std::to_string(localFinished_m) + "] C = ");
                            displayMatrix(M, 1, M.size(), false, "Rank " + std::to_string(rank) + "[" + std::to_string(localFinished_n) + "][" + std::to_string(localFinished_m) + "] M = ");
                            displayMatrix(vecD, 1, vecD.size(), false, "Rank " + std::to_string(rank) + "[" + std::to_string(localFinished_n) + "][" + std::to_string(localFinished_m) + "] D = ");
                            std::cout << std::endl;
                        }
                    }
                #endif
                
                ProcessNeighborsOfMatched();

                int index;

                //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
                //checking if the matrix vertices have been matched
                if(!localFinished_n)
                {
                    localFinished_n = true;

                    //checking if the local excess matrix node has been matched
                    if(excess_vertex_n)
                    {
                        index = vertex_count_n*p + rank;
                        localFinished_n = localFinished_n && M[index] != -1;
                    }

                    //checking if the non-excess matrix vertices are matched
                    for(int i = 0; i < vertex_count_n && localFinished_n; i++)
                    {
                        index = vertex_count_n*rank + i;
                        localFinished_n = localFinished_n && M[index] != -1;
                    }

                    localChangeOccured_n = localFinished_n;
                }

                //checking if the transpose vetices have been matched
                if(!localFinished_m)
                {
                    localFinished_m = true;

                    //checking if the local excess transpose node has been matched
                    if(excess_vertex_m)
                    {
                        index = vertex_count_m*p + rank + n;
                        localFinished_m = localFinished_m && M[index] != -1;
                    }

                    //checking if the non-excess transpose vertices are matched
                    for(int i = 0; i < vertex_count_m && localFinished_m; i++)
                    {
                        index = vertex_count_m*rank + i + n;
                        localFinished_m = localFinished_m && M[index] != -1;
                    }
                    localChangeOccured_m = localFinished_m;
                }

                #if SHOW_COMPLETIONS
                    if(localFinished)
                        std::cout << "*******Rank " << rank << " is done.******** " << std::endl;
                #endif
            }
                
            
            MPI_Allgather(&localFinished_n, 1, MPI_INT, &recentFinishedProcesses_n[0], 1, MPI_INT, MPI_COMM_WORLD);
            MPI_Allgather(&localFinished_m, 1, MPI_INT, &recentFinishedProcesses_m[0], 1, MPI_INT, MPI_COMM_WORLD);

            #if PRINT_DEBUG_STATEMENTS
                std::cout << "Determining Global Changes at Rank " << rank << std::endl;
            #endif
            
            //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
            //removing all the matrix vertices associated with the ranks that have recently matched all of them
            for(int i = 0; i < p; i++)
            {
                if(i != rank && recentFinishedProcesses_n[i] && !finishedProcesses_n[i])
                {
                    int index;

                    //removing the excess matrix vertex that is associate with rank i
                    if(vertex_count_n*p != n && i < n - vertex_count_n*p)
                    {
                        index = vertex_count_n*p + i;
                        M[index] = n+m;
                    }

                    //removing all the non-excess matrix vertices that are associated with rank i
                    for(int j = 0; j < vertex_count_n; j++)
                    {
                        index = vertex_count_n*i + j;
                        M[index] = n+m;
                    }
                }
            }

            //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
            //removing all the transpose vertices assocaited with the ranks that have recently matched all of them
            for(int i = 0; i < p; i++)
            {
                if(i != rank && recentFinishedProcesses_m[i] && !finishedProcesses_m[i])
                {
                    int index;

                    //removing the excess transpose vertex that is associated with rank i
                    if(vertex_count_m*p != m && i < m - vertex_count_m*p)
                    {
                        index = vertex_count_m*p + i + n;
                        M[index] = n+m;
                    }

                    //removing all the non-excess transpose vertices that are associated with rank i
                    for(int j = 0; j < vertex_count_m; j++)
                    {
                        index = vertex_count_m*i + j + n;
                        M[index] = n+m;
                    }
                }
            }

            finishedProcesses_n = recentFinishedProcesses_n;
            finishedProcesses_m = recentFinishedProcesses_m;

            int globalChangeOccured_n, globalChangeOccured_m;
            MPI_Allreduce(&localChangeOccured_n, &globalChangeOccured_n, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            MPI_Allreduce(&localChangeOccured_m, &globalChangeOccured_m, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            globalChangeOccured = globalChangeOccured_n || globalChangeOccured_m;

            #if SHOW_INTERMEDIATE_RESULTS
                for(int current_rank = 0; current_rank < p; current_rank++)
                {
                    MPI_Barrier(MPI_COMM_WORLD);
                    if(rank == current_rank)
                    {
                        displayMatrix(C, 1, C.size(), false, "Rank " + std::to_string(rank) + "[" + std::to_string(localFinished_n) + "][" + std::to_string(localFinished_m) + "] C = ");
                        displayMatrix(M, 1, M.size(), false, "Rank " + std::to_string(rank) + "[" + std::to_string(localFinished_n) + "][" + std::to_string(localFinished_m) + "] M = ");
                        displayMatrix(finishedProcesses_n, 1, finishedProcesses_n.size(), false, "Rank " + std::to_string(rank) + "[" + std::to_string(localFinished_n) + "][" + std::to_string(localFinished_m) + "] Fn = ");
                        displayMatrix(finishedProcesses_m, 1, finishedProcesses_m.size(), false, "Rank " + std::to_string(rank) + "[" + std::to_string(localFinished_n) + "][" + std::to_string(localFinished_m) + "] Fm = ");
                        std::cout << std::endl;
                    }
                }
            #endif
        }
        //end of determining if anyone is locally complete and adjusting the remaining nodes accordingly--------------------------------------------

        //beginning of determining whether the algorithm should end--------------------------------------------
        #if PRINT_DEBUG_STATEMENTS 
            std::cout << "Determining Algorithm Termination at Rank " << rank << std::endl;
        #endif

        int globalFinished = true;
        for(int i = 0; i < p && globalFinished; i++)
        {
            if(n <= m)
                globalFinished = globalFinished && finishedProcesses_n[i];
            else
                globalFinished = globalFinished && finishedProcesses_m[i];
        }

        #if PRINT_DEBUG_STATEMENTS
            if(globalFinished)
                std::cout << "Algorithm Termination at Rank " << rank << std::endl;
            else
                std::cout << "Algorithm Continues at Rank " << rank << std::endl;
        #endif

        if(globalFinished)
            break;
        //end of determining whether the algorithm should end--------------------------------------------

        //used for determining if a rank should be communicated with
        int localFinished = localFinished_n && localFinished_m;
        std::vector<int> finishedProcesses(p, false);
        for(int i = 0; i < finishedProcesses.size(); i++)
            finishedProcesses[i] = finishedProcesses_n[i] && finishedProcesses_m[i]; 

        //beginning of generating requests--------------------------------------------
        #if PRINT_DEBUG_STATEMENTS
            if(!localFinished) std::cout << "Generating Requests at Rank " <<  rank << std::endl;
        #endif

        //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
        //generating the request from the excess matrix vertex
        if(excess_vertex_n && !localFinished_n)
        {
            int index = vertex_count_n*p + rank;
            int goal = C[index] - n; //(C[index] - (C[index] < n ? 0 : n)); //not the complete goal yes
            if(goal < vertex_count_n*p)
                goal = goal / vertex_count_n; //requesting a vertex with 0 <= vertex_id < vertex_count*p
            else
                goal = goal - vertex_count_n*p; //requesting a vertex with vertex_count*p <= vertex_id < n

            if(M[index] == -1 && goal != rank && !finishedProcesses_m[goal])
            {
                vertexSendRequests[goal].push_back(C[index]);
                vertexSendRequests[goal].push_back(index);
            }
        }

        //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
        //generating the request from the excess transpose vertex
        if(excess_vertex_m && !localFinished_m)
        {
            int index = vertex_count_m*p + rank + n;
            int goal = C[index]; //(C[index] - (C[index] < n ? 0 : n)); //not the complete goal yes
            if(goal < vertex_count_m*p)
                goal = goal / vertex_count_m;
            else
                goal = goal - vertex_count_m*p;
            if(M[index] == -1 && goal != rank && !finishedProcesses_n[goal])
            {
                vertexSendRequests[goal].push_back(C[index]);
                vertexSendRequests[goal].push_back(index);
            }
        }

        //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
        //generating the requests from the non-excess matrix vertices
        for(int i = 0; i < vertex_count_n && !localFinished_n; i++)
        {
            int index = vertex_count_n*rank + i;
            int goal = C[index] - n; //(C[index] - (C[index] < n ? 0 : n)); //not the complete goal yes
            if(goal < vertex_count_n*p)
                goal = goal / vertex_count_n; //requesting a vertex with 0 <= vertex_id < vertex_count*p
            else
                goal = goal - vertex_count_n*p; //requesting a vertex with vertex_count*p <= vertex_id < n

            if(M[index] == -1 && goal != rank && !finishedProcesses_m[goal])
            {
                vertexSendRequests[goal].push_back(C[index]);
                vertexSendRequests[goal].push_back(index);
            }
        }

        //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
        //generating the requests from the non-excess transpose vertices
        for(int i = 0; i < vertex_count_m && !localFinished_m; i++)
        {
            int index = vertex_count_m*rank + i + n;
            int goal = C[index]; //(C[index] - (C[index] < n ? 0 : n)); //not the complete goal yes
            if(goal < vertex_count_m*p)
                goal = goal / vertex_count_m; //requesting a vertex with 0 <= vertex_id < vertex_count*p
            else
                goal = goal - vertex_count_m*p; //requesting a vertex with vertex_count*p <= vertex_id < n
            if(M[index] == -1 && goal != rank && !finishedProcesses_n[goal])
            {
                vertexSendRequests[goal].push_back(C[index]);
                vertexSendRequests[goal].push_back(index);
            }
        }

        //end of generating requests--------------------------------------------

        //beginning of sending request sizes--------------------------------------------
        #if PRINT_DEBUG_STATEMENTS
            if(!localFinished) std::cout << "Sending Request Sizes at Rank " << rank << std::endl;
        #endif

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
        #if PRINT_DEBUG_STATEMENTS
            if(!localFinished) std::cout << "Receiving Request Sizes at Rank " << rank << std::endl;
        #endif

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
        #if PRINT_DEBUG_STATEMENTS
            if(!localFinished) std::cout << "Sending Requests at Rank " << rank << std::endl;
        #endif

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
        #if PRINT_DEBUG_STATEMENTS 
            if(!localFinished) std::cout << "Receiving Requests at Rank " << rank << std::endl;
        #endif

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
        #if PRINT_DEBUG_STATEMENTS
            if(!localFinished) std::cout << "Processing Requests and Generating Responses at Rank " << rank << std::endl;
        #endif
        
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
        #if PRINT_DEBUG_STATEMENTS 
            if(!localFinished) std::cout << "Sending Responses at Rank " << rank << std::endl;
        #endif
        
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
        #if PRINT_DEBUG_STATEMENTS 
            if(!localFinished) std::cout << "Receiving Responses at Rank " << rank << std::endl;
        #endif

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
        #if PRINT_DEBUG_STATEMENTS 
            if(!localFinished) std::cout << "Processing Responses at Rank " << rank << std::endl;
        #endif

        for(int i = 0; i < p && !localFinished; i++)
        {
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
                        M[vertexSendRequests[i][2*j]] = n+m;
                }
            }
        }
        //ending of processing responses--------------------------------------------
	}

    finishedProcesses_n.clear();
    finishedProcesses_m.clear();

    //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
    int out_index = 0;
    if(n <= m)
    {
        int index;
        for(int i = 0; i < vertex_count_n; i++)
        {
            index = vertex_count_n*rank + i;
            output[out_index++] = index;
            output[out_index++] = M[index];
        }

        if(excess_vertex_n)
        {
            index = vertex_count_n*p + rank;
            output[out_index++] = index;
            output[out_index++] = M[index];
        }
    }
    else
    {
        int index;
        for(int i = 0; i < vertex_count_m; i++)
        {
            index = vertex_count_m*rank + i + n;
            output[out_index++] = index;
            output[out_index++] = M[index];
        }

        if(excess_vertex_m)
        {
            index = vertex_count_m*p + rank + n;
            output[out_index++] = index;
            output[out_index++] = M[index];
        }
    }
    //*************************THIS SHOULD BE THE END OF THE ALGORITHM*************************
    MPI_Barrier(MPI_COMM_WORLD);
    #if !TESTING
        if(rank == 0) std::cout << "Finished." << std::endl;
    #endif
    #if SHOW_INDIVIDUAL_RESULTS 
        for(int current_rank = 0; current_rank < p; current_rank++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            if(rank == current_rank)
                displayMatrix(M, 1, M.size(), false, "Rank " + std::to_string(rank) + "Results = ");
        }
    #endif
}

void SequentialMatchingClass::FindBestMatch()
{
    #if PRINT_DEBUG_STATEMENTS
        std::cout << "FindBestMatch at Rank " << rank << std::endl;
    #endif

    //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
    //finding the best match for the excess matrix vertex
    if(excess_vertex_n)
    {
        int index = vertex_count_n*p + rank;
        SetUpdates(index, matrix_excess_node, 0, m, n, m, n);
    }

    //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
    //finding the best match for the excess transpose node
    if(excess_vertex_m)
    {
        int index = vertex_count_m*p + rank + n;
        SetUpdates(index, transpose_excess_node, 0, n, 0, n, 0);
    } 

    //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
    //finding the best match for each of the local non-excess vertices in the matrix
    for(int i = 0; i < vertex_count_n; i++)
    {
        int index = vertex_count_n*rank + i;
        SetUpdates(index, matrix, i*m, (i+1)*m, n, m, n);
    }

    //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
    //finding the best match for each of the local non-excess vertices in the transpose
    for(int i = 0; i < vertex_count_m; i++)
    {
        int index = vertex_count_m*rank + i + n;
        SetUpdates(index, transpose, i*n, (i+1)*n, 0, n, 0);
    }
}

void SequentialMatchingClass::ProcessNeighborsOfMatched()
{
    #if PRINT_DEBUG_STATEMENTS
        std::cout << "ProcessNeighborsOfMatched at Rank " << rank << std::endl;
    #endif

    //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
    while(!D.empty())
    {
        int v;
        v = *(D.begin());
        D.erase(v);

        if(IsMatrixIndex(v, true) || IsExcessMatrixIndex(v, true))
        {
            //doing computations on the excess vertices in the transpose
            if(excess_vertex_m)
            {
                int index = vertex_count_m*p + rank + n;
                SetUpdates(index, transpose_excess_node, 0, n, 0, n, 0);
            }

            //doing computations on the non-excess vertices in the transpose
            for(int i = 0; i < vertex_count_m; i++)
            {
                int index = vertex_count_m*rank + i + n;
                SetUpdates(index, transpose, i*n, (i+1)*n, 0, n, 0);
            }
        }
        else
        {
            //doing computations on the excess vertices in the matrix
            if(excess_vertex_n)
            {
                int index = vertex_count_n*p + rank;
                SetUpdates(index, matrix_excess_node, 0, m, n, m, n);
            }

            //doing computations on the non-excess vertices in the matrix
            for(int i = 0; i < vertex_count_n; i++)
            {
                int index = vertex_count_n*rank + i;
                SetUpdates(index, matrix, i*m, (i+1)*m, n, m, n);
            }
        }
    }
}

void SequentialMatchingClass::SetUpdates(int index, std::vector<int>& matrix, int start, int end, int shift, int modulus, int additive)
{
    #if PRINT_DEBUG_STATEMENTS
        std::cout << "SetUpdates at Rank " << rank << std::endl;
    #endif

    //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
    if(M[index] == -1)
    {
        C[index] = MaxIndex(matrix, start, end, shift) % modulus + additive;
        if(PairEvaluation(index, C[index]))
        {
            D.insert(index);
            D.insert(C[index]);
            M[C[index]] = index;
            M[index] = C[index];
        }
    }
}

int SequentialMatchingClass::MaxIndex(std::vector<int>& matrix, int start, int end, int shift)
{
    #if PRINT_DEBUG_STATEMENTS
        std::cout << "MaxIndex at Rank " << rank << std::endl;
    #endif

    int index, maxIndex;
    index = maxIndex = start;
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

bool SequentialMatchingClass::PairEvaluation(int localIndex, int potentialIndexPair)
{
    #if PRINT_DEBUG_STATEMENTS
        std::cout << "PairEvaluation at Rank " << rank << std::endl;
    #endif

    bool localPotential = IsLocalIndex(potentialIndexPair);

    if(C[localIndex] == -1) //the localIndex hasn't found someone it wants to pair with
        return false;
    
    if(M[localIndex] != -1) //the localIndex has already committed to another pair
        return false;

    if(localPotential)
    {
        if(C[potentialIndexPair] == -1)
            return false;
        if(M[potentialIndexPair] != -1)
            return false;
    }

    //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
    if(C[localIndex] == potentialIndexPair || C[potentialIndexPair] == localIndex)
        return true;
    else
    {
        if(IsMatrixIndex(localIndex))
        {
            int row = localIndex - vertex_count_n*rank;
            int col = potentialIndexPair - n;
            if(matrix[row*m + C[localIndex]] == matrix[row*m + col])
            {
                C[localIndex] = potentialIndexPair;
                C[potentialIndexPair] = localIndex;
                return true;
            }
        }

        if(IsTransposeIndex(localIndex))
        {
            int row = localIndex - vertex_count_m*rank - n;
            int col = potentialIndexPair;
            if(transpose[row*n + C[localIndex]] == transpose[row*n + col])
            {
                C[localIndex] = potentialIndexPair;
                C[potentialIndexPair] = localIndex;
                return true;
            }
        }
        
        if(IsExcessMatrixIndex(localIndex))
        {
            int row = 0;
            int col = potentialIndexPair - n;
            if(matrix_excess_node[row*m + C[localIndex]] == matrix_excess_node[row*m + col])
            {
                C[localIndex] = potentialIndexPair;
                C[potentialIndexPair] = localIndex;
                return true;
            }
        }

        if(IsExcessTransposeIndex(localIndex))
        {
            int row = 0;
            int col = potentialIndexPair;
            if(transpose_excess_node[row*n + C[localIndex]] == transpose_excess_node[row*n + col])
            {
                C[localIndex] = potentialIndexPair;
                C[potentialIndexPair] = localIndex;
                return true;
            }
        }
    } 
    return false;
}

bool SequentialMatchingClass::IsLocalIndex(int index)
{
    #if PRINT_DEBUG_STATEMENTS
        std::cout << "IsLocalIndex at Rank " << rank << std::endl;
    #endif

    bool result = IsMatrixIndex(index) || IsTransposeIndex(index) || IsExcessMatrixIndex(index) || IsExcessTransposeIndex(index);
    return result;
}

bool SequentialMatchingClass::IsMatrixIndex(int index, bool global)
{
    #if PRINT_DEBUG_STATEMENTS
        std::cout << "IsMatrixIndex at Rank " << rank << std::endl;
    #endif

    //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
    bool result = ((vertex_count_n*rank) <= index && index < (vertex_count_n*(rank + 1))) || (global && 0 <= index && index < vertex_count_n*p);
    return result;
}

bool SequentialMatchingClass::IsTransposeIndex(int index, bool global)
{
    #if PRINT_DEBUG_STATEMENTS
        std::cout << "IsTransposeIndex at Rank " << rank << std::endl;
    #endif

    //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
    bool result = ((vertex_count_m*rank + n) <= index && index < (vertex_count_m*(rank + 1) + n)) || (global && n <= index && index < vertex_count_m*p);
    return result;
}

bool SequentialMatchingClass::IsExcessMatrixIndex(int index, bool global)
{
    #if PRINT_DEBUG_STATEMENTS
        std::cout << "IsExcessMatrixIndex at Rank " << rank << std::endl;
    #endif

    //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
    bool result = ((excess_vertex_n && (index == (vertex_count_n*p + rank)))) || (global && vertex_count_n*p <= index && index < n);
    return result;
}

bool SequentialMatchingClass::IsExcessTransposeIndex(int index, bool global)
{
    #if PRINT_DEBUG_STATEMENTS
        std::cout << "IsExcessTransposeIndex at Rank " << rank << std::endl;
    #endif

    //TODO: ADJUST THIS TO ACCOUNT FOR THE FACT THAT ACCOUNTS FOR THE FACT THAT p>n OR p>m
    bool result = ((excess_vertex_m && (index == (vertex_count_m*p + rank + n)))) || (global && (n + vertex_count_m*p) <= index && index < (n+m));
    return result;
}

void createMatrix(std::vector<int>& matrix, int n, int m)
{
    #if PRINT_DEBUG_STATEMENTS
        std::cout << "createMatrix at Rank " << "UNKNOWN" << std::endl;
    #endif

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    int min, max;

    min = std::numeric_limits<int>::min();
    max = std::numeric_limits<int>::max();
    #if SHOW_MATRICES_AND_RESULTS
        min = -9;
        max = 9;
    #endif
    
    matrix.clear();
    matrix.resize(n*m);
    std::uniform_int_distribution<int> dist(min, max);
    for (unsigned i = 0; i < n; i++)
        for(unsigned j = 0; j < m; j++)
            matrix[i*m + j] = dist(gen);
}

void generateTranspose(std::vector<int>& matrix, std::vector<int>& transpose, int n, int m)
{
    #if PRINT_DEBUG_STATEMENTS
        std::cout << "generateTranspose at Rank " << "UNKNOWN" << std::endl;
    #endif

    transpose.clear();
    transpose.resize(n*m);
    for(unsigned i = 0; i < n; i++)
        for(unsigned j = 0; j < m; j++)
            transpose[j*n + i] = matrix[i*m + j];
}

void displayMatrix(std::vector<int>& matrix, int rows, int cols, bool result, std::string prefix)
{
    #if PRINT_DEBUG_STATEMENTS
        std::cout << "displayMatrix at Rank " << "UNKNOWN" << std::endl;
    #endif

    if(result)
    {
        std::cout << prefix;
        for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
                std::printf("%3d ", matrix[i*cols + j]);
            std::cout << std::endl;
        }
    }
    else
    {
        std::string output = prefix;
        for(int i = 0; i < rows*cols; i++)
            output = output + (i%cols == 0 && i != 0 ? "| " : "") + ((matrix[i] >= 0) ? " " : "") + + ((abs(matrix[i]) < 10) ? " " : "") + std::to_string(matrix[i]) + " ";
        std::cout << output << std::endl;
    }
}