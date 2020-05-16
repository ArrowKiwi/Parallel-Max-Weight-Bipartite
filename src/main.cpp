#include "./main.hpp"

#define PRINT_DEBUG_STATEMENTS 0
#define SHOW_MATRICES_AND_RESULTS 0
#define TESTING 0
#define PAPER_DEF 0
#define SHOW_COMPLETIONS 0

int main(int argc, char** argv)
{
    //used for measuring performance, just declaring not using current values
    auto start = std::chrono::high_resolution_clock::now();

    //same and sent
    int p; //processor name
    int n; //number of random numbers generated, assuming that it is divisible by p
    int tag = 0; //same for all

    //same and computed
    int vertex_count; //also the vertex count per vertex disjoint set; ideally the same 

    //unique and sent
    int rank;
    std::vector<int> matrix; //int * matrix;
    std::vector<int> transpose; //int * transpose;

    if(argc ==  2)
    {
        //MPI SETUP-------------------------
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &p);
        //std::cout << "Process Rank " << rank << std::endl;

        //SENDING INFORMATION TO NODES-----------------------
        if(rank == 0)
        {
            n = atoi(argv[1]); 
            //matrix = new int[n*n];
            //transpose = new int[n*n];
            matrix.resize(n*n);
            transpose.resize(n*n);
            createMatrices(matrix, transpose, n);
            if(SHOW_MATRICES_AND_RESULTS) displayMatrix(matrix, n, n, true, "");
            if(SHOW_MATRICES_AND_RESULTS) std::cout << "--------" << std::endl;
            if(SHOW_MATRICES_AND_RESULTS) displayMatrix(transpose, n, n, true, "");
        }

        if(rank == 0)
            start = std::chrono::high_resolution_clock::now();

        //send out the size of the matrix
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        vertex_count = n/p;
        
        if(rank == 0)
        {
            MPI_Request* requests = new MPI_Request[p-1];
            
            //send out the matrices
            for(int i = 1; i < p; i++)
                MPI_Isend(&matrix[n*vertex_count*i], vertex_count*n, MPI_INT, i, tag, MPI_COMM_WORLD, &requests[i-1]);
            
            //wait until everyone received their transpose segments
            MPI_Waitall(p-1, requests, MPI_STATUS_IGNORE);

            //send out the transposes
            for(int i = 1; i < p; i++)
                MPI_Isend(&transpose[n*vertex_count*i], vertex_count*n, MPI_INT, i, tag, MPI_COMM_WORLD, &requests[i-1]);

            //wait until everyone received their transpose segments
            MPI_Waitall(p-1, requests, MPI_STATUS_IGNORE);

            delete [] requests;
        }
        else
        {
            matrix.resize(vertex_count*n); //matrix = new int[vertex_count*n];
            transpose.resize(vertex_count*n); //transpose = new int[vertex_count*n];

            //receive portion of the matrix
            MPI_Recv(&matrix[0], vertex_count*n, MPI_INT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            //receive portion of the transpose
            MPI_Recv(&transpose[0], vertex_count*n, MPI_INT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if(PRINT_DEBUG_STATEMENTS) std::cout << "Hello from Rank " << rank << std::endl;
        SequentialMatching(matrix, transpose, vertex_count, rank, n, p);
        //delete[] matrix; 
        //delete[] transpose;
        MPI_Finalize();

        if(rank == 0)
        {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
            if(TESTING)
                std::cout << duration.count() << std::endl; 
            else
                std::cout << "Total Execution Time: " << duration.count() << " microseconds." << std::endl; 
        }
    }
    else
        if(rank == 0)
            std::cout << "You did not provide enough command line arguments, please enter a number of test integers and a number of nodes." << std::endl;

    return 0;
}

//note that is is assumed that the indexes of a row are shifted by vertex_count
//void SequentialMatching(int * matrix, int * transpose, int vertex_count, int rank, int n, int p)
void SequentialMatching(std::vector<int>& matrix, std::vector<int>& transpose, int vertex_count, int rank, int n, int p)
{
    std::vector<int> C(2*n, -1); //contains the current most ideal vertex pair
    std::vector<int> M(2*n, -1); //this holds the final edges, comes in pairs of two
    std::unordered_set<int> D; //all the matched vertices that have successfully been matched

    //finding internal dominating edges
    FindBestMatch(D, C, M, matrix, transpose, n, vertex_count, rank);
    //ProcessNeighborsOfMatched(D, C, M, matrix, transpose, vertex_count, rank, n, p);

    //determining if we need to do local computations or continue all together
    int localFinished = false;

    std::vector<int> finishedProcesses(p, false);
    int tag;

    while(true)
    {
        //sending request sizes structures
        std::vector<int> vertexSendSizeRequests(p,0);
        std::vector<MPI_Request> vertexSendSizeRequestsMessages;

        //receiving request size structures (pause after using these)
        std::vector<int> vertexRecvSizeRequests(p,0);
        std::vector<MPI_Request> vertexRecvSizeRequestsMessages;

        //sending request structures
        std::vector<std::vector<int>> vertexSendRequests(p);
        std::vector<MPI_Request> vertexSendRequestsMessages;

        //receiving requests (pause after using these)
        std::vector<std::vector<int>> vertexRecvRequests(p);
        std::vector<MPI_Request> vertexRecvRequestsMessages;

        //sending responses (assumming that the size will be equal to some constant times the request size)
        std::vector<std::vector<int>> vertexSendResponses(p);
        std::vector<MPI_Request> vertexSendResponsesMessages;

        //receiving responses (using previous assumption) (pause after using these)
        std::vector<std::vector<int>> vertexRecvResponses(p);
        std::vector<MPI_Request> vertexRecvResponsesMessages;

        //recent completed processes
        std::vector<int> recentFinishedProcesses(p, false);

        //beginning of determining if locally complete, adjusting to others that are locally complete and determining if the graph is globally complete--------------------------------------------
        int globalChangeOccured = true;
        if(PRINT_DEBUG_STATEMENTS) std::cout << "Determining Local And Global Changes at Rank " << rank << std::endl;
        while(globalChangeOccured)
        {
            #if PAPER_DEF==1
            globalChangeOccured = false;
            #endif

            if(!localFinished && D.size() == 0)
                FindBestMatch(D, C, M, matrix, transpose, n, vertex_count, rank);

            if(!localFinished)
                ProcessNeighborsOfMatched(D, C, M, matrix, transpose, vertex_count, rank, n, p);
            
            int localChangeOccured;
            if(!localFinished)
            {
                localFinished = true;
                for(int i = 0; i < vertex_count; i++)
                {
                    int index = rank*vertex_count + i;
                    localFinished = localFinished && M[index] != -1;

                    index = rank*vertex_count + i + n;
                    localFinished = localFinished && M[index] != -1;

                    if(!localFinished)
                        break;
                }
                if(localFinished)
                {
                    if(SHOW_COMPLETIONS) std::cout << "*******Rank " << rank << " is done.******** " << std::endl;
                    localChangeOccured = true;
                }
            }
            else
                localChangeOccured = false;

            MPI_Allgather(&localFinished, 1, MPI_INT, &recentFinishedProcesses[0], 1, MPI_INT, MPI_COMM_WORLD);

            #if PAPER_DEF==0
            for(int i = 0; i < p && !localFinished; i++)
            {
                if(i != rank && recentFinishedProcesses[i] && !finishedProcesses[i])
                {
                    for(int j = 0; j < vertex_count; j++)
                    {
                        int index = i*vertex_count + j;
                        M[index] = 2*n;

                        index = i*vertex_count + j + n;
                        M[index] = 2*n;
                    }
                }
            }
            finishedProcesses = recentFinishedProcesses;

            if(!localFinished)
            {
                FindBestMatch(D, C, M, matrix, transpose, n, vertex_count, rank);
                ProcessNeighborsOfMatched(D, C, M, matrix, transpose, vertex_count, rank, n, p);
            }

            MPI_Allreduce(&localChangeOccured, &globalChangeOccured, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            #endif
        }
        //end of determining if locally complete, adjusting to others that are locally complete and determining if the graph is globally complete--------------------------------------------

        //beginning of determining whether the algorithm should end--------------------------------------------
        if(PRINT_DEBUG_STATEMENTS) std::cout << "Determining Algorithm Termination at Rank " << rank << std::endl;
        int globalFinished = true;
        for(int i = 0; i < p; i++) 
        {
            globalFinished = globalFinished && finishedProcesses[i];
            if(!globalFinished)
                break;
        }

        if(globalFinished)
            break;
        //end of determining whether the algorithm should end--------------------------------------------

        //beginning of generating requests--------------------------------------------
        if(PRINT_DEBUG_STATEMENTS && !localFinished) std::cout << "Generating Requests at Rank " << rank << std::endl;
        for(int i = 0; i < vertex_count && !localFinished; i++)
        {
            int index = rank*vertex_count + i;
            int goal = (C[index] - (C[index] < n ? 0 : n)) / vertex_count;
            if(M[index] == -1 && goal != rank && !finishedProcesses[goal])
            {
                vertexSendRequests[goal].push_back(C[index]);
                vertexSendRequests[goal].push_back(index);
            }
            
            index = rank*vertex_count + i + n;
            goal = (C[index] - (C[index] < n ? 0 : n)) / vertex_count;
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

                    if(vertexSendResponses[i][j/2] && vertexSendResponses[i][j/2] != -1)
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
                    {
                        M[vertexSendRequests[i][2*j]] = 2*n;
                    }
                }
            }
        }
        //ending of processing responses--------------------------------------------

        //*************************THIS SHOULD BE THE END OF THE ALGORITHM*************************
	}

    finishedProcesses.clear();
    if(rank == 0 && !TESTING) std::cout << "Finished." << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    if(SHOW_MATRICES_AND_RESULTS) displayMatrix(M, 1, M.size(), false, "Rank " + std::to_string(rank) + ":");

}

//void FindBestMatch(std::unordered_set<int>& D, std::vector<int>& C, std::vector<int>& M, int * matrix, int * transpose, int n, int vertex_count, int rank)
void FindBestMatch(std::unordered_set<int>& D, std::vector<int>& C, std::vector<int>& M, std::vector<int>& matrix, std::vector<int>& transpose, int n, int vertex_count, int rank)
{
        for(int i = 0; i < vertex_count; i++)
        {
            int index = rank*vertex_count + i;
            if(M[index] == -1)
            {
                C[index] = MaxIndex(transpose, i*n, (i+1)*n, M, n) % n + n;
                if(C[C[index]] == index)
                {
                    D.insert(index);
                    D.insert(C[index]);
                    M[C[index]] = index;
                    M[index] = C[index];
                    //std::cout << "0 " << rank*vertex_count + i + n <<  " " << C[rank*vertex_count + i + n] << " " << std::endl;
                }
            }
        }

        for(int i = 0; i < vertex_count;  i++)
        {
            int index = rank*vertex_count + i + n;
            if(M[index] == -1)
            {
                C[index] = MaxIndex(matrix, i*n, (i+1)*n, M, 0) % n; //rows look for matches in the columns
                //std::cout << rank*vertex_count + i + n << " " << C[rank*vertex_count + i + n] << std::endl;
                if(C[C[index]] == index)
                {
                    D.insert(index);
                    D.insert(C[index]);
                    M[C[index]] = index;
                    M[index] = C[index];
                    //std::cout << "0 " << rank*vertex_count + i + n <<  " " << C[rank*vertex_count + i + n] << " " << std::endl;
                }
            }
        }
}

//void ProcessNeighborsOfMatched(std::unordered_set<int>& D, std::vector<int>& C, std::vector<int>& M, int * matrix, int * transpose, int vertex_count, int rank, int n, int p)
void ProcessNeighborsOfMatched(std::unordered_set<int>& D, std::vector<int>& C, std::vector<int>& M, std::vector<int>& matrix, std::vector<int>& transpose, int vertex_count, int rank, int n, int p)
{
    while(!D.empty())
    {
        //int * searchMatrix;
        std::vector<int> searchMatrix;
        int v, index, shift;
        v = *(D.begin());
        D.erase(v);

        if(v < n) { shift = n; searchMatrix = transpose; }
        else      { shift = 0; searchMatrix = matrix; }
    
        for(int x = 0; x < vertex_count; x++)
        {
            index = rank*vertex_count + x + shift;
            if(M[index] == -1)
            {
                C[index] = MaxIndex(searchMatrix, x*n, (x+1)*n, M, n - shift) % n + (n - shift);
                if(C[C[index]] == index)
                {
                    D.insert(index);
                    D.insert(C[index]);
                    M[index] = C[index];
                    M[C[index]] = index;
                    //std::cout << index <<  " " << C[index] << std::endl;
                }
            }
        }
    }
}

//int MaxIndex(int * matrix, int start, int end, std::vector<int>& Edges, int shift)
int MaxIndex(std::vector<int>& matrix, int start, int end, std::vector<int>& Edges, int shift)
{
    int index, maxIndex;
    maxIndex = start;
    for(int i = start; i < end; i++)
        if(Edges[i - start + shift] == -1)
        {
            index = maxIndex = i;
            break;
        }
    
    for(int i = index; i < end; i++)
        if(matrix[i] > matrix[maxIndex] && Edges[i - start + shift] == -1)
            maxIndex = i;
    
    return maxIndex;
}

//void createMatrices(int * matrix, int * transpose, int n)
void createMatrices(std::vector<int>& matrix, std::vector<int>& transpose, int n)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    int min, max;

    min = std::numeric_limits<int>::min();
    max = std::numeric_limits<int>::max();
    if(SHOW_MATRICES_AND_RESULTS)
    {
        min = -9;
        max = 9;
    }
    
    std::uniform_int_distribution<int> dist(min, max);
    for (unsigned i = 0; i < n; i++)
        for(unsigned j = 0; j < n; j++)
        {
            int val = dist(gen);
            matrix[i*n + j] = val;
            transpose[j*n + i] = val;
        }
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