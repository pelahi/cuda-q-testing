// Compile and run with:
// 
// nvq++ -D_CUDA -D_MPI -L/opt/nvidia/hpc_sdk/Linux_aarch64/24.5/cuda/12.4/targets/sbsa-linux/lib/ -I/opt/nvidia/hpc_sdk/Linux_aarch64/24.5/cuda/12.4/targets/sbsa-linux/include/ -I../../../profile_util/include/ -L../../../profile_util/build/src/ simple_cuda_q.cpp -o ghz -lprofile_util -lmpi -lcudart
// 

#include <iostream>
#include <vector>
#include <algorithm>
#include <profile_util.h>
// include cudaq.h
#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <cudaq/algorithms/draw.h>


/// Define a CUDA Quantum kernel that is fully specified
/// at compile time via templates. __qpu__ indicates a quantum kernel 
struct ghz {
    auto operator()(const size_t N) __qpu__ {
        cudaq::qvector q(N);
        // why don't I need the namespace?
        cudaq::h(q[0]);
        for (int i = 0; i < N - 1; i++) {
            cudaq::x<cudaq::ctrl>(q[i], q[i + 1]);
        }
        cudaq::mz(q);
    }
};

struct Options {
  int nqubits = 1;
  int nshots = 1;
};

///Outputs the usage to stdout
void usage(void)
{
    Options opt;
    std::cerr<<"USAGE:\n";
    std::cerr<<"\n";
    std::cerr<<"-n number of qubits ("<<opt.nqubits<<")"<<std::endl;
    std::cerr<<"-s number of qubits ("<<opt.nshots<<")"<<std::endl;
#ifdef _MPI
    cudaq::mpi::finalize();
#endif
    exit(1);
}


///routine to get arguments from command line
void GetArgs(int argc, char *argv[], Options &opt)
{
    int option;
    int NumArgs = 0;
    while ((option = getopt(argc, argv, ":n:s:")) != EOF)
    {
        switch(option)
        {
            case 'n':
                opt.nqubits = atoi(optarg);
                NumArgs += 2;
                break;
            case 's':
                opt.nshots = atoi(optarg);
                NumArgs += 2;
                break;
            case '?':
                usage();
        }
    }
}

int main(int argc,char **argv) {

    // cudaq makes use of mpi plugin object
    // to initialize mpi, get ranks, etc. 
    // appears at first glance no sub communicator
    // construction
    int ThisTask = 0;
#ifdef _MPI
    auto comm = MPI_COMM_WORLD;
    cudaq::mpi::initialize(argc, argv);
    // MPI_Init(&argc, &argv);
    MPISetLoggingComm(comm);
    MPI_Comm_rank(comm, &ThisTask);
#endif 
#ifdef _MPI
    MPILog0Version();
    MPILog0ParallelAPI();
    MPILog0Binding();
    MPILog0NodeSystemMem();
    MPI_Barrier(comm);
#else 
    LogVersion();
    LogParallelAPI();
    LogBinding();
    LogSystemMem();
#endif

    Options opt;
    GetArgs(argc, argv, opt);
    // set the kernel to a n qubit simulation
    // why does the number of qubits need to be set at compile time?
    auto kernel = ghz{};
    // lets output the circuit
    if (ThisTask == 0) Log() << cudaq::draw(kernel, opt.nqubits) <<std::endl;

    auto t1 = NewTimerHostOnly();
    auto s1 = NewComputeSampler(0.01);
    auto counts = cudaq::sample(opt.nshots, kernel, opt.nqubits);
    // I find it odd that this dump is only done by rank 0 but let's see
    // what happens. Maybe counts.dump has an allgather?
    if (!cudaq::mpi::is_initialized() || cudaq::mpi::rank() == 0) {
        counts.dump();
    }
    LogTimeTaken(t1);
    LogCPUUsage(s1);
    LogGPUStatistics(s1);

    if (!cudaq::mpi::is_initialized() || cudaq::mpi::rank() == 0) {
        // Fine grain access to the bits and counts
        for (auto &[bits, count] : counts) {
            Log()<<"Observed: "<<bits.data() <<" "<<count<<std::endl;
        }
    }
    cudaq::mpi::finalize();
    return 0;
}

