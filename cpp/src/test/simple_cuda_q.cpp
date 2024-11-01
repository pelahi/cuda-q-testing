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
  bool iverbose = false;
  /// @brief  noise parameters
  //@{
  struct noise_prob {
  float dep = 1.0;
  float bf = 0.5;
  float pf = 0.5;
  } noise_probs;

  //@} 
};

///Outputs the usage to stdout
void usage(void)
{
    Options opt;
    std::cerr<<"USAGE:\n";
    std::cerr<<"\n";
    std::cerr<<"-n number of qubits ("<<opt.nqubits<<")"<<std::endl;
    std::cerr<<"-s number of qubits ("<<opt.nshots<<")"<<std::endl;
    std::cerr<<"-d number of qubits ("<<opt.noise_probs.dep<<")"<<std::endl;
    std::cerr<<"-b number of qubits ("<<opt.noise_probs.bf<<")"<<std::endl;
    std::cerr<<"-p number of qubits ("<<opt.noise_probs.pf<<")"<<std::endl;
    std::cerr<<"-v verbose ("<<opt.iverbose<<")"<<std::endl;
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
    while ((option = getopt(argc, argv, ":n:s:d:b:p:v:")) != EOF)
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
            case 'd':
                opt.noise_probs.dep = atof(optarg);
                NumArgs += 2;
                break;
            case 'b':
                opt.noise_probs.bf = atof(optarg);
                NumArgs += 2;
                break;
            case 'p':
                opt.noise_probs.pf = atof(optarg);
                NumArgs += 2;
                break;
            case 'v':
                opt.iverbose = atoi(optarg);
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
    if (ThisTask == 0 and opt.iverbose) Log() << cudaq::draw(kernel, opt.nqubits) <<std::endl;

    {
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
            Log()<<"Observed (ideal): "<<bits.data() <<" "<<count<<std::endl;
        }
    }
    }

    // Now with noise and using a lambda qpu kernel

    // We will begin by defining an empty noise model that we will add
    // our depolarization channel to.
    cudaq::noise_model noise;

    // Depolarization channel with a specific probability of the qubit state
    // being scrambled.
    cudaq::depolarization_channel depolarization(opt.noise_probs.dep);
    // and bit flip channel
    cudaq::bit_flip_channel bf(opt.noise_probs.bf);
    // Phase flip channel 
    cudaq::phase_flip_channel pf(opt.noise_probs.pf);

    // We will apply the channel to any X-gate on qubit 0. Meaning,
    // for each X-gate on our qubit, the qubit will have a `1.0`
    // probability of decaying into a mixed state.
    noise.add_channel<cudaq::types::x>({0}, depolarization);
    // We will apply this channel to any X gate on the qubit, giving each X-gate
    // a probability of `1.0` of undergoing an extra X-gate.
    noise.add_channel<cudaq::types::x>({0}, bf);
    // We will apply this channel to any Z gate on the qubit.
    // Meaning, after each Z gate on qubit 0, there will be a
    // probability of `1.0` that the qubit undergoes an extra
    // Z rotation.
    noise.add_channel<cudaq::types::z>({0}, pf);    

   // Our ghz kernel
    auto ghzlambda = [](const size_t N) __qpu__ {
        cudaq::qvector q(N);
        // cudaq::h(q[0]);
        // for (int i = 0; i < N - 1; i++) {
        //     cudaq::x<cudaq::ctrl>(q[i], q[i + 1]);
        //     cudaq::z(q[i]);
        // }
        cudaq::x(q);
        cudaq::mz(q);
    };

    // Now let's set the noise and we're ready to run the simulation!
    {
    auto t1 = NewTimerHostOnly();
    auto s1 = NewComputeSampler(0.01);
    cudaq::set_noise(noise);
    auto noisy_counts = cudaq::sample(opt.nshots, ghzlambda, opt.nqubits);
    if (!cudaq::mpi::is_initialized() || cudaq::mpi::rank() == 0) {
        noisy_counts.dump();
    }
    cudaq::unset_noise();
    LogTimeTaken(t1);
    LogCPUUsage(s1);
    LogGPUStatistics(s1);

    if (!cudaq::mpi::is_initialized() || cudaq::mpi::rank() == 0) {
        // Fine grain access to the bits and counts
        for (auto &[bits, count] : noisy_counts) {
            Log()<<"Observed (noisy dep="<<opt.noise_probs.dep<<", bf="<<opt.noise_probs.bf<<", pf="<<opt.noise_probs.pf<<" ): "<<bits.data() <<" "<<count<<std::endl;
        }
    }
    }

    cudaq::mpi::finalize();
    return 0;
}

