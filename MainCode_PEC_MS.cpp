#include "Core/SimulationContext.h"
#include "Core/Preprocess.h"
#include "Core/MatrixAssembly.h"
#include "Core/Solver.h"
#include "Core/ComputeRCS.h"
#include "Utility/Timing.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <metis.h>
#include <mpi.h>
#include <windows.h>
#include <psapi.h>

using namespace std;
using namespace Timing;
auto EMsim = std::make_unique<SimulationContext>();

template <typename Derived>
void writeMatrixToFile(const std::string& filename,
    const Eigen::MatrixBase<Derived>& M,
    int precision = 6,
    bool scientific = false)
{
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Error: cannot open file " << filename << std::endl;
        return;
    }

    if (scientific) {
        out.setf(std::ios::scientific);
    }
    else {
        out.setf(std::ios::fixed);
    }
    out.precision(precision);

    for (int i = 0; i < M.rows(); ++i) {
        for (int j = 0; j < M.cols(); ++j) {
            out << M(i, j);
            if (j + 1 < M.cols()) out << ' ';
        }
        out << '\n';
    }
    out.close();
}

void SaveIntToFile(int value, const std::string& filename)
{
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "Error: cannot open file " << filename << " for writing.\n";
        return;
    }

    ofs << value;
}

int main(int argc, char* argv[])
{
    //---------------------------------------------------------------
    // MPI Initialization
    //---------------------------------------------------------------
    MPI_Init(&argc, &argv);
    EMsim->mpi.comm = MPI_COMM_WORLD;
    MPI_Comm_rank(EMsim->mpi.comm, &EMsim->mpi.rank);
    MPI_Comm_size(EMsim->mpi.comm, &EMsim->mpi.size);

    //---------------------------------------------------------------
    // Main Code
    //---------------------------------------------------------------
    auto start_TOTAL = std::chrono::steady_clock::now();
    int check = 0;

    // Fixed Code ---------------------------------------------------
    Seconds duration_STEP1 = measure([&] {
        InitSimContext(argc, argv, *EMsim);
        });
    if (EMsim->mpi.rank == 0) {
        ++check;
        SaveIntToFile(check, "check.txt");
    }
    Seconds duration_STEP2 = measure([&] {
        Preprocess::RunFixed(EMsim->pre, EMsim->phys, EMsim->mesh, EMsim->mpi);
        MatAssembly::RunFixed(EMsim->mat, EMsim->pre, EMsim->phys);
        if (EMsim->mpi.rank == 0) {
            ++check;
            SaveIntToFile(check, "check.txt");
        }
        FETIsolver::RunFixed(EMsim->solver, EMsim->mat, EMsim->pre);
        RCS::RunFixed(EMsim->rcs, *EMsim, EMsim->mesh.pec2nodes);
        if (EMsim->mpi.rank == 0) {
            ++check;
            SaveIntToFile(check, "check.txt");
        }
        });
    
    
    // Sweep Code ---------------------------------------------------    
    int nAngle = EMsim->config.nAngle;
    double EoPhi = EMsim->config.EoPhi;
    double EoTheta = EMsim->config.EoTheta;
    if (EMsim->mpi.rank == 0) {
        cout << "\n\n ===================================================================" << std::flush;
        cout << "\n Solving... (Monostatic sweep, " << nAngle << " angles)"              << std::flush;
        cout << "\n ===================================================================" << std::flush;
    }
    
    Timing::Seconds duration_STEP3{ 0.0 };
    Timing::Seconds duration_STEP4{ 0.0 };
    Eigen::VectorXd RCS_dBsm(nAngle); RCS_dBsm.setZero();
    for (int j = 0; j < nAngle; j++) {

        double phi = EMsim->config.SWavePhi(j);
        double theta = EMsim->config.SWaveTheta(j);

        // --- STEP3: Preprocess + MatAssembly + Solver ---
        duration_STEP3 += Timing::measure([&] {
            Preprocess::RunSweep(EMsim->pre, EMsim->mesh, EMsim->phys, phi, theta, EoPhi, EoTheta);
            MatAssembly::RunSweep(EMsim->mat, EMsim->pre);
            FETIsolver::RunSweep(EMsim->solver, EMsim->mpi, EMsim->mat, EMsim->pre, EMsim->config);
            });

        // --- STEP4: RCS ---
        duration_STEP4 += Timing::measure([&] {
            RCS::RunSweep(EMsim->rcs, *EMsim, EMsim->mesh.pec2nodes, phi, theta, EoPhi, EoTheta);
            });

        if (EMsim->mpi.rank == 0) {
            RCS_dBsm(j) = EMsim->rcs.RCS_dBsm;

            constexpr int LABEL_WIDTH = 9;
            std::cout << "\n step no." << j << "   converged at iteration "
                << EMsim->solver.DualIter + 1 << "." << std::flush;
        }
    };

    if (EMsim->mpi.rank == 0) {
        std::cout << "\n  calculation finished." << std::flush;
        ++check;
        SaveIntToFile(check, "check.txt");
    }
    if (EMsim->mpi.rank == 0) {
        ++check;
        SaveIntToFile(check, "check.txt");        
    }
    
    //---------------------------------------------------------------
    auto end_TOTAL = std::chrono::steady_clock::now();
    Seconds duration_TOTAL = end_TOTAL - start_TOTAL;

    //---------------------------------------------------------------
    // MEMORY MONITOR 
    //---------------------------------------------------------------
    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));

    size_t N = 1024 * 1024 * 1024;
    SIZE_T peakMemory = pmc.PeakWorkingSetSize;
    SIZE_T currentMemory = pmc.WorkingSetSize;
    double peakMemoryGB = static_cast<double>(peakMemory) / N;
    double currentMemoryGB = static_cast<double>(currentMemory) / N;

    double totalPeakGB = 0.0;
    double totalCurrentGB = 0.0;
    MPI_Reduce(&peakMemoryGB, &totalPeakGB, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&currentMemoryGB, &totalCurrentGB, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double maxPeakGB = 0.0;
    double maxCurrentGB = 0.0;
    MPI_Reduce(&peakMemoryGB, &maxPeakGB, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&currentMemoryGB, &maxCurrentGB, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    auto sys_now = std::chrono::system_clock::now();
    time_t now_c = std::chrono::system_clock::to_time_t(sys_now);
    tm local_tm; localtime_s(&local_tm, &now_c);
    //---------------------------------------------------------------

    if (EMsim->mpi.rank == 0) {

        constexpr int LABEL_WIDTH = 24;

        std::cout << "\n\n ===================================================================" << std::flush;
        std::cout << "\n          Simulation Summary             " << std::flush;
        std::cout << "\n ===================================================================" << std::flush;
        std::cout << "\n Execution complete date/time" << " :  " << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S") << std::flush;
        std::cout << std::fixed << std::setprecision(3) << std::flush;
        std::cout << "\n * " << std::left << std::setw(LABEL_WIDTH) << "Input / Data Load" << ":  " << duration_STEP1.count() << " s" << std::flush;
        std::cout << "\n * " << std::left << std::setw(LABEL_WIDTH) << "Pre-processing"    << ":  " << duration_STEP2.count() << " s" << std::flush;
        std::cout << "\n * " << std::left << std::setw(LABEL_WIDTH) << "Solver"            << ":  " << duration_STEP3.count() << " s" << std::flush;
        std::cout << "\n * " << std::left << std::setw(LABEL_WIDTH) << "Post-processing"   << ":  " << duration_STEP4.count() << " s" << std::flush;
        std::cout << "\n * " << std::left << std::setw(LABEL_WIDTH) << "Total run time"    << ":  " << duration_TOTAL.count() << " s" << std::endl;
        std::cout << std::fixed << std::setprecision(2) << std::flush;
        std::cout << "\n " << std::left << std::setw(LABEL_WIDTH+2) << "Peak Memory Usage"    << ":  " << totalPeakGB    << " GB" << std::flush;
        std::cout << "\n " << std::left << std::setw(LABEL_WIDTH+2) << "Average Memory Usage" << ":  " << totalCurrentGB << " GB" << std::flush;
        std::cout << "\n " << std::left << std::setw(LABEL_WIDTH+2) << "Max Peak per rank"    << ":  " << maxPeakGB      << " GB" << std::flush;
        std::cout << "\n " << std::left << std::setw(LABEL_WIDTH+2) << "Max Current per rank" << ":  " << maxCurrentGB   << " GB\n" << std::endl;
        
        // RCS write
        std::string RCS_File = "rcs.txt";
        std::ofstream RCS_out(RCS_File);
        RCS_out.setf(std::ios::scientific);
        RCS_out.precision(8);

        const int nn = RCS_dBsm.rows();
        for (int i = 0; i < nn; ++i) {
            double val = RCS_dBsm(i);
            RCS_out << val << '\n';
        }
        RCS_out.close();

        writeMatrixToFile("node_coord.txt", EMsim->mesh.nodes2coord, 6, false);
        writeMatrixToFile("obj_face.txt", EMsim->mesh.pec2nodes, 0, false);

    }

    MPI_Finalize();
    return 0;
}


//Eigen::VectorXd RCS_dBsm(nAngle); RCS_dBsm.setZero();
//Seconds duration_STEP3 = measure([&] {

//    for (int j = 0; j < nAngle; j++) {

//        double phi = EMsim->config.SWavePhi(j);
//        double theta = EMsim->config.SWaveTheta(j);

//        Preprocess::RunSweep(EMsim->pre, EMsim->mesh, EMsim->phys, phi, theta, EoPhi, EoTheta);
//        MatAssembly::RunSweep(EMsim->mat, EMsim->pre);
//        FETIsolver::RunSweep(EMsim->solver, EMsim->mpi, EMsim->mat, EMsim->pre);
//        RCS::RunSweep(EMsim->rcs, *EMsim, EMsim->mesh.pec2nodes, phi, theta, EoPhi, EoTheta);
//        if (EMsim->mpi.rank == 0) {
//            RCS_dBsm(j) = EMsim->rcs.RCS_dBsm;
//        
//            constexpr int LABEL_WIDTH = 9;
//            std::cout << "\n step no." << j << "   converged at iteration "
//                << EMsim->solver.DualIter + 1 << "." << std::flush;
//            /*if (j == 0) {
//                std::string solution_File = EMsim->config.BasePath + "solution.txt";
//                std::ofstream solution_out(solution_File);
//                solution_out.setf(std::ios::scientific);
//                solution_out.precision(8);

//                const int n = EMsim->solver.sol.rows();
//                for (int i = 0; i < n; ++i) {
//                    double re = EMsim->solver.sol(i).real();
//                    double im = EMsim->solver.sol(i).imag();

//                    if (std::abs(re) < 1e-15) re = 0.0;
//                    if (std::abs(im) < 1e-15) im = 0.0;

//                    solution_out << re;
//                    if (im >= 0.0)
//                        solution_out << "+" << im << "i";
//                    else
//                        solution_out << im << "i";
//                    solution_out << '\n';
//                }
//                solution_out.close();
//            }*/
//        }            
//    }});
