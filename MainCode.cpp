#include "Core/SimulationContext.h"
#include "Core/Preprocess.h"
#include "Core/MatrixAssembly.h"
#include "Core/Solver.h"
#include "Core/ComputeRCS.h"
#include "Post/ComputeElectricField.h"
#include "Utility/Timing.h"
#include "Post/WriteVTK.h"

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

void SaveIntToFile(int value, const std::string& filename)
{
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "Error: cannot open file " << filename << " for writing.\n";
        return;
    }

    ofs << value;
}

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
    // Main Code (FETI-DP + 2ndOrder Nedelec + PEC object)
    //---------------------------------------------------------------
    auto start_TOTAL = std::chrono::steady_clock::now();
    int check = 0;

    // STEP 1 
    Seconds duration_STEP1 = measure([&] {
        InitSimContext(argc, argv, *EMsim);
        });
    if (EMsim->mpi.rank == 0) {
        ++check;
        SaveIntToFile(check, "check.txt");
    }
    // STEP 2 
    Seconds duration_STEP2 = measure([&] {
        Preprocess::RunAll(EMsim->pre, EMsim->phys,
            EMsim->mesh, EMsim->config, EMsim->mpi);
        });
    if (EMsim->mpi.rank == 0) {
        ++check;
        SaveIntToFile(check, "check.txt");
    }    
    // STEP 3 
    Seconds duration_STEP3 = measure([&] {
        MatAssembly::RunAll(EMsim->mat, EMsim->pre, EMsim->phys);
        });
    if (EMsim->mpi.rank == 0) {
        ++check;
        SaveIntToFile(check, "check.txt");
    }
    // STEP 4 
    Seconds duration_STEP4 = measure([&] {
        FETIsolver::RunAll(EMsim->solver, EMsim->mpi,
            EMsim->mat, EMsim->pre, EMsim->config);
        });
    if (EMsim->mpi.rank == 0) {
        ++check;
        SaveIntToFile(check, "check.txt");
    }
    // STEP 5 
    Seconds duration_STEP5 = measure([&] {
        RCS::RunAll(*EMsim, EMsim->mesh.pec2nodes);
        });
    if (EMsim->mpi.rank == 0) {
        ++check;
        SaveIntToFile(check, "check.txt");
    }
    // STEP 6
    if (EMsim->mpi.rank == 0) {
        //std::vector<ElementEAvg> elemE = ComputeElementEfield(*EMsim);
        //WriteEfieldVTK_CellData("Output_E_cell.vtk", *EMsim, elemE);

        std::vector<NodeEAvg> nodeE = ComputeNodeEfield(*EMsim);
        WriteEfieldVTK_PointData("plot_field.vtk", *EMsim, nodeE);
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
    SIZE_T peakMemory    = pmc.PeakWorkingSetSize;
    SIZE_T currentMemory = pmc.WorkingSetSize;
    double peakMemoryGB    = static_cast<double>(peakMemory) / N;
    double currentMemoryGB = static_cast<double>(currentMemory) / N;

    double totalPeakGB    = 0.0;
    double totalCurrentGB = 0.0;
    MPI_Reduce(&peakMemoryGB, &totalPeakGB, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&currentMemoryGB, &totalCurrentGB, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double maxPeakGB    = 0.0;
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
        std::cout << "\n Execution complete date/time" << " : " << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S") << std::flush;
        std::cout << std::fixed << std::setprecision(3) << std::flush;
        std::cout << "\n * " << std::left << std::setw(LABEL_WIDTH) << "Input / Data Load" << ":  " << duration_STEP1.count() << " s" << std::flush;
        std::cout << "\n * " << std::left << std::setw(LABEL_WIDTH) << "Pre-processing"    << ":  " << duration_STEP2.count() << " s" << std::flush;
        std::cout << "\n * " << std::left << std::setw(LABEL_WIDTH) << "Solver"            << ":  " << duration_STEP4.count() << " s" << std::flush;
        std::cout << "\n * " << std::left << std::setw(LABEL_WIDTH) << "Post-processing"   << ":  " << duration_STEP5.count() << " s" << std::flush;
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

        const int nn = EMsim->rcs.RCS_dBsw.rows();
        for (int i = 0; i < nn; ++i) {
            double val = EMsim->rcs.RCS_dBsw(i);
            RCS_out << val << '\n';
        }
        RCS_out.close();

        writeMatrixToFile("node_coord.txt", EMsim->mesh.nodes2coord, 6, false);
        writeMatrixToFile("obj_face.txt", EMsim->mesh.pec2nodes, 0, false);

    }

    MPI_Finalize();
    return 0;
}
