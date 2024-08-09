#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <metis.h>
#include <vector>
#include "file_reader.h"
#include "generate_signs_matrix.h"
#include "generate_edges_or_faces.h"
#include "generate_boundary_faces.h"
#include "generate_boundary_edges.h"
#include "generate_boundary_normal.h"
#include "generate_boundary_faces_index.h"
#include "get_gaussian_quadrature.h"
#include "get_gaussian_quadrature_face.h"
#include "get_Nedelec0_basis_functions.h"

#include "zmumps_c.h"
#include "ZMumps_solver_Krr.h"
#include "ZMumps_solver_Kcc.h"
#include "BiCGSTAB_Dual_solver.h"

#include "igl/sort.h"
#include "igl/unique_rows.h"
#include "igl/slice.h"
#include "igl/find.h"
#include "igl/ismember_rows.h"
#include "igl/intersect.h"
#include "igl/setdiff.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SVD>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/PardisoSupport>
#include <time.h>

#include "igl/unique.h"
#include "igl/ismember.h"
#include "igl/setdiff.h"
#include "igl/cat.h"
#include "igl/cross.h"
#include "igl/sort.h"

#define PI 3.141592653589793
#define JOB_END -2;

using namespace std;
typedef complex<double> cdouble;
typedef Eigen::Triplet<double> S_T, M_T;
typedef Eigen::Triplet<cdouble> ABC_T, IBC_T, SOTC_T;
using VectorXi = Eigen::VectorXi;
using VectorXd = Eigen::VectorXd;
using VectorXcd = Eigen::VectorXcd;
using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXcd = Eigen::Matrix<cdouble, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;


// Define dimensions and the number of vertices
int dim = 3, nverts = 4, nbasis = 6;
cdouble i(0.0, 1.0); // Unit imaginary;

// Input properties
cdouble mu = 1.0 + 0.0 * i;
cdouble eps = 1.0 + 0.0 * i;
cdouble mu_d = 1.0 + 0.0 * i;
cdouble eps_d = 2.5 - 0.5 * i;

double f = 0.3 * pow(10, 9);
double c0 = 3 * pow(10, 8);
double lambda = c0 / f;
double k0 = 2 * PI / lambda;
double tau = 0.01 * lambda;
cdouble eta = i * sqrt(mu_d / eps_d) * tan(k0 * sqrt(mu_d * eps_d) * tau);



// Define the order matrices for edges and faces
MatrixXi orderEdges(nbasis, dim - 1);
MatrixXi orderFaces(nverts, dim);

int main(int argc, char* argv[]) {

    //----------------------------------------------------------------------------------------------------
    // Read mesh information from files (Global)
    //----------------------------------------------------------------------------------------------------
    string intFilename("elems2nodes.txt");
    string doubleFilename("nodes2coord.txt");
    MatrixXi elems2nodes = readIntDataFromFile(intFilename, nverts);
    MatrixXd nodes2coord = readDoubleDataFromFile(doubleFilename, dim);

    string intFilename1("pec2elems.txt");
    string intFilename2("pec2nodes.txt");
    string intFilename3("abc2elems.txt");
    string intFilename4("abc2nodes.txt");
    MatrixXi obj2elems = readIntDataFromFile(intFilename1, 1);
    MatrixXi obj2nodes = readIntDataFromFile(intFilename2, dim);
    MatrixXi abc2elems = readIntDataFromFile(intFilename3, 1);
    MatrixXi abc2nodes = readIntDataFromFile(intFilename4, dim);

    /*string intFilename5("DD.txt");
    MatrixXi DD = readIntDataFromFile(intFilename5, 1);*/
    //----------------------------------------------------------------------------------------------------
    clock_t start, finish;
    double duration;
    start = clock();

    // Edge numbering process
    orderEdges << 0, 1, 0, 2, 0, 3, 1, 2, 2, 3, 3, 1;
    auto edgesResult = generateEdgesOrFaces(elems2nodes, orderEdges);
    MatrixXi elems2edges = edgesResult.first;
    MatrixXi edges2nodes = edgesResult.second;
    MatrixXi signs2edges = generateSignsMatrix(elems2nodes, orderEdges);

    orderFaces << 0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3;
    auto facesResult = generateEdgesOrFaces(elems2nodes, orderFaces);
    MatrixXi elems2faces = facesResult.first;
    MatrixXi faces2nodes = facesResult.second;

    std::pair<MatrixXi, MatrixXi> BF_result = generateBoundaryFacesToNodes(elems2faces, faces2nodes);
    MatrixXi bfaces2nodes = BF_result.first;
    MatrixXi bfaces2elems = BF_result.second;
    MatrixXi bedges2edges = generateBoundaryEdgesToEdges(bfaces2nodes, edges2nodes);

    // Get boundary edge & face information
    VectorXi obj2faceID, abc2faceID;
    obj2faceID = generateBoundaryFacesIndex(obj2elems, obj2nodes, elems2faces, faces2nodes);
    abc2faceID = generateBoundaryFacesIndex(abc2elems, abc2nodes, elems2faces, faces2nodes);

    // Get Gaussian quadarature points in 3D
    int p = 2;
    std::pair<MatrixXd, MatrixXd> GQ_result = GaussianQuadrature::getGaussianQuadrature(p, dim);
    MatrixXd ip = GQ_result.first;
    MatrixXd w = GQ_result.second;

    int nelems = elems2nodes.rows();
    int nnodes = nodes2coord.rows();
    int nedges = elems2edges.maxCoeff() + 1;
    int dofs = nedges;
    VectorXi globEdges = Eigen::VectorXi::LinSpaced(nedges, 0, nedges);

    VectorXd edges2length(nedges);
    for (size_t j = 0; j < dofs; j++) {
        int e1 = edges2nodes(j, 0);
        int e2 = edges2nodes(j, 1);
        edges2length(j) = (nodes2coord.row(e1) - nodes2coord.row(e2)).norm();
    }
    double h_min = edges2length.minCoeff();

    cdouble k_max = PI / h_min;
    cdouble mu_sq = (mu + mu) / 2.;
    cdouble eps_sq = (eps + eps) / 2.;

    cdouble alpha = i * k0 * sqrt(mu_sq * eps_sq);
    cdouble k_tilde = -i * sqrt(pow(k_max, 2) - pow(k0, 2) * eps_sq * mu_sq);
    cdouble beta = -i / (k0 * sqrt(mu_sq * eps_sq) + k_tilde);


    //----------------------------------------------------------------------------------------------------
    // Parallel computing with MPI
    //----------------------------------------------------------------------------------------------------
    MPI_Status status;
    MPI_Init(&argc, &argv);
    int comm_sz, my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    //----------------------------------------------------------------------------------------------------
    // METIS - Domain decomposition with nparts
    //----------------------------------------------------------------------------------------------------

    idx_t ne = nelems; // The number of elements in the mesh
    idx_t nn = nnodes;  // The number of nodes in the mesh
    idx_t nparts = comm_sz; // The number of parts to partition the graph
    idx_t ncon = 2; // The number of balnacing constraints. It should be at least 1
    idx_t numflag = 0;
    idx_t ncommon = 3;
    vector<idx_t> eptr(ne + 1);
    vector<idx_t> eind(elems2nodes.size());
    vector<idx_t> epart(ne);
    vector<idx_t> npart(nn);

    eptr[0] = 0;
    for (size_t j = 0; j < ne; ++j) {
        eptr[1 * j + 1] = nverts * (j + 1);
        for (size_t k = 0; k < nverts; k++) {
            eind[nverts * j + k] = elems2nodes(j, k);
        }
    }
    idx_t* xadj, * adjncy;
    idx_t objval;
    int status_metis = METIS_PartMeshNodal(&ne, &nn, eptr.data(), eind.data(), NULL, NULL, &nparts, NULL, NULL, &objval, epart.data(), npart.data());
    VectorXi DD(ne);
    for (size_t ik = 0; ik < ne; ik++) {
        DD(ik) = epart[ik];
    }

    if (my_rank == 0) {
        std::cout << "STEP 0. Domain decomposition process done\n" << endl;
    }
    //---------------------------------------
    // Divide subdomains
    //---------------------------------------
    VectorXi sub_idx;
    igl::find(DD.array() == my_rank, sub_idx);

    MatrixXi sub_elems2nodes(sub_idx.size(), nverts);
    for (size_t j = 0; j < sub_idx.size(); j++) {
        sub_elems2nodes.row(j) = elems2nodes.row(sub_idx(j));
    }
    MatrixXi sub_elems2faces(sub_idx.size(), nverts);
    for (size_t j = 0; j < sub_idx.size(); j++) {
        sub_elems2faces.row(j) = elems2faces.row(sub_idx(j));
    }
    MatrixXi sub_elems2signs(sub_idx.size(), nbasis);
    for (size_t j = 0; j < sub_idx.size(); j++) {
        sub_elems2signs.row(j) = signs2edges.row(sub_idx(j));
    }
    MatrixXi sub_elems2edges(sub_idx.size(), nbasis);
    for (size_t j = 0; j < sub_idx.size(); j++) {
        sub_elems2edges.row(j) = elems2edges.row(sub_idx(j));
    }

    VectorXi subEdges;
    igl::unique(sub_elems2edges.reshaped(sub_idx.size() * nbasis, 1), subEdges);

    VectorXi glob2sub, LOCB;
    igl::ismember_rows(globEdges, subEdges, glob2sub, LOCB);
    VectorXi check(nedges * comm_sz);
    MPI_Allgather(glob2sub.data(), nedges, MPI_INT, check.data(), nedges, MPI_INT, MPI_COMM_WORLD);
    MatrixXi check_mat = check.transpose().reshaped(nedges, comm_sz);
    VectorXi check_sum = check_mat.rowwise().sum();
    VectorXi ei, eb, ec;
    igl::find(check_sum.array() == 1, ei);
    igl::find(check_sum.array() == 2, eb);
    igl::find(check_sum.array() >= 3, ec);
    int ni = ei.size(), nb = eb.size(), nc = ec.size();

    VectorXi eidia, ebdia, ecdia, LOCBI, LOCBB, LOCBC;
    igl::ismember_rows(subEdges, ei, eidia, LOCBI);
    igl::ismember_rows(subEdges, eb, ebdia, LOCBB);
    igl::ismember_rows(subEdges, ec, ecdia, LOCBC);

    VectorXi sub_ei, sub_eb, sub_ec, sub_ri, sub_rb, sub_rc;
    igl::find(eidia.array() == 1, sub_ri); igl::slice(subEdges, sub_ri, 1, sub_ei);
    igl::find(ebdia.array() == 1, sub_rb); igl::slice(subEdges, sub_rb, 1, sub_eb);
    igl::find(ecdia.array() == 1, sub_rc); igl::slice(subEdges, sub_rc, 1, sub_ec);
    int sub_ni = sub_ei.size(), sub_nb = sub_eb.size(), sub_nc = sub_ec.size();
    int sub_nr = sub_ni + sub_nb;

    VectorXi sub_edgeOrder(sub_ni + sub_nb + sub_nc); // Concatenate ei, eb, ec 
    sub_edgeOrder << sub_ei, sub_eb, sub_ec;
    int sub_dofs = sub_edgeOrder.size();

    MatrixXi gli, sub_elems2edges_order_vec;
    igl::ismember_rows(sub_elems2edges.transpose().reshaped(sub_idx.size() * nbasis, 1), sub_edgeOrder, gli, sub_elems2edges_order_vec);
    MatrixXi sub_elems2edges_order = sub_elems2edges_order_vec.reshaped(nbasis, sub_idx.size()).transpose();

    int sub_nbc = sub_nb + sub_nc;
    VectorXi sub_ebc(sub_nbc); sub_ebc << sub_eb, sub_ec;
    MatrixXi sub_check(sub_nbc, comm_sz);
    for (size_t j = 0; j < sub_nbc; j++) {
        sub_check.row(j) = check_mat.row(sub_ebc(j));
    }
    MatrixXi sub_check_sum(1, comm_sz);
    sub_check_sum = sub_check.colwise().sum();

    VectorXi rI, cI, vv, aI;
    igl::find(sub_check_sum, rI, cI, vv);
    remove(cI.begin(), cI.end(), my_rank);
    VectorXi sub_dom(cI.size() - 1);
    sub_dom = cI.head(cI.size() - 1);

    std::cout << "[MY RANK " << my_rank << "] >> DOFs = " << sub_dofs << ", sub_ni = " << sub_ni << ", sub_nb = " << sub_nb << ", sub_nr = " << sub_nr << ", sub_nc = " << sub_nc << "\n" << endl;
    // ---------------------------------------
    // Boundary reordering
    // ---------------------------------------
    VectorXi obj_idx, obj_logic, LOCBP;
    igl::ismember_rows(obj2elems, sub_idx, obj_logic, LOCBP);
    igl::find(obj_logic.array() == 1, obj_idx);

    VectorXi sub_obj2elems(obj_idx.size());
    MatrixXi sub_obj2Fnodes(obj_idx.size(), dim);
    for (size_t j = 0; j < obj_idx.size(); j++) {
        sub_obj2elems(j) = obj2elems(obj_idx(j));
        sub_obj2Fnodes.row(j) = obj2nodes.row(obj_idx(j));
    }
    MatrixXi sub_obj2nodes(obj_idx.size(), nverts);
    MatrixXi sub_obj2faces(obj_idx.size(), nverts);
    MatrixXi sub_obj2edges(obj_idx.size(), nbasis);
    MatrixXi sub_obj2signs(obj_idx.size(), nbasis);
    for (size_t j = 0; j < obj_idx.size(); j++) {
        sub_obj2nodes.row(j) = elems2nodes.row(sub_obj2elems(j));
        sub_obj2faces.row(j) = elems2faces.row(sub_obj2elems(j));
        sub_obj2edges.row(j) = elems2edges.row(sub_obj2elems(j));
        sub_obj2signs.row(j) = signs2edges.row(sub_obj2elems(j));
    }
    VectorXi sub_obj2faceID(obj_idx.size());
    for (size_t j = 0; j < obj_idx.size(); j++) {
        sub_obj2faceID(j) = obj2faceID(obj_idx(j));
    }
    MatrixXi sub_obj2faceInf(obj_idx.size(), 2);
    for (size_t j = 0; j < obj_idx.size(); j++) {
        int n = sub_obj2faceID(j);
        sub_obj2faceInf(j, 0) = sub_obj2elems(j);
        sub_obj2faceInf(j, 1) = sub_obj2faces(j, n);
    }
    VectorXi subEdgesOBJ;
    igl::unique(sub_obj2edges.reshaped(obj_idx.size() * nbasis, 1), subEdgesOBJ);
    MatrixXi gli_obj, sub_obj2edges_order_vec;
    igl::ismember_rows(sub_obj2edges.transpose().reshaped(obj_idx.size() * nbasis, 1), sub_edgeOrder, gli_obj, sub_obj2edges_order_vec);
    MatrixXi sub_obj2edges_order = sub_obj2edges_order_vec.reshaped(nbasis, obj_idx.size()).transpose();

    // ---------------------------------------
    VectorXi abc_idx, abc_logic, LOCBA;
    igl::ismember_rows(abc2elems, sub_idx, abc_logic, LOCBA);
    igl::find(abc_logic.array() == 1, abc_idx);

    VectorXi sub_abc2elems(abc_idx.size());
    MatrixXi sub_abc2Fnodes(abc_idx.size(), dim);
    for (size_t j = 0; j < abc_idx.size(); j++) {
        sub_abc2elems(j) = abc2elems(abc_idx(j));
        sub_abc2Fnodes.row(j) = abc2nodes.row(abc_idx(j));
    }
    MatrixXi sub_abc2nodes(abc_idx.size(), nverts);
    MatrixXi sub_abc2faces(abc_idx.size(), nverts);
    MatrixXi sub_abc2edges(abc_idx.size(), nbasis);
    MatrixXi sub_abc2signs(abc_idx.size(), nbasis);
    for (size_t j = 0; j < abc_idx.size(); j++) {
        sub_abc2nodes.row(j) = elems2nodes.row(sub_abc2elems(j));
        sub_abc2faces.row(j) = elems2faces.row(sub_abc2elems(j));
        sub_abc2edges.row(j) = elems2edges.row(sub_abc2elems(j));
        sub_abc2signs.row(j) = signs2edges.row(sub_abc2elems(j));
    }
    VectorXi sub_abc2faceID(abc_idx.size());
    for (size_t j = 0; j < abc_idx.size(); j++) {
        sub_abc2faceID(j) = abc2faceID(abc_idx(j));
    }
    MatrixXi sub_abc2faceInf(abc_idx.size(), 2);
    for (size_t j = 0; j < abc_idx.size(); j++) {
        int n = sub_abc2faceID(j);
        sub_abc2faceInf(j, 0) = sub_abc2elems(j);
        sub_abc2faceInf(j, 1) = sub_abc2faces(j, n);
    }
    VectorXi subEdgesABC;
    igl::unique(sub_abc2edges.reshaped(abc_idx.size() * nbasis, 1), subEdgesABC);
    MatrixXi gli_abc, sub_abc2edges_order_vec;
    igl::ismember_rows(sub_abc2edges.transpose().reshaped(abc_idx.size() * nbasis, 1), sub_edgeOrder, gli_abc, sub_abc2edges_order_vec);
    MatrixXi sub_abc2edges_order = sub_abc2edges_order_vec.reshaped(nbasis, abc_idx.size()).transpose();

    if (my_rank == 0) {
        std::cout << "STEP 1. Pre-process done\n" << endl;
    }

    //---------------------------------------
    // Stiffness matrix and Mass matrix
    //---------------------------------------
    std::vector<S_T> S_trp;
    std::vector<M_T> M_trp;
    Eigen::SparseMatrix<double> STIFF(sub_dofs, sub_dofs);
    Eigen::SparseMatrix<double> MASS(sub_dofs, sub_dofs);
    for (size_t ne = 0; ne < sub_elems2nodes.rows(); ne++) {
        // Initialization
        VectorXi curE = sub_elems2nodes.row(ne);
        MatrixXd curV(nverts, dim);
        for (size_t nv = 0; nv < nverts; nv++) {
            curV.row(nv) = nodes2coord.row(curE(nv));
        }
        VectorXi signE = sub_elems2signs.row(ne);
        VectorXi edgeE = sub_elems2edges_order.row(ne);
        // Affine transformation (B_K, B_K_detA=abs(det(B_K)), b_K)
        MatrixXd B_K = (curV.bottomRows(curV.rows() - 1).array() - curV.row(0).replicate(curV.rows() - 1., 1).array()).cast<double>().transpose();
        MatrixXd b_K = curV.row(0);
        double B_K_detA = abs(B_K.determinant());
        // Mass matrix and Stiffness matrix
        MatrixXd mass(nbasis, nbasis); mass.setZero();
        MatrixXd stiff(nbasis, nbasis); stiff.setZero();
        for (int j = 0; j < nbasis; j++) {
            std::pair<MatrixXd, MatrixXd> Basis = Nedelec0BasisFunctions::getNedelec0BasisFunction(ip, j + 1);
            MatrixXd Nj = Basis.first;
            MatrixXd cNj = Basis.second;
            for (int k = 0; k < nbasis; k++) {
                std::pair<MatrixXd, MatrixXd> Basis = Nedelec0BasisFunctions::getNedelec0BasisFunction(ip, k + 1);
                MatrixXd Nk = Basis.first;
                MatrixXd cNk = Basis.second;
                for (int m = 0; m < ip.rows(); m++) {
                    S_trp.push_back(S_T(edgeE[j], edgeE[k], (1 / B_K_detA) * w(m) * (signE(j) * B_K * cNj.row(m).transpose()).transpose() * (signE(k) * B_K * cNk.row(m).transpose())));
                    M_trp.push_back(M_T(edgeE[j], edgeE[k], B_K_detA * w(m) * (signE(j) * B_K.inverse().transpose() * Nj.row(m).transpose()).transpose() * (signE(k) * B_K.inverse().transpose() * Nk.row(m).transpose())));
                }
            }
        }
    }
    STIFF.setFromTriplets(S_trp.begin(), S_trp.end());
    MASS.setFromTriplets(M_trp.begin(), M_trp.end());

    //---------------------------------------
    // Boundary integral on ABC
    //---------------------------------------    
    std::vector<ABC_T> ABC_trp;
    Eigen::SparseMatrix<cdouble> ABC(sub_dofs, sub_dofs);
       
    for (size_t ne = 0; ne < sub_abc2elems.size(); ne++) {
        // Initialization
        VectorXi curE = sub_abc2nodes.row(ne);
        MatrixXd curEV(curE.rows(), dim);
        for (size_t nv = 0; nv < curEV.rows(); nv++) {
            curEV.row(nv) = nodes2coord.row(curE(nv));
        }
        VectorXi curF = sub_abc2Fnodes.row(ne);
        MatrixXd curFV(curF.rows(), dim);
        for (size_t nv = 0; nv < curFV.rows(); nv++) {
            curFV.row(nv) = nodes2coord.row(curF(nv));
        }
        VectorXi signE = sub_abc2signs.row(ne);
        VectorXi edgeE = sub_abc2edges_order.row(ne);
        VectorXd normal = generateBoundaryNormalVector(curEV, curFV, 0);
        cdouble area = normal.norm() / 2;
        Eigen::Vector3d curNV = normal / normal.norm();
        // Get Gaussian quadarature points in 3D Faces
        int pF = 2;
        std::pair<MatrixXd, MatrixXd> GQ_resultF = GaussianQuadratureFace::getGaussianQuadratureFace(pF, abc2faceID(abc_idx(ne)));
        MatrixXd ipF = GQ_resultF.first;
        MatrixXd wF = GQ_resultF.second;
        // Affine transformation (B_K, B_K_detA=abs(det(B_K)), b_K)
        MatrixXd B_K = (curEV.bottomRows(curEV.rows() - 1).array() - curEV.row(0).replicate(curEV.rows() - 1., 1).array()).cast<double>().transpose();
        MatrixXd b_K = curEV.row(0);
        double B_K_detA = abs(B_K.determinant());
        // Surface integral on ABC    
        for (int j = 0; j < nbasis; j++) {
            std::pair<MatrixXd, MatrixXd> Basis = Nedelec0BasisFunctions::getNedelec0BasisFunction(ipF, j + 1);
            MatrixXd Nj = Basis.first;
            MatrixXd cNj = Basis.second;
            for (int k = 0; k < nbasis; k++) {
                std::pair<MatrixXd, MatrixXd> Basis = Nedelec0BasisFunctions::getNedelec0BasisFunction(ipF, k + 1);
                MatrixXd Nk = Basis.first;
                MatrixXd cNk = Basis.second;
                for (int m = 0; m < ipF.rows(); m++) {
                    Eigen::Vector3d Eb = Nj.row(m).transpose();
                    Eigen::Vector3d cEb = cNj.row(m).transpose();
                    Eigen::Vector3d Et = Nk.row(m).transpose();
                    Eigen::Vector3d cEt = cNk.row(m).transpose();
                    Eigen::Vector3d BEb = signE(j) * B_K.inverse().transpose() * Eb;
                    Eigen::Vector3d BEt = signE(k) * B_K.inverse().transpose() * Et;
                    Eigen::Vector3d BcEb = signE(j) * B_K * cEb / B_K_detA;
                    Eigen::Vector3d BcEt = signE(k) * B_K * cEt / B_K_detA;

                    ABC_trp.push_back(ABC_T(edgeE[j], edgeE[k], (i * k0) * area * wF(m) * (curNV.cross(BEb)).transpose() * (curNV.cross(BEt))));
                    //ABC_trp.push_back(ABC_T(edgeE[j], edgeE[k], (1. / (2. * i * k0)) * area * wF(m) * (curNV.transpose().dot(BcEb)) * (curNV.transpose().dot(BcEt))));
                }
            }
        }
    }
    ABC.setFromTriplets(ABC_trp.begin(), ABC_trp.end());
    
    //---------------------------------------
    // Boundary integral on IBC
    //---------------------------------------    
    std::vector<IBC_T> IBC_trp;
    Eigen::SparseMatrix<cdouble> IBC(sub_dofs, sub_dofs);
    std::vector<Eigen::Triplet<cdouble>> F_trp;
    Eigen::SparseMatrix<cdouble> F(sub_dofs, 1);
    Eigen::MatrixXd N(sub_obj2elems.rows(), dim);
    for (size_t ne = 0; ne < sub_obj2elems.size(); ne++) {
        // Initialization
        VectorXi curE = sub_obj2nodes.row(ne);
        MatrixXd curEV(curE.rows(), dim);
        for (size_t nv = 0; nv < curEV.rows(); nv++) {
            curEV.row(nv) = nodes2coord.row(curE(nv));
        }
        VectorXi curF = sub_obj2Fnodes.row(ne);
        MatrixXd curFV(curF.rows(), dim);
        for (size_t nv = 0; nv < curFV.rows(); nv++) {
            curFV.row(nv) = nodes2coord.row(curF(nv));
        }
        VectorXi signE = sub_obj2signs.row(ne);
        VectorXi edgeE = sub_obj2edges_order.row(ne);
        VectorXd normal = generateBoundaryNormalVector(curEV, curFV, 0);
        cdouble area = normal.norm() / 2;
        Eigen::Vector3d curNV = -normal / normal.norm();
        N.row(ne) = curNV;
        // Get Gaussian quadarature points in 3D Faces
        int pF = 2;
        std::pair<MatrixXd, MatrixXd> GQ_resultF = GaussianQuadratureFace::getGaussianQuadratureFace(pF, obj2faceID(obj_idx(ne)));
        MatrixXd ipF = GQ_resultF.first;
        MatrixXd wF = GQ_resultF.second;
        // Affine transformation (B_K, B_K_detA=abs(det(B_K)), b_K)
        MatrixXd B_K = (curEV.bottomRows(curEV.rows() - 1).array() - curEV.row(0).replicate(curEV.rows() - 1., 1).array()).cast<double>().transpose();
        MatrixXd b_K = curEV.row(0);
        double B_K_detA = abs(B_K.determinant());
        // Surface integral on IBC
        for (int j = 0; j < nbasis; j++) {
            std::pair<MatrixXd, MatrixXd> Basis = Nedelec0BasisFunctions::getNedelec0BasisFunction(ipF, j + 1);
            MatrixXd Nj = Basis.first;
            MatrixXd cNj = Basis.second;
            for (int k = 0; k < nbasis; k++) {
                std::pair<MatrixXd, MatrixXd> Basis = Nedelec0BasisFunctions::getNedelec0BasisFunction(ipF, k + 1);
                MatrixXd Nk = Basis.first;
                MatrixXd cNk = Basis.second;
                for (int m = 0; m < ipF.rows(); m++) {
                    Eigen::Vector3d Eb = Nj.row(m).transpose();
                    Eigen::Vector3d cEb = cNj.row(m).transpose();
                    Eigen::Vector3d Et = Nk.row(m).transpose();
                    Eigen::Vector3d cEt = cNk.row(m).transpose();
                    Eigen::Vector3d BEb = signE(j) * B_K.inverse().transpose() * Eb;
                    Eigen::Vector3d BEt = signE(k) * B_K.inverse().transpose() * Et;
                    IBC_trp.push_back(IBC_T(edgeE[j], edgeE[k], area * wF(m) * (curNV.cross(BEb)).transpose() * (curNV.cross(BEt))));
                }
            }
        }
        for (int k = 0; k < nbasis; k++) {
            std::pair<MatrixXd, MatrixXd> Basis = Nedelec0BasisFunctions::getNedelec0BasisFunction(ipF, k + 1);
            MatrixXd Nk = Basis.first;
            MatrixXd cNk = Basis.second;
            for (int m = 0; m < ipF.rows(); m++) {
                VectorXd cipF = ipF.row(m);
                VectorXd cipFp = B_K * cipF + b_K.transpose();
                Eigen::Vector3d Et = Nk.row(m).transpose();
                Eigen::Vector3d BEt = signE(k) * B_K.inverse().transpose() * Et;

                //Stealth
                Eigen::Vector3cd Fv; Fv << 0, 0, exp(i * k0 * cipFp(0));
                Eigen::Vector3cd cFv; cFv << 0, -i * k0 * exp(i * k0 * cipFp(0)), 0;
                //Sphere
                //Eigen::Vector3cd Fv; Fv << exp(-i * k0 * cipFp(2)), 0, 0;
                //Eigen::Vector3cd cFv; cFv << 0, -i * k0 * exp(-i * k0 * cipFp(2)), 0;
                //Boat
                //Eigen::Vector3cd Fv; Fv << exp(i * k0 * cipFp(2)), 0, 0;
                //Eigen::Vector3cd cFv; cFv << 0, i * k0 * exp(i * k0 * cipFp(2)), 0;

                Eigen::Vector3cd ncFv; ncFv << curNV(1) * Fv(2) - curNV(2) * Fv(1), curNV(2)* Fv(0) - curNV(0) * Fv(2), curNV(0)* Fv(1) - curNV(1) * Fv(0);
                Eigen::Vector3cd nccFv; nccFv << curNV(1) * cFv(2) - curNV(2) * cFv(1), curNV(2)* cFv(0) - curNV(0) * cFv(2), curNV(0)* cFv(1) - curNV(1) * cFv(0);

                F_trp.push_back(Eigen::Triplet<cdouble>(edgeE(k), 0, -area * wF(m) * nccFv.transpose() * BEt));
                F_trp.push_back(Eigen::Triplet<cdouble>(edgeE(k), 0, -(i * k0 / eta) * area * wF(m) * ncFv.transpose() * curNV.cross(BEt)));

            }
        }
    }
    IBC.setFromTriplets(IBC_trp.begin(), IBC_trp.end());
    F.setFromTriplets(F_trp.begin(), F_trp.end());
    //---------------------------------------
    // Construct K matrix
    //---------------------------------------  
    Eigen::SparseMatrix<cdouble> K(nedges, nedges);
    K = (1.0 / mu) * STIFF - pow(k0, 2) * eps * MASS + ABC + (i * k0 / eta) * IBC;
    K.makeCompressed();

    //---------------------------------------
    // Boundary integral on SOTC
    //---------------------------------------    
    orderFaces << 0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3;
    auto sub_facesResult = generateEdgesOrFaces(sub_elems2nodes, orderFaces);
    MatrixXi sub_elems2faces_loc = sub_facesResult.first;
    MatrixXi sub_faces2nodes_loc = sub_facesResult.second;
    std::pair<MatrixXi, MatrixXi> sub_BF_result = generateBoundaryFacesToNodes(sub_elems2faces_loc, sub_faces2nodes_loc);
    MatrixXi sub_bfaces2nodes = sub_BF_result.first;
    MatrixXi sub_bfaces2elems_loc = sub_BF_result.second;
    //VectorXi sub_bfaces2elems = sub_idx(sub_bfaces2elems_loc);
    VectorXi sub_bfaces2elems(sub_bfaces2elems_loc.rows());
    for (size_t j = 0; j < sub_bfaces2elems_loc.rows(); j++) {
        sub_bfaces2elems(j) = sub_idx(sub_bfaces2elems_loc(j));
    }
    VectorXi sub_bfaces2faceID;
    sub_bfaces2faceID = generateBoundaryFacesIndex(sub_bfaces2elems, sub_bfaces2nodes, elems2faces, faces2nodes);
    MatrixXi sub_bfaces2faceInf(sub_bfaces2elems.rows(), 3);
    for (size_t j = 0; j < sub_bfaces2elems.rows(); j++) {
        int n = sub_bfaces2faceID(j);
        sub_bfaces2faceInf(j, 0) = sub_bfaces2elems(j);
        sub_bfaces2faceInf(j, 1) = elems2faces(sub_bfaces2elems(j), n);
        sub_bfaces2faceInf(j, 2) = n;
    }
    VectorXi babc_logic0, LOCBAb0, babc_logic1, LOCBAb1;
    igl::ismember_rows(sub_bfaces2faceInf.col(0), sub_abc2faceInf.col(0), babc_logic0, LOCBAb0);
    igl::ismember_rows(sub_bfaces2faceInf.col(1), sub_abc2faceInf.col(1), babc_logic1, LOCBAb1);
    VectorXi babc_logic_sum = babc_logic0 + babc_logic1;
    VectorXi sotc_idx1;
    igl::find(babc_logic_sum.array() != 2, sotc_idx1);
    VectorXi bobj_logic0, LOCBOb0, bobj_logic1, LOCBOb1;
    igl::ismember_rows(sub_bfaces2faceInf.col(0), sub_obj2faceInf.col(0), bobj_logic0, LOCBOb0);
    igl::ismember_rows(sub_bfaces2faceInf.col(1), sub_obj2faceInf.col(1), bobj_logic1, LOCBOb1);
    VectorXi bobj_logic_sum = bobj_logic0 + bobj_logic1;
    VectorXi sotc_idx2;
    igl::find(bobj_logic_sum.array() != 2, sotc_idx2);
    VectorXi sub_sotc_idx;
    igl::intersect(sotc_idx1, sotc_idx2, sub_sotc_idx);
    int sub_ifn = sub_sotc_idx.size();
    VectorXi sub_sotc2elems(sub_ifn);
    VectorXi sub_sotc2faceID(sub_ifn);
    MatrixXi sub_sotc2Fnodes(sub_ifn, dim);
    MatrixXi sub_sotc2nodes(sub_ifn, nverts);
    MatrixXi sub_sotc2edges(sub_ifn, nbasis);
    MatrixXi sub_sotc2signs(sub_ifn, nbasis);

    for (size_t j = 0; j < sub_ifn; j++) {
        sub_sotc2elems(j) = sub_bfaces2faceInf(sub_sotc_idx(j), 0);
        sub_sotc2faceID(j) = sub_bfaces2faceInf(sub_sotc_idx(j), 2);
        sub_sotc2Fnodes.row(j) = sub_bfaces2nodes.row(sub_sotc_idx(j));
    }
    for (size_t j = 0; j < sub_ifn; j++) {
        sub_sotc2nodes.row(j) = elems2nodes.row(sub_sotc2elems(j));
        sub_sotc2edges.row(j) = elems2edges.row(sub_sotc2elems(j));
        sub_sotc2signs.row(j) = signs2edges.row(sub_sotc2elems(j));
    }
    VectorXi subEdgesSOTC;
    igl::unique(sub_sotc2edges.reshaped(sub_sotc2elems.size() * nbasis, 1), subEdgesSOTC);

    MatrixXi gli_sotc, sub_sotc2edges_order_vec;
    igl::ismember_rows(sub_sotc2edges.transpose().reshaped(sub_sotc2elems.size() * nbasis, 1), sub_edgeOrder, gli_sotc, sub_sotc2edges_order_vec);
    MatrixXi sub_sotc2edges_order = sub_sotc2edges_order_vec.reshaped(nbasis, sub_sotc2elems.size()).transpose();


    std::vector<SOTC_T> SOTC_trp;
    Eigen::SparseMatrix<cdouble> SOTC(sub_dofs, sub_dofs);
    for (size_t ne = 0; ne < sub_sotc2elems.size(); ne++) {
        // Initialization
        VectorXi curE = sub_sotc2nodes.row(ne);
        MatrixXd curEV(curE.rows(), dim);
        for (size_t nv = 0; nv < curEV.rows(); nv++) {
            curEV.row(nv) = nodes2coord.row(curE(nv));
        }
        VectorXi curF = sub_sotc2Fnodes.row(ne);
        MatrixXd curFV(curF.rows(), dim);
        for (size_t nv = 0; nv < curFV.rows(); nv++) {
            curFV.row(nv) = nodes2coord.row(curF(nv));
        }
        VectorXi signE = sub_sotc2signs.row(ne);
        VectorXi edgeE = sub_sotc2edges_order.row(ne);
        VectorXd normal = generateBoundaryNormalVector(curEV, curFV, 0);
        cdouble area = normal.norm() / 2;
        Eigen::Vector3cd curNV = normal / normal.norm();
        // Get Gaussian quadarature points in 3D Faces
        int pF = 2;
        std::pair<MatrixXd, MatrixXd> GQ_resultF = GaussianQuadratureFace::getGaussianQuadratureFace(pF, sub_sotc2faceID(ne));
        MatrixXd ipF = GQ_resultF.first;
        MatrixXd wF = GQ_resultF.second;
        // Affine transformation (B_K, B_K_detA=abs(det(B_K)), b_K)
        MatrixXd B_K = (curEV.bottomRows(curEV.rows() - 1).array() - curEV.row(0).replicate(curEV.rows() - 1., 1).array()).cast<double>().transpose();
        MatrixXd b_K = curEV.row(0);
        double B_K_detA = abs(B_K.determinant());
        // Surface integral on ABC
        for (int j = 0; j < nbasis; j++) {
            std::pair<MatrixXd, MatrixXd> Basis = Nedelec0BasisFunctions::getNedelec0BasisFunction(ipF, j + 1);
            MatrixXd Nj = Basis.first;
            MatrixXd cNj = Basis.second;
            for (int k = 0; k < nbasis; k++) {
                std::pair<MatrixXd, MatrixXd> Basis = Nedelec0BasisFunctions::getNedelec0BasisFunction(ipF, k + 1);
                MatrixXd Nk = Basis.first;
                MatrixXd cNk = Basis.second;
                for (int m = 0; m < ipF.rows(); m++) {
                    Eigen::Vector3d Eb = Nj.row(m).transpose();
                    Eigen::Vector3d cEb = cNj.row(m).transpose();
                    Eigen::Vector3d Et = Nk.row(m).transpose();
                    Eigen::Vector3d cEt = cNk.row(m).transpose();
                    Eigen::Vector3d BEb = signE(j) * B_K.inverse().transpose() * Eb;
                    Eigen::Vector3d BEt = signE(k) * B_K.inverse().transpose() * Et;
                    Eigen::Vector3d BcEb = signE(j) * B_K * cEb / B_K_detA;
                    Eigen::Vector3d BcEt = signE(k) * B_K * cEt / B_K_detA;
                    SOTC_trp.push_back(SOTC_T(edgeE[j], edgeE[k], alpha * area * wF(m) * (curNV.cross(BEb)).transpose() * (curNV.cross(BEt))));
                    SOTC_trp.push_back(SOTC_T(edgeE[j], edgeE[k], beta * area * wF(m) * (curNV.transpose().dot(BcEb)) * (curNV.transpose().dot(BcEt))));
                }
            }
        }
    }
    SOTC.setFromTriplets(SOTC_trp.begin(), SOTC_trp.end());
    if (my_rank == 0) {
        std::cout << "STEP 2. Construction of K, M matrix done\n" << endl;
    }
    //---------------------------------------
    // Construct block matrix ([K_rr], [K_rc], [K_cr], [K_cc], [M_bb], [M_bc], {F_r}, {F_c})
    //---------------------------------------    
    Eigen::SparseMatrix<cdouble> K_rr, K_rc, K_cr, M_bb, M_bc, K_cc;
    Eigen::SparseMatrix<cdouble> K_rr_1, K_rr_2, M_bb_1, M_bc_1;
    K_rr_1 = K.topLeftCorner(sub_nr, sub_nr);
    K_rr_2 = SOTC.topLeftCorner(sub_nr, sub_nr);
    K_rr = K_rr_1 + K_rr_2;
    M_bb_1 = SOTC.topLeftCorner(sub_nr, sub_nr);
    M_bb = M_bb_1.bottomRightCorner(sub_nb, sub_nb);
    VectorXcd F_r = F.topRightCorner(sub_nr, 1);
    VectorXcd F_c;
    if (sub_nc > 0) {
        K_rc = K.topRightCorner(sub_nr, sub_nc);
        K_cr = K.bottomLeftCorner(sub_nc, sub_nr);
        M_bc_1 = SOTC.topLeftCorner(sub_nr, sub_dofs);
        M_bc = M_bc_1.bottomRightCorner(sub_nb, sub_nc);
        K_cc = K.bottomRightCorner(sub_nc, sub_nc);
        F_c = F.bottomRightCorner(sub_nc, 1);
    }

    //---------------------------------------
    // Global corner-related FE system
    //---------------------------------------
    VectorXi glob2sub_ec_logic, glob2sub_ec_ind;
    igl::ismember_rows(sub_ec, ec, glob2sub_ec_logic, glob2sub_ec_ind);

    ZMUMPS_STRUC_C id_Krr = ZMumpsSolverKrr::AnalyzeAndFactorize(K_rr);
    VectorXcd K_rr_inv_F = ZMumpsSolverKrr::solveVector(id_Krr, F_r);

    // Construction of Kt_cc
    MatrixXcd Kt_cc_sub, K_rr_inv_K_rc_rm;
    Eigen::SparseMatrix<cdouble> K_rc_rm;
    Eigen::SparseMatrix<cdouble> Kt_cc(nc, nc);
    if (sub_nc > 0) {
        K_rc_rm = K_rc;
        for (size_t j = 0; j < sub_nb; j++) {
            for (size_t k = 0; k < sub_nc; k++) {
                K_rc_rm.coeffRef(sub_ni + j, k) += M_bc.coeffRef(j, k);
            }
        }
        K_rr_inv_K_rc_rm = ZMumpsSolverKrr::solveMatrix(id_Krr, K_rc_rm);
        Kt_cc_sub = K_cc - K_cr * K_rr_inv_K_rc_rm;

        std::vector<Eigen::Triplet<cdouble>> tripletList_Kt_cc;
        tripletList_Kt_cc.reserve(glob2sub_ec_ind.size() * glob2sub_ec_ind.size());
        for (int j = 0; j < glob2sub_ec_ind.size(); j++) {
            for (int k = 0; k < glob2sub_ec_ind.size(); k++) {
                tripletList_Kt_cc.emplace_back(glob2sub_ec_ind(j), glob2sub_ec_ind(k), Kt_cc_sub(j, k));
            }
        }
        Kt_cc.setFromTriplets(tripletList_Kt_cc.begin(), tripletList_Kt_cc.end());
    }

    // Construction of {Ft_c} and {F_c}
    VectorXcd K_rr_f;
    if (sub_nc > 0) {
        K_rr_f = F_c - K_cr * K_rr_inv_F;
    }
    VectorXcd Ft_c(nc); Ft_c.setZero(); Ft_c(glob2sub_ec_ind) = K_rr_f;

    if (my_rank == 0) {
        std::cout << "STEP 3. Global corner-related FE system done\n" << endl;
    }
    //---------------------------------------
    // Global interface system 
    //---------------------------------------
    MatrixXcd F_bc;
    if (sub_nc > 0) {
        F_bc = M_bb * K_rr_inv_K_rc_rm.bottomRightCorner(sub_nb, sub_nc);
    }
    VectorXcd d_b = M_bb * K_rr_inv_F.bottomRightCorner(sub_nb, 1);


    ZMUMPS_STRUC_C id_Kcc = ZMumpsSolverKcc::AnalyzeAndFactorize(glob2sub_ec_ind, Kt_cc);
    int maxIter = 100;
    double tolerance = 1e-04;

    tuple<int, double, double, VectorXcd> DUAL = BiCGSTABdualSolver(status, id_Krr, id_Kcc, my_rank, maxIter, tolerance, sub_eb, sub_dom, glob2sub_ec_ind, d_b, F_bc, Ft_c, K_cr, M_bb, M_bc);
    int DUAL_Iter = get<0>(DUAL); double DUAL_Time = get<1>(DUAL), DUAL_Error = get<2>(DUAL); VectorXcd dual = get<3>(DUAL);
    if (my_rank == 0) {
        std::cout << "Compute dual variable >>> Iter : " << DUAL_Iter << " / Time : " << DUAL_Time << " / Error :  " << DUAL_Error << endl;
    }  

    VectorXcd R(sub_ni); R.setZero(); VectorXcd Rd; igl::cat(1, R, dual, Rd);
    VectorXcd KrrInv_Rd = ZMumpsSolverKrr::solveVector(id_Krr, Rd);
    VectorXcd Kcr_KrrInv_Rd;
    if (sub_nc > 0) {
        Kcr_KrrInv_Rd = K_cr * KrrInv_Rd;
    }
    VectorXcd Kt_cb(nc); Kt_cb.setZero(); Kt_cb(glob2sub_ec_ind) = Kcr_KrrInv_Rd;
    VectorXcd RHS_Ec = Ft_c + Kt_cb;
    VectorXcd Ec = ZMumpsSolverKcc::solve(my_rank, id_Kcc, RHS_Ec);  id_Kcc.job = JOB_END;
    VectorXcd E_c = Ec(glob2sub_ec_ind);
    VectorXcd RHS_Er(F_r.size());
    if (sub_nc > 0) {
        RHS_Er = F_r - Rd - K_rc_rm * E_c;
    }
    else {
        RHS_Er = F_r - Rd;
    }

    VectorXcd E_r = ZMumpsSolverKrr::solveVector(id_Krr, RHS_Er); id_Krr.job = JOB_END;
    VectorXcd E_rc; igl::cat(1, E_r, E_c, E_rc);

    VectorXcd sol_sort(dofs); sol_sort.setZero();
    sol_sort(sub_edgeOrder) = E_rc;

    VectorXcd sol_sum(dofs);
    if (my_rank != 0) {
        MPI_Send(sol_sort.data(), sol_sort.size(), MPI_DOUBLE_COMPLEX, 0, 004, MPI_COMM_WORLD);
    }
    if (my_rank == 0) {
        sol_sum = sol_sort;
        for (int j = 1; j < comm_sz; j++) {
            VectorXcd sol_0(dofs);
            MPI_Recv(sol_0.data(), sol_0.size(), MPI_DOUBLE_COMPLEX, j, 004, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sol_sum += sol_0;
        }
    }
    VectorXcd sol(dofs);
    if (my_rank == 0) {
        sol = sol_sum.array() / check_sum.cast<double>().array();
    }

    finish = clock();
    duration = (double)(finish - start);

    if (my_rank == 0) {
        std::cout << " Total elpased time = " << duration / CLOCKS_PER_SEC << " (s)\n" << endl;
    }

    if (my_rank == 0) {
        std::cout << "Degree Of Freedoms = " << dofs << "\n" << endl;

        ofstream fout18("sol_real.txt"); ofstream fout19("sol_imag.txt");
        for (int j = 0; j < sol.size(); j++) {
            fout18 << sol(j).real() << endl;
            fout19 << sol(j).imag() << endl;
        }
        fout18.close(); fout19.close();
    }

    /* End MPI */
    MPI_Finalize();
    return EXIT_SUCCESS;
}
