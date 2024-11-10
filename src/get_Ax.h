#ifndef COMPUTE_AX_H
#define COMPUTE_AX_H

#include <mpi.h>
#include "zmumps_c.h"
#include "igl/cat.h"
#include "igl/find.h"
#include "igl/ismember_rows.h"

typedef std::complex<double> cdouble;
using VectorXi = Eigen::VectorXi;
using VectorXd = Eigen::VectorXd;
using VectorXcd = Eigen::VectorXcd;
using MatrixXcd = Eigen::Matrix<cdouble, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

VectorXcd ComputeAx(const VectorXcd& x, MPI_Status& status, ZMUMPS_STRUC_C& id_Krr, ZMUMPS_STRUC_C& id_Kcc,
    const int& nc, const int& my_rank, const int& iters, const double& tol_error,
    const VectorXi& sub_eb, const VectorXi& sub_dom, const VectorXi& glob2sub_ec_ind,
    const MatrixXcd& F_bc, const Eigen::SparseMatrix<cdouble>& K_cr, const Eigen::SparseMatrix<cdouble>& M_bb, const Eigen::SparseMatrix<cdouble>& M_bc);

#endif