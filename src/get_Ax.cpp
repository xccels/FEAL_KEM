#include "get_Ax.h"
#include "ZMumps_solver_Krr.h"
#include "ZMumps_solver_Kcc.h"

#include <mpi.h>
#include <Eigen/Core>
#include "igl/cat.h"
#include "igl/find.h"
#include "igl/ismember_rows.h"


VectorXcd ComputeAx(const VectorXcd& x, MPI_Status& status, ZMUMPS_STRUC_C& id_Krr, ZMUMPS_STRUC_C& id_Kcc,
    const int& nc, const int& my_rank, const int& iters, const double& tol_error,
    const VectorXi& sub_eb, const VectorXi& sub_dom, const VectorXi& glob2sub_ec_ind,
    const MatrixXcd& F_bc, const Eigen::SparseMatrix<cdouble>& K_cr, const Eigen::SparseMatrix<cdouble>& M_bb, const Eigen::SparseMatrix<cdouble>& M_bc) {

    //----------------------------------------------------------------------
    
    int sub_nr = id_Krr.n;
    int sub_nb = M_bb.rows();
    int sub_ni = sub_nr - sub_nb;
    int sub_nc = glob2sub_ec_ind.size();
    VectorXcd R(sub_ni); R.setZero();
    VectorXcd Rx; igl::cat(1, R, x, Rx);
    VectorXcd KrrInv_Rx = ZMumpsSolverKrr::solveVector(id_Krr, Rx);
       
    VectorXcd FbbX = M_bb * KrrInv_Rx.bottomRightCorner(sub_nb, 1);
    VectorXcd A1_sub = x - 2. * FbbX;

    VectorXcd Kt_cb(nc); Kt_cb.setZero();
    VectorXcd Kcr_KrrInv_Rx;
    if (sub_nc > 0) {
        Kcr_KrrInv_Rx = K_cr * KrrInv_Rx;
        Kt_cb(glob2sub_ec_ind) = Kcr_KrrInv_Rx;
    }
    VectorXcd KccInvKcbX = ZMumpsSolverKcc::solve(my_rank, id_Kcc, Kt_cb);
    VectorXcd Bc_KccInvKcbX = KccInvKcbX(glob2sub_ec_ind);
    VectorXcd A2_sub(sub_nb);
    if (sub_nc > 0) {
        A2_sub = 2. * (F_bc - M_bc) * Bc_KccInvKcbX;
    }
    else {
        A2_sub.setZero();
    }    
    
    VectorXcd A1(sub_nb); A1.setZero();
    VectorXcd A2(sub_nb); A2.setZero();
    for (size_t j = 0; j < sub_dom.size(); j++) {
        int q = sub_dom(j);
        int send_tag4 = 1000 * (my_rank + 1) + 100 * 4 + (q + 1);   int recv_tag4 = 1000 * (q + 1) + 100 * 4 + (my_rank + 1);
        int send_tag5 = 1000 * (my_rank + 1) + 100 * 5 + (q + 1);   int recv_tag5 = 1000 * (q + 1) + 100 * 5 + (my_rank + 1);
        int send_tag6 = 1000 * (my_rank + 1) + 100 * 6 + (q + 1);   int recv_tag6 = 1000 * (q + 1) + 100 * 6 + (my_rank + 1);
        int send_tag7 = 1000 * (my_rank + 1) + 100 * 7 + (q + 1);   int recv_tag7 = 1000 * (q + 1) + 100 * 7 + (my_rank + 1);

        int sub_nb_q;
        MPI_Sendrecv(&sub_nb, 1, MPI_INT, q, send_tag4, &sub_nb_q, 1, MPI_INT, q, recv_tag4, MPI_COMM_WORLD, &status);

        VectorXi sub_eb_q(sub_nb_q);
        MPI_Sendrecv(sub_eb.data(), sub_nb, MPI_INT, q, send_tag5, sub_eb_q.data(), sub_nb_q, MPI_INT, q, recv_tag5, MPI_COMM_WORLD, &status);
        

        VectorXcd A1_sub_q(sub_nb_q);
        MPI_Sendrecv(A1_sub.data(), sub_nb, MPI_DOUBLE_COMPLEX, q, send_tag6, A1_sub_q.data(), sub_nb_q, MPI_DOUBLE_COMPLEX, q, recv_tag6, MPI_COMM_WORLD, &status);

        VectorXcd A2_sub_q(sub_nb_q);
		MPI_Sendrecv(A2_sub.data(), sub_nb, MPI_DOUBLE_COMPLEX, q, send_tag7, A2_sub_q.data(), sub_nb_q, MPI_DOUBLE_COMPLEX, q, recv_tag7, MPI_COMM_WORLD, &status);
           
        //--------------------------------------------------------
        VectorXi tqs_logic, tsq_rowf;
        igl::ismember_rows(sub_eb_q, sub_eb, tqs_logic, tsq_rowf);
        VectorXi tqs_idx, tsq_row, tsq_col, tsq_val;
        igl::find(tqs_logic.array() == 1, tqs_idx);
        igl::find(tsq_rowf.array() != -1, tsq_row);
        int sub_nb_sq = tqs_idx.size();
        VectorXi tsq_idx(sub_nb_sq);
        for (size_t k = 0; k < sub_nb_sq; k++) {
            tsq_idx(k) = tsq_rowf(tsq_row(k));
        }
        //--------------------------------------------------------
        for (size_t k = 0; k < sub_nb_sq; k++) {
            A1(tsq_idx(k)) = A1(tsq_idx(k)) + A1_sub_q(tqs_idx(k));
            A2(tsq_idx(k)) = A2(tsq_idx(k)) + A2_sub_q(tqs_idx(k));
        }
    }
    A1 = A1 + x;

    VectorXcd Ax = A1 - A2;
 /*   VectorXcd Ax(sub_nb); Ax.setZero();*/
	return Ax;
}