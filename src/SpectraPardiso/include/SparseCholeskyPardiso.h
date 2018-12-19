// modified from sparsecholesky.h from spectra
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SPARSE_CHOLESKY_PARDISO_H
#define SPARSE_CHOLESKY_PARDISO_H

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <stdexcept>
#include <Util/CompInfo.h>
//#include <SolverPardiso.h>

namespace Spectra {


///
/// \ingroup MatOp
///
/// This class defines the operations related to Cholesky decomposition on a
/// sparse positive definite matrix, \f$B=LL'\f$, where \f$L\f$ is a lower triangular
/// matrix. It is mainly used in the SymGEigsSolver generalized eigen solver
/// in the Cholesky decomposition mode.
///
template <typename Scalar, int Uplo = Eigen::Lower, int Flags = 0, typename StorageIndex = int>
class SparseCholeskyPardiso
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<const Vector> MapConstVec;
    typedef Eigen::Map<Vector> MapVec;
    typedef Eigen::SparseMatrix<Scalar, Flags, StorageIndex> SparseMatrix;

    const int m_n;
    mutable SolverPardiso<Eigen::SparseMatrix<double,Eigen::RowMajor> > m_solver;
//    int m_info;  // status of the decomposition

public:
    ///
    /// Constructor to create the matrix operation object.
    ///
    /// \param mat_ An **Eigen** sparse matrix object, whose type is
    /// `Eigen::SparseMatrix<Scalar, ...>`.
    ///
    SparseCholeskyPardiso(Eigen::SparseMatrix<double,Eigen::RowMajor> &mat_) :
        m_n(mat_.rows())
    {
        if(mat_.rows() != mat_.cols())
            throw std::invalid_argument("SparseCholesky: matrix must be square");

        if(!m_solver.symbolicFactorization(mat_)) std::cout<<"error: pardiso symbolic factorization fail in spectra."<<std::endl;
        
        if(!m_solver.numericalFactorization()) std::cout<<"error: pardiso numerical factorization fail in spectra."<<std::endl;
        
    }

    ///
    /// Returns the number of rows of the underlying matrix.
    ///
    int rows() const { return m_n; }
    ///
    /// Returns the number of columns of the underlying matrix.
    ///
    int cols() const { return m_n; }

    ///
    /// Returns the status of the computation.
    /// The full list of enumeration values can be found in \ref Enumerations.
    ///
//    int info() const { return m_info; }

    ///
    /// Performs the lower triangular solving operation \f$y=L^{-1}x\f$.
    ///
    /// \param x_in  Pointer to the \f$x\f$ vector.
    /// \param y_out Pointer to the \f$y\f$ vector.
    ///
    // y_out = inv(L) * x_in
    void lower_triangular_solve(Scalar* x_in, Scalar* y_out) const
    {
        MapVec x(x_in,  m_n);
        MapVec      y(y_out, m_n);
        m_solver.lower_triangular_solve(x);
        y.noalias() = m_solver.getX();
    }

    ///
    /// Performs the upper triangular solving operation \f$y=(L')^{-1}x\f$.
    ///
    /// \param x_in  Pointer to the \f$x\f$ vector.
    /// \param y_out Pointer to the \f$y\f$ vector.
    ///
    // y_out = inv(L') * x_in
    void upper_triangular_solve(Scalar* x_in, Scalar* y_out) const
    {
        MapVec x(x_in,  m_n);
        MapVec      y(y_out, m_n);
        m_solver.upper_triangular_solve(x);
        y.noalias() = m_solver.getX();
    }
};


} // namespace Spectra

#endif // SPARSE_CHOLESKY_H
