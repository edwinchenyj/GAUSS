//
//  TimeStepperSIIMEX.h
//  Gauss
//
//  Created by Yu Ju Edwin Chen on 2018-11-13.
//

#ifndef TimeStepperSIIMEX_h
#define TimeStepperSIIMEX_h


#include <World.h>
#include <Assembler.h>
#include <TimeStepper.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
//#include <UtilitiesEigen.h>
//#include <UtilitiesMATLAB.h>
#include <FEMIncludes.h>
#include <GaussIncludes.h>
#include <UtilitiesFEM.h>
#include <State.h>
#include <ParticleSystemIncludes.h>
#include <ConstraintFixedPoint.h>
#include <unsupported/Eigen/SparseExtra>
//#include <Eigen/SparseCholesky>
#include <SolverPardiso.h>
#include <ExponentialIMEX.h>
#include <limits>
#include <igl/speye.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include <complex>
#include <cmath>
#include <Eigen/Core>
#include <GenEigsSolver.h>


class MatrixReplacement;
using Eigen::SparseMatrix;

namespace Eigen {
    namespace internal {
        // MatrixReplacement looks-like a SparseMatrix, so let's inherits its traits:
        template<>
        struct traits<MatrixReplacement> :  public Eigen::internal::traits<Eigen::SparseMatrix<double> > {};
    }
}
// Example of a matrix-free wrapper from a user type to Eigen's compatible type
// For the sake of simplicity, this example simply wrap a Eigen::SparseMatrix.
class MatrixReplacement : public Eigen::EigenBase<MatrixReplacement> {
public:
    // Required typedefs, constants, and method:
    typedef double Scalar;
    typedef double RealScalar;
    typedef int StorageIndex;

    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };

    Index rows() const { return 2*(M_lumped->rows()); }
    Index cols() const { return 2*(MinvK->cols()); }

    template<typename Rhs>
    Eigen::Product<MatrixReplacement,Rhs,Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
        return Eigen::Product<MatrixReplacement,Rhs,Eigen::AliasFreeProduct>(*this, x.derived());

    }
    // Custom API:
//    MatrixReplacement() : NK(0) {}
    void attachMyMatrix(const SparseMatrix<double> &minvk, const Eigen::VectorXx<double> &mass, double a, double b, double dt) {
//        mp_mat = &mat;
        MinvK = &minvk;
        M_lumped = &mass;
        this->a = a;
        this->b = b;
        this->dt = dt;
        Eigen::saveMarketDat(*MinvK, "MinvK.dat");
//        (*minvk) = -mass.asDiagonal().inverse() * nStiff;
    }
//    const SparseMatrix<double> my_matrix() const { return *mp_mat; }
    const SparseMatrix<double> minvk() const { return *MinvK; }
    const Eigen::VectorXx<double> mass() const { return *M_lumped; }
//    const SparseMatrix<double> minvk() const { return (*minvk); }
    double a, b, dt;
private:
//    const SparseMatrix<double> *mp_mat;
    const SparseMatrix<double> *MinvK;
    const Eigen::VectorXx<double> *M_lumped;
//    SparseMatrix<double> *minvk;
    

};
// Implementation of MatrixReplacement * Eigen::DenseVector though a specialization of internal::generic_product_impl:
namespace Eigen {
    namespace internal {
        template<typename Rhs>
        struct generic_product_impl<MatrixReplacement, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
        : generic_product_impl_base<MatrixReplacement,Rhs,generic_product_impl<MatrixReplacement,Rhs> >
        {
            typedef typename Product<MatrixReplacement,Rhs>::Scalar Scalar;
            template<typename Dest>
            static void scaleAndAddTo(Dest& dst, const MatrixReplacement& lhs, const Rhs& rhs, const Scalar& alpha)
            {
                // This method should implement "dst += alpha * lhs * rhs" inplace,
                // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
                assert(alpha==Scalar(1) && "scaling is not implemented");
                EIGEN_ONLY_USED_FOR_DEBUG(alpha);
                // Here we could simply call dst.noalias() += lhs.my_matrix() * rhs,
                // but let's do something fancier (and less efficient):
//                for(Index i=0; i<lhs.cols(); ++i)
//                    dst += rhs(i) * lhs.my_matrix().col(i);
                dst.head(lhs.rows()/2) = rhs.head(lhs.rows()/2) - lhs.dt * rhs.tail(lhs.rows()/2);
                dst.tail(lhs.rows()/2) = lhs.dt * lhs.minvk()*rhs.head(lhs.rows()/2) + rhs.tail(lhs.rows()/2) - lhs.dt * lhs.b * lhs.minvk() * rhs.tail(lhs.rows()/2);
            }
        };
//        template<typename Rhs>
//        struct generic_product_impl<MatrixReplacement, Rhs, SparseShape, DenseShape, GemmProduct> // GEMM stands for matrix-matrix
//        : generic_product_impl_base<MatrixReplacement,Rhs,generic_product_impl<MatrixReplacement,Rhs> >
//        {
//            typedef typename Product<MatrixReplacement,Rhs>::Scalar Scalar;
//            template<typename Dest>
//            static void scaleAndAddTo(Dest& dst, const MatrixReplacement& lhs, const Rhs& rhs, const Scalar& alpha)
//            {
//                // This method should implement "dst += alpha * lhs * rhs" inplace,
//                // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
//                assert(alpha==Scalar(1) && "scaling is not implemented");
//                EIGEN_ONLY_USED_FOR_DEBUG(alpha);
//                // Here we could simply call dst.noalias() += lhs.my_matrix() * rhs,
//                // but let's do something fancier (and less efficient):
//                //                for(Index i=0; i<lhs.cols(); ++i)
//                //                    dst += rhs(i) * lhs.my_matrix().col(i);
//                dst.head(lhs.rows()/2) = rhs.head(lhs.rows()/2) - lhs.dt * rhs.tail(lhs.rows()/2);
//                dst.tail(lhs.rows()/2) = lhs.dt * lhs.minvk()*rhs.head(lhs.rows()/2) + rhs.tail(lhs.rows()/2) - lhs.dt * lhs.b * lhs.minvk() * rhs.tail(lhs.rows()/2);
//            }
//        };
    }
}



void phi(Eigen::MatrixXd &A, Eigen::MatrixXd &output)
{



    Eigen::EigenSolver<Eigen::MatrixXd> es(A);
    Eigen::MatrixXcd D;
    D.resize(A.rows(),A.rows());
    D.setZero();
    D = es.eigenvalues().asDiagonal();
//    cout<<D<<endl;
    Eigen::MatrixXcd D_new;
    D_new.resize(A.rows(),A.rows());
    D_new.setZero();
    
    for (int j = 0; j < D.rows(); j++) {
        if(norm(D(j,j)) > 1e-10)
        {
//            cout<<D(j,j)<<endl;
//            double realpart = real(D(j,j));
//            cout<<realpart<<endl;
//            D(j.j).real(realpart-1);
//            cout<<D(j,j)<<endl;
            std::complex<double> tempc;
//            tempc.real(exp(D(j,j)).real());
//            D_new(j,j).real(tempc.real()-1);
//            cout<<D_new(j,j)<<endl;
            tempc.real(exp(D(j,j)).real() - 1);
            tempc.imag(exp(D(j,j)).imag());
                       tempc = tempc/D(j,j);
//            cout<<tempc<<endl;
            D_new(j,j).real(tempc.real());
            D_new(j,j).imag(tempc.imag());
//            cout<<D_new(j,j)<<endl;
////            std::complex<double> z3 = exp(1i * M_PI);
//            cout<<es.eigenvalues()[j]<<endl;
////            std::complex<double> z = (es.eigenvalues()[j]);
////            D(j,j) = (exp(z) )/es.eigenvalues()[j];
        }
        else
        {
            D_new(j,j).real(1.0);
            D_new(j,j).imag(0.0);

        }
    }
//
    Eigen::MatrixXcd U;
//    U.resize(A.rows(),A.rows());
    U = es.eigenvectors();
//    Eigen::saveMarketDat(U,"eigenvectors.dat");
//    Eigen::saveMarketDat(U.inverse(),"eigenvectors_inv.dat");
//    Eigen::saveMarketDat(D,"eigenvalues.dat");
//    Eigen::saveMarketDat(A,"mat.dat");
    output = ((U) * (D_new) * (U.inverse())).real();
//    Eigen::saveMarketDat(D_new,"Dnew.dat");
//    Eigen::saveMarketDat(output,"output.dat");

    //    cout<<"U:"<<endl;
//    cout<<U<<endl;
//    cout<<"U inv:"<<endl;
//    cout<<U.inverse()<<endl;
//    cout<<"D:"<<endl;
//    cout<<(D) <<endl;
//    cout<<"output:"<<endl;
//    cout<<output<<endl;
}
//TODO Solver Interface
namespace Gauss {
    
    //Given Initial state, step forward in time using linearly implicit Euler Integrator
    template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
    class TimeStepperImplSIIMEXImpl
    {
    public:
        
        template<typename Matrix>
        TimeStepperImplSIIMEXImpl(Matrix &P, double a, double b) {
            
            //            std::cout<<m_P.rows()<<std::endl;
            m_P = P;
            m_P2.resize(m_P.rows()*2, m_P.cols()*2);
            
            
            typedef Eigen::Triplet<DataType> T;
            std::vector<T> tripletList;
            tripletList.reserve(2*(m_P).nonZeros());
                for (int k=0; k<(m_P).outerSize(); ++k)
                {
                    for (Eigen::SparseMatrix<double>::InnerIterator it(m_P,k); it; ++it)
                    {
                        tripletList.push_back(T(it.row(), it.col(), it.value()));
                        tripletList.push_back(T(it.row()+m_P.rows(), it.col()+m_P.cols(), it.value()));

                    }
                }
            m_P2.setFromTriplets(tripletList.begin(), tripletList.end());

//            Eigen::saveMarketDat(m_P,"m_P.dat");
//            Eigen::saveMarketDat(m_P2,"m_P2.dat");
            
            
            
            m_factored = false;
            // refactor for every solve
            m_refactor = true;
            
            // init residual
            res = std::numeric_limits<double>::infinity();
            
            it_outer = 0;
            it_inner = 0;
            // constants from Nocedal and Wright
            step_size = 1;
            c1 = 1e-4;
            c2 = 0.9;
            
            this->a = a;
            this->b = b;
            
            inv_mass_calculated = false;
            mass_calculated = false;
            mass_lumped_calculated = false;
            
            Eigen::VectorXd ones(m_P.rows());
            ones.setOnes();
            rayleigh_b_scaling.resize(m_P.rows()*2);
            rayleigh_b_scaling.setZero();
            rayleigh_b_scaling.head(m_P.rows()) = ones;
            rayleigh_b_scaling.tail(m_P.rows()) = -b*ones;
            
        }
        
        TimeStepperImplSIIMEXImpl(const TimeStepperImplSIIMEXImpl &toCopy) {
            
        }
        
        ~TimeStepperImplSIIMEXImpl() { }
        
        //Methods
        //init() //initial conditions will be set at the begining
        template<typename World>
        void step(World &world, double dt, double t);
        
        inline typename VectorAssembler::MatrixType & getLagrangeMultipliers() { return m_lagrangeMultipliers; }
        
        double* rhs  = NULL;
        double* v_old  = NULL;
        double* v_temp = NULL;
        double* q_old = NULL;
        double* q_temp = NULL;
        
        
        // for damping
        double a;
        // negative for opposite sign for stiffness
        double b;
        
        
    protected:
        
        
        Eigen::SparseMatrix<DataType> inv_mass;
        Eigen::VectorXx<DataType> mass_lumped;
        Eigen::VectorXx<DataType> mass_lumped_inv;
        Eigen::VectorXx<DataType> mass_lumped_inv2; // double the size
        Eigen::VectorXx<DataType> rayleigh_b_scaling; // scaling matrinx for the second order system using rayleigh coeff b
        bool inv_mass_calculated, mass_calculated, mass_lumped_calculated;
        double inv_mass_norm;
        
        MatrixAssembler m_massMatrix;
//        MatrixAssembler m_massMatrix2;
//        MatrixAssembler m_massMatrix3;
        MatrixAssembler m_stiffnessMatrix;
        MatrixAssembler m_J;
        VectorAssembler m_forceVector;
        
        VectorAssembler m_fExt;
        
        //        // for calculating the residual. ugly
        //        MatrixAssembler m_massMatrixNew;
        //        MatrixAssembler m_stiffnessMatrixNew;
        //        VectorAssembler m_forceVectorNew;
        //        VectorAssembler m_fExtNew;
        
        
        Eigen::SparseMatrix<DataType> m_P;
        Eigen::SparseMatrix<DataType> m_P2; // 2 blocks of projection for second order system
        Eigen::SparseMatrix<DataType,Eigen::RowMajor> m_M;
        Eigen::SparseMatrix<DataType,Eigen::RowMajor> m_MInv;
        //storage for lagrange multipliers
        typename VectorAssembler::MatrixType m_lagrangeMultipliers;
        
        bool m_factored, m_refactor;
        
        // iteration counter
        int it_outer, it_inner;
        // residual
        double res, res_old, step_size, c1, c2;
        
        //        Eigen::VectorXd res;
        
#ifdef GAUSS_PARDISO
        
        SolverPardiso<Eigen::SparseMatrix<DataType, Eigen::RowMajor> > m_pardiso_mass;
        //        SolverPardiso<Eigen::SparseMatrix<DataType, Eigen::RowMajor> > m_pardiso_res;
        SolverPardiso<Eigen::SparseMatrix<DataType, Eigen::RowMajor> > m_pardiso;
        SolverPardiso<Eigen::SparseMatrix<DataType, Eigen::RowMajor> > m_pardiso_y;
        SolverPardiso<Eigen::SparseMatrix<DataType, Eigen::RowMajor> > m_pardiso_sol2;
#else
#endif
        
    private:
    };
}

template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
template<typename World>
void TimeStepperImplSIIMEXImpl<DataType, MatrixAssembler, VectorAssembler>::step(World &world, double dt, double t) {
    
    // TODO: should not be here... set the rayleigh damping parameter
    //    a = (std::get<0>(world.getSystemList().getStorage())[0])->a;
    //    b = (std::get<0>(world.getSystemList().getStorage())[0])->b;
    
    MatrixAssembler &stiffnessMatrix = m_stiffnessMatrix;
    MatrixAssembler &massMatrix = m_massMatrix;
    VectorAssembler &forceVector = m_forceVector;
    VectorAssembler &fExt = m_fExt;
    

    
    
    //Grab the state
    Eigen::Map<Eigen::VectorXd> q = mapStateEigen<0>(world);
    Eigen::Map<Eigen::VectorXd> qDot = mapStateEigen<1>(world);
    
    //get stiffness matrix
    ASSEMBLEMATINIT(stiffnessMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
    ASSEMBLELIST(stiffnessMatrix, world.getSystemList(), getStiffnessMatrix);
    ASSEMBLELIST(stiffnessMatrix, world.getForceList(), getStiffnessMatrix);
    ASSEMBLEEND(stiffnessMatrix);
    
    //Need to filter internal forces seperately for this applicat
    ASSEMBLEVECINIT(forceVector, world.getNumQDotDOFs());
    ASSEMBLELIST(forceVector, world.getSystemList(), getImpl().getInternalForce);
    ASSEMBLEEND(forceVector);
    
    ASSEMBLEVECINIT(fExt, world.getNumQDotDOFs());
    ASSEMBLELIST(fExt, world.getSystemList(), getImpl().getBodyForce);
    ASSEMBLEEND(fExt);
    
    
    if (!mass_lumped_calculated) {
        ASSEMBLEMATINIT(massMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
        ASSEMBLELIST(massMatrix, world.getSystemList(), getMassMatrix);
        ASSEMBLEEND(massMatrix);
        
        //constraint Projection
        (*massMatrix) = m_P*(*massMatrix)*m_P.transpose();
        
        Eigen::VectorXx<DataType> ones(m_P.rows());
        ones.setOnes();
        mass_lumped = ((*massMatrix)*ones);
        mass_lumped_inv = mass_lumped.cwiseInverse();
        mass_lumped_inv2.resize(2*mass_lumped_inv.rows());
        mass_lumped_inv2.head(mass_lumped_inv.rows()) = mass_lumped_inv;
        mass_lumped_inv2.tail(mass_lumped_inv.rows()) = mass_lumped_inv;
        mass_lumped_calculated = true;
    }
    
    (*stiffnessMatrix) = m_P*(*stiffnessMatrix)*m_P.transpose();
    
    (*forceVector) = m_P*(*forceVector);
    
    // add damping
    (*forceVector) = (*forceVector) -  (a * (*massMatrix) + b*(*stiffnessMatrix)) * m_P * ( qDot);
    
    // add external force
    (*forceVector) = (*forceVector) + m_P*(*fExt);
    
    
//    MatrixReplacement LHS;
//    LHS.attachMyMatrix(*stiffnessMatrix);
//    LHS.attachMyMatrix(mass_lumped.asDiagonal().inverse()*(-(*stiffnessMatrix)),mass_lumped,a,b,dt);
    
    if(!mass_calculated)
    {
        m_M.resize(mass_lumped.rows(),mass_lumped.rows());
        m_MInv.resize(mass_lumped_inv.rows(),mass_lumped_inv.rows());
//        Eigen::SparseMatrix<double,Eigen::RowMajor> M(mass_lumped.rows(),mass_lumped.rows());
        //    M.setZero();
        //    M += mass_lumped.asDiagonal();
        typedef Eigen::Triplet<double> T;
        std::vector<T> tripletList;
        std::vector<T> tripletList_inv;
        tripletList.reserve(mass_lumped.rows());
        for(int i = 0; i < mass_lumped.rows(); i++)
        {
            tripletList.push_back(T(i,i,mass_lumped(i)));
            tripletList_inv.push_back(T(i,i,mass_lumped_inv(i)));
        }
        m_M.setFromTriplets(tripletList.begin(),tripletList.end());
        m_MInv.setFromTriplets(tripletList_inv.begin(),tripletList_inv.end());
        
        mass_calculated = true;
    }
    std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > m_Us;
    m_Us = generalizedEigenvalueProblemNotNormalized((*stiffnessMatrix), m_M, 10, 0.00);
//    Eigen::saveMarketDat(m_M,"mass.dat");
//    Eigen::saveMarketDat(m_Us.first,"not_normalized.dat");
    Eigen::VectorXd normalizing_const;
    normalizing_const = (m_Us.first.transpose() * mass_lumped.asDiagonal() * m_Us.first).diagonal();
    normalizing_const = normalizing_const.cwiseSqrt().cwiseInverse();
    
    m_Us.first = m_Us.first * (normalizing_const.asDiagonal());
//    Eigen::saveMarketDat(m_Us.first,"normalized.dat");
    Eigen::SparseMatrix<double,Eigen::RowMajor> J;
    J.resize((*stiffnessMatrix).rows()*2,(*stiffnessMatrix).rows()*2);
    //get stiffness matrix for J
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve((*stiffnessMatrix).nonZeros()*2);
    for (int k=0; k<(*stiffnessMatrix).outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<double,Eigen::RowMajor>::InnerIterator it((*stiffnessMatrix),k); it; ++it)
        {
            tripletList.push_back(T(it.row()+(*stiffnessMatrix).rows(), it.col(), it.value()));
            tripletList.push_back(T(it.row()+(*stiffnessMatrix).rows(), it.col()+(*stiffnessMatrix).rows(), it.value()));
            
        }
    }
    (J).setFromTriplets(tripletList.begin(), tripletList.end());

//
//    ASSEMBLEMATINIT(J, 2*world.getNumQDotDOFs(), 2*world.getNumQDotDOFs());
//    ASSEMBLELISTOFFSET(J, world.getSystemList(), getStiffnessMatrix,world.getNumQDotDOFs(),0);
//    ASSEMBLELISTOFFSET(J, world.getForceList(), getStiffnessMatrix,world.getNumQDotDOFs(),0);
//    ASSEMBLELISTOFFSET(J, world.getSystemList(), getStiffnessMatrix,world.getNumQDotDOFs(),world.getNumQDotDOFs());
//    ASSEMBLELISTOFFSET(J, world.getForceList(), getStiffnessMatrix,world.getNumQDotDOFs(),world.getNumQDotDOFs());
//    ASSEMBLEEND(J);
//    (*J) = m_P2 * (*J) * m_P2.transpose();
//    Eigen::saveMarketDat(*J,"J.dat");
//    Eigen::saveMarketDat(*stiffnessMatrix,"stiffness.dat");
    
    (J) = mass_lumped_inv2.asDiagonal() * (J) * rayleigh_b_scaling.asDiagonal();

    for (int ind = 0; ind < m_P.rows(); ind++) {
        (J).coeffRef(ind,ind+m_P.rows()) = 1;
    }
//
//    Eigen::saveMarketDat(*J,"J.dat");
//    Eigen::saveMarketDat(*stiffnessMatrix,"stiffness.dat");

//
//    Eigen::VectorXd r = Eigen::VectorXd::Random(LHS.cols());
////    r.Random();
//
//    Eigen::VectorXd sol;
//    sol = LHS*r;
//
//    Eigen::saveMarketDat(*stiffnessMatrix,"stiffness.dat");
//    Eigen::saveMarketVectorDat(mass_lumped,"mass_lumped.dat");
//    Eigen::saveMarketDat(m_P,"m_P.dat");
//    Eigen::saveMarketVectorDat(r,"r.dat");
//    Eigen::saveMarketVectorDat(sol,"sol.dat");

    
//
//    q = m_P.transpose() * X.head(N);
//    qDot =  m_P.transpose() *( X.segment(N,N));
    Eigen::MatrixXx<double> U1;
    Eigen::MatrixXx<double> V1;
    Eigen::MatrixXx<double> U2;
    Eigen::MatrixXx<double> V2;
    
    U1.resize(m_Us.first.rows()*2,m_Us.first.cols());
    V1.resize(m_Us.first.rows()*2,m_Us.first.cols());
    U2.resize(m_Us.first.rows()*2,m_Us.first.cols());
    V2.resize(m_Us.first.rows()*2,m_Us.first.cols());

    U1.setZero();
    V1.setZero();
    U2.setZero();
    V2.setZero();
    
    U1.block(0,0,m_Us.first.rows(),m_Us.first.cols()) << m_Us.first;
    V1.block(m_Us.first.rows(),0,m_Us.first.rows(),m_Us.first.cols()) << mass_lumped.asDiagonal() * m_Us.first;
//    U2.block(m_Us.first.rows(),0,m_Us.first.rows(),m_Us.first.cols()) << m_Us.first * (m_Us.first.transpose() * (*stiffnessMatrix)) * m_Us.first;
    U2.block(m_Us.first.rows(),0,m_Us.first.rows(),m_Us.first.cols()) << m_Us.first * (m_Us.second.asDiagonal());
//    cout<<"mult: "<<endl;
//    cout<<((m_Us.first.transpose() * (*stiffnessMatrix)) * m_Us.first).diagonal()<<endl;
//    cout<<"ev: "<<endl;
//    cout<<m_Us.second<<endl;
    V2.block(0,0,m_Us.first.rows(),m_Us.first.cols()) << mass_lumped.asDiagonal() * m_Us.first;
    
//    Eigen::saveMarketDat(U1,"U1.dat");
//    Eigen::saveMarketDat(U2,"U2.dat");
//    Eigen::saveMarketDat(V1,"V1.dat");
//    Eigen::saveMarketDat(V2,"V2.dat");
//    //
    
    Eigen::MatrixXx<double> dt_J_G_reduced;
    dt_J_G_reduced.resize(m_Us.first.cols()*2,m_Us.first.cols()*2);
    dt_J_G_reduced.setZero();
    dt_J_G_reduced.block(0,m_Us.first.cols(),m_Us.first.cols(),m_Us.first.cols()).setIdentity();
//    dt_J_G_reduced.block(m_Us.first.cols(),0,m_Us.first.cols(),m_Us.first.cols()) << (m_Us.first.transpose() * (*stiffnessMatrix)) * m_Us.first;
    for (int ind = 0; ind < m_Us.first.cols() ; ind++) {
        dt_J_G_reduced(m_Us.first.cols() + ind ,0 + ind ) = m_Us.second(ind);
    }
//    dt_J_G_reduced.block(m_Us.first.cols(),0,m_Us.first.cols(),m_Us.first.cols()) << (m_Us.second.asDiagonal());
    dt_J_G_reduced *= dt;
//    Eigen::saveMarketDat(dt_J_G_reduced,"dt_J_G_reduced.dat");
//
//
//    Eigen::saveMarketVectorDat(q,"q.dat");
//    Eigen::saveMarketVectorDat(qDot,"qDot.dat");
//    Eigen::saveMarketVectorDat((*forceVector),"force.dat");
//
    Eigen::VectorXx<double> vG;
    vG.resize(m_P.rows());
    vG = m_Us.first * (m_Us.first.transpose() * mass_lumped.asDiagonal() * (m_P * qDot));
    
    Eigen::VectorXx<double> vH;
    vH.resize(m_P.rows());
    vH = m_P*qDot - vG;
    
//    Eigen::saveMarketVectorDat(vG,"vG.dat");
//    Eigen::saveMarketVectorDat(vH,"vH.dat");
    
    Eigen::VectorXx<double> fG;
    fG.resize(m_P.rows());
    fG = (mass_lumped.asDiagonal() * m_Us.first ) * (m_Us.first.transpose() * (*forceVector));
    Eigen::VectorXx<double> fH;
    fH.resize(m_P.rows());
    fH = (*forceVector) - fG;
    
//    Eigen::saveMarketVectorDat(fG,"fG.dat");
//    Eigen::saveMarketVectorDat(fH,"fH.dat");

    
    Eigen::SparseMatrix<double,Eigen::RowMajor> A;
    A.resize((J).rows(), (J).cols());
    A.setIdentity();
    A -= dt * (J);
    
//    Eigen::saveMarketDat(A,"A.dat");

    
#ifdef GAUSS_PARDISO
    
    m_pardiso.symbolicFactorization(A);
    m_pardiso.numericalFactorization();
//    m_pardiso.solve(*forceVector);
    

    Eigen::VectorXx<double> rhs1;
    rhs1.resize(m_P.rows()*2);
    Eigen::VectorXx<double> rhs2;
    rhs2.resize(m_P.rows()*2);
    Eigen::VectorXx<double> rhs;
    rhs.resize(m_P.rows()*2);
    
    rhs1.head(m_P.rows()) = (-dt) * vH;
    rhs1.tail(m_P.rows()) = (-dt) * mass_lumped_inv.asDiagonal() * fH;
    
//    Eigen::saveMarketVectorDat(rhs1,"rhs1c.dat");

    
    Eigen::VectorXx<double> reduced_vec;
    reduced_vec.resize(dt_J_G_reduced.cols());
    reduced_vec.head(dt_J_G_reduced.cols()/2) = m_Us.first.transpose() * (mass_lumped.asDiagonal() * (m_P * qDot));
    reduced_vec.tail(dt_J_G_reduced.cols()/2) = (m_Us.first.transpose() * (*forceVector));
//    Eigen::saveMarketVectorDat(reduced_vec,"reduced_vec.dat");
    
    Eigen::MatrixXx<double> block_diag_eigv;
    
    block_diag_eigv.resize(m_Us.first.rows()*2,m_Us.first.cols()*2);
    block_diag_eigv.setZero();
    block_diag_eigv.block(0,0,m_Us.first.rows(),m_Us.first.cols()) << m_Us.first;
    block_diag_eigv.block(m_Us.first.rows(),m_Us.first.cols(),m_Us.first.rows(),m_Us.first.cols()) << m_Us.first;
//    Eigen::saveMarketDat(block_diag_eigv,"block_diag_eigv.dat");
    
    Eigen::MatrixXx<double> phi_reduced;
    phi_reduced.resize(m_Us.first.cols()*2,m_Us.first.cols()*2);
    phi_reduced.setZero();
    
    phi((dt_J_G_reduced), phi_reduced);
//    Eigen::saveMarketDat(phi_reduced,"phi_red.dat");

    rhs2 = (-dt) * block_diag_eigv * phi_reduced * reduced_vec;
//    Eigen::saveMarketVectorDat(rhs2,"rhs2c.dat");
//    Eigen::saveMarketVectorDat(block_diag_eigv * phi_reduced * reduced_vec * (dt) * (-1),"rhs2c.dat");
    
    rhs = rhs1 + rhs2;
//    Eigen::saveMarketVectorDat(rhs,"rhsc.dat");

//    m_pardiso.solve(rhs);
//    cout<<rhs.rows()<<endl;
//    cout<<U1.rows()<<endl;
    
    Eigen::VectorXd x0;
    m_pardiso.solve(rhs);
    x0 = m_pardiso.getX();
//    Eigen::saveMarketVectorDat(x0,"x0c.dat");

    
    U1 *= dt;
    Eigen::MatrixXd x1;
    m_pardiso.solve(U1);
    x1 = m_pardiso.getX();

//    Eigen::saveMarketDat(x1,"x1c.dat");

    U2 *= dt;
    Eigen::MatrixXd x2;
    m_pardiso.solve(U2);
    x2 = m_pardiso.getX();
    
//    Eigen::saveMarketDat(x2,"x2c.dat");


    Eigen::MatrixXd Is;
    Is.resize(U1.cols(),U1.cols());
    Is.setIdentity();
    
    
    
    Eigen::MatrixXd yLHS = Is + V1.transpose()*x1;
//    Eigen::LDLT<Eigen::MatrixXd> yLHS(Is + V1.transpose()*x1);
    Eigen::VectorXd y0;
//    Eigen::VectorXd yRHS1 = V1.transpose()*x0;
//    m_pardiso_y.solve(yRHS1);
    y0 = x0 - x1 * yLHS.ldlt().solve(V1.transpose()*x0);
//    Eigen::saveMarketVectorDat(y0,"y0c.dat");

//
    Eigen::VectorXd y1;
    y1.resize(x2.rows());
    Eigen::MatrixXd yRHS2 = V1.transpose()*x2;
//    m_pardiso_y.solve(yRHS2);
    x2 -= x1 * (yLHS.ldlt().solve(yRHS2));
//    Eigen::saveMarketDat(x2,"y1c.dat");
//    y1 = x2 - x1 * (yLHS.ldlt().solve(yRHS2));
//
    Eigen::MatrixXd sol2LHS = Is + V2.transpose()*x2;
//    m_pardiso_sol2.symbolicFactorization(sol2LHS);
//    m_pardiso_sol2.numericalFactorization();
    Eigen::VectorXd sol2;
    Eigen::MatrixXd sol2RHS = V2.transpose()*y0;
//    m_pardiso_sol2.solve(sol2RHS);
    y0 -= x2 * (sol2LHS).ldlt().solve(sol2RHS);
//    Eigen::saveMarketDat(y0,"sol2.dat");
//    sol2 = y0 - y1 * (Is + V2.transpose()*y1).ldlt().solve(V2.transpose()*y0);
    
    auto state = mapStateEigen(world);
    
    state -= m_P2.transpose()*y0;

#else
#endif

//    U2 *= dt;
//    cout<<m_pardiso.getX().rows()<<endl;
//    cout<<m_pardiso.getX().cols()<<endl;

    //
//    rhs2.head(m_P.rows()) = (-dt) * vH;
//    rhs2.tail(m_P.rows()) = (-dt) * mass_lumped_inv.asDiagonal() * fH;
//    Eigen::MatrixXx<double> zeros_block;
//    zeros_block.resize(m_Us.first.rows(),m_Us.first.cols());
    
}

template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
using TimeStepperSIIMEX = TimeStepper<DataType, TimeStepperImplSIIMEXImpl<DataType, MatrixAssembler, VectorAssembler> >;

#endif /* TimeStepperSIIMEX_h */