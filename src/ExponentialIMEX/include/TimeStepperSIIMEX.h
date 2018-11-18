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
        bool inv_mass_calculated, mass_calculated, mass_lumped_calculated;
        double inv_mass_norm;
        
        MatrixAssembler m_massMatrix;
        MatrixAssembler m_stiffnessMatrix;
        VectorAssembler m_forceVector;
        VectorAssembler m_fExt;
        
        //        // for calculating the residual. ugly
        //        MatrixAssembler m_massMatrixNew;
        //        MatrixAssembler m_stiffnessMatrixNew;
        //        VectorAssembler m_forceVectorNew;
        //        VectorAssembler m_fExtNew;
        
        
        Eigen::SparseMatrix<DataType> m_P;
        Eigen::SparseMatrix<DataType,Eigen::RowMajor> m_M;
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
    
//    std::cout<<"b: "<<b<<std::endl;
    //First two lines work around the fact that C++11 lambda can't directly capture a member variable.
    MatrixAssembler &massMatrix = m_massMatrix;
    
    MatrixAssembler &stiffnessMatrix = m_stiffnessMatrix;
    VectorAssembler &forceVector = m_forceVector;
    VectorAssembler &fExt = m_fExt;
    
    
    //Grab the state
    Eigen::Map<Eigen::VectorXd> q = mapStateEigen<0>(world);
    Eigen::Map<Eigen::VectorXd> qDot = mapStateEigen<1>(world);
    
    
    //get mass matrix
    ASSEMBLEMATINIT(massMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
    ASSEMBLELIST(massMatrix, world.getSystemList(), getMassMatrix);
    ASSEMBLEEND(massMatrix);
    
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
    
    //constraint Projection
    (*massMatrix) = m_P*(*massMatrix)*m_P.transpose();
    
    if (!mass_lumped_calculated) {
        
        Eigen::VectorXx<DataType> ones(m_P.rows());
        ones.setOnes();
        mass_lumped = ((*massMatrix)*ones);
        
        mass_lumped_calculated = true;
    }
    
    (*stiffnessMatrix) = m_P*(*stiffnessMatrix)*m_P.transpose();
    
    (*forceVector) = m_P*(*forceVector);
    
    
    // add damping
    (*forceVector) = (*forceVector) -  (a * (*massMatrix) + b*(*stiffnessMatrix)) * m_P * ( qDot);
    
    // add external force
    (*forceVector) = (*forceVector) + m_P*(*fExt);
    
    
    MatrixReplacement LHS;
//    LHS.attachMyMatrix(*stiffnessMatrix);
    LHS.attachMyMatrix(mass_lumped.asDiagonal().inverse()*(-(*stiffnessMatrix)),mass_lumped,a,b,dt);
    
    if(!mass_calculated)
    {
        m_M.resize(mass_lumped.rows(),mass_lumped.rows());
//        Eigen::SparseMatrix<double,Eigen::RowMajor> M(mass_lumped.rows(),mass_lumped.rows());
        //    M.setZero();
        //    M += mass_lumped.asDiagonal();
        typedef Eigen::Triplet<double> T;
        std::vector<T> tripletList;
        tripletList.reserve(mass_lumped.rows());
        for(int i = 0; i < mass_lumped.rows(); i++)
        {
            tripletList.push_back(T(i,i,mass_lumped(i)));
        }
        m_M.setFromTriplets(tripletList.begin(),tripletList.end());
//        m_M = M;
        mass_calculated = true;
    }
    std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > m_Us;
    m_Us = generalizedEigenvalueProblemNegative((*stiffnessMatrix), m_M, 10, 0.00);
//
//    Eigen::VectorXd r = Eigen::VectorXd::Random(LHS.cols());
////    r.Random();
//
//    Eigen::VectorXd sol;
//    sol = LHS*r;
//
//    Eigen::saveMarketDat(*stiffnessMatrix,"stiffness.dat");
//    Eigen::saveMarketVectorDat(mass_lumped,"mass.dat");
//    Eigen::saveMarketDat(m_P,"m_P.dat");
//    Eigen::saveMarketVectorDat(r,"r.dat");
//    Eigen::saveMarketVectorDat(sol,"sol.dat");

    
//
//    q = m_P.transpose() * X.head(N);
//    qDot =  m_P.transpose() *( X.segment(N,N));
}

template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
using TimeStepperSIIMEX = TimeStepper<DataType, TimeStepperImplSIIMEXImpl<DataType, MatrixAssembler, VectorAssembler> >;

#endif /* TimeStepperSIIMEX_h */
