//
//  TimeStepperSI.h
//  Gauss
//
//  Created by Yu Ju Edwin Chen on 2018-12-05.
//

#ifndef TimeStepperSI_h
#define TimeStepperSI_h



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


using Eigen::SparseMatrix;

//TODO Solver Interface
namespace Gauss {
    
    //Given Initial state, step forward in time using linearly implicit Euler Integrator
    template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
    class TimeStepperImplSIImpl
    {
    public:
        
        template<typename Matrix>
        TimeStepperImplSIImpl(Matrix &P, double a, double b) {
            
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
        
        TimeStepperImplSIImpl(const TimeStepperImplSIImpl &toCopy) {
            
        }
        
        ~TimeStepperImplSIImpl() { }
        
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
        MatrixAssembler m_stiffnessMatrix;
        MatrixAssembler m_J;
        VectorAssembler m_forceVector;
        
        VectorAssembler m_fExt;
        
        
        Eigen::SparseMatrix<DataType> m_P;
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
void TimeStepperImplSIImpl<DataType, MatrixAssembler, VectorAssembler>::step(World &world, double dt, double t) {
    
    // TODO: should not be here... set the rayleigh damping parameter
    //    a = (std::get<0>(world.getSystemList().getStorage())[0])->a;
    //    b = (std::get<0>(world.getSystemList().getStorage())[0])->b;
    
    MatrixAssembler &stiffnessMatrix = m_stiffnessMatrix;
    MatrixAssembler &massMatrix = m_massMatrix;
    VectorAssembler &forceVector = m_forceVector;
    VectorAssembler &fExt = m_fExt;
    
    
    // init rhs
    rhs = new double[m_P.rows()];
    Eigen::Map<Eigen::VectorXd> eigen_rhs(rhs,m_P.rows());
    
    // velocity from previous step (for calculating residual)
    v_old = new double[world.getNumQDotDOFs()];
    Eigen::Map<Eigen::VectorXd> eigen_v_old(v_old,world.getNumQDotDOFs());
    
    // velocity from previous step (for calculating residual)
    v_temp = new double[world.getNumQDotDOFs()];
    Eigen::Map<Eigen::VectorXd> eigen_v_temp(v_temp,world.getNumQDotDOFs());
    
    q_old = new double[world.getNumQDotDOFs()];
    Eigen::Map<Eigen::VectorXd> eigen_q_old(q_old,world.getNumQDotDOFs());
    
    q_temp = new double[world.getNumQDotDOFs()];
    Eigen::Map<Eigen::VectorXd> eigen_q_temp(q_temp,world.getNumQDotDOFs());
    
    
    //Grab the state
    Eigen::Map<Eigen::VectorXd> q = mapStateEigen<0>(world);
    Eigen::Map<Eigen::VectorXd> qDot = mapStateEigen<1>(world);
    
    // this is the velocity and q from the end of the last time step
    for (int ind = 0; ind < qDot.rows(); ind++) {
        eigen_v_old(ind) = qDot(ind);
        eigen_q_old(ind) = q(ind);
        eigen_v_temp(ind) = qDot(ind);
    }
    
        eigen_q_temp = eigen_q_old + dt * (qDot);
    
        // set the state
        q = eigen_q_temp;
        
    
    
    
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
        mass_lumped_calculated = true;
    }
    
    (*stiffnessMatrix) = m_P*(*stiffnessMatrix)*m_P.transpose();
    
    (*forceVector) = m_P*(*forceVector);
    
    // add damping
    (*forceVector) = (*forceVector) -  (a * (*massMatrix) + b*(*stiffnessMatrix)) * m_P * ( qDot);
    
    // add external force
    (*forceVector) = (*forceVector) + m_P*(*fExt);
    
    
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
    
    
    //setup RHS
    eigen_rhs = (*massMatrix)*m_P*(qDot-eigen_v_old) - dt*(*forceVector);
    
    // std::cout<<"res_old: "<<res_old << std::endl;
    
    Eigen::VectorXd x0;
    // last term is damping
    Eigen::SparseMatrix<DataType, Eigen::RowMajor> systemMatrix = -(m_M)+ dt*dt*(*m_stiffnessMatrix)- dt * (a *(m_M) + b * (*m_stiffnessMatrix));
    
#ifdef GAUSS_PARDISO
    
    m_pardiso.symbolicFactorization(systemMatrix);
    m_pardiso.numericalFactorization();
    
    m_pardiso.solve(eigen_rhs);
    x0 = m_pardiso.getX();
    
#else
#endif
    auto Dv = m_P.transpose()*x0;
    //        std::cout<<"m_S*Dv"<< m_S*Dv <<std::endl;
    eigen_v_temp = qDot;
    eigen_v_temp = eigen_v_temp + Dv*step_size;
    //update state
    //        std::cout<<"m_S*eigen_v_temp"<< m_S*eigen_v_temp <<std::endl;
    
//    q = eigen_q_old + dt*eigen_v_temp;
    
    qDot = eigen_v_temp;
    q = eigen_q_old + dt * (qDot);

    
}

template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
using TimeStepperSI = TimeStepper<DataType, TimeStepperImplSIImpl<DataType, MatrixAssembler, VectorAssembler> >;

#endif /* TimeStepperSI_h */
