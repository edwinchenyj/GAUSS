//
//  TimeStepperEigenFitSI.h
//  Gauss
//
//  Created by Edwin Chen on 2018-05-15.
//
//

#ifndef TimeStepperEigenFitSI_h
#define TimeStepperEigenFitSI_h



#include <World.h>
#include <Assembler.h>
#include <TimeStepper.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <UtilitiesEigen.h>
#include <UtilitiesMATLAB.h>
#include <Eigen/SparseCholesky>
#include <SolverPardiso.h>
#include <EigenFit.h>
#include <limits>
#include <Eigen/Eigenvalues>

#include <unsupported/Eigen/MatrixFunctions>

#include <math.h>
#include <algorithm>

//TODO Solver Interface
namespace Gauss {
    
    //Given Initial state, step forward in time using linearly implicit Euler Integrator
    template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
    class TimeStepperImplEigenFitSIImpl
    {
    public:
        
        template<typename Matrix>
        TimeStepperImplEigenFitSIImpl(Matrix P, Matrix P2, int constraint_switch, unsigned int num_modes, double a = 0.0, double b = -1e-2, std::string integrator = "SI") {
            
            this->integrator = integrator;
            
            m_num_modes = (num_modes);
            
            m_P = P;
            m_P2 = P2;
            
            m_constraint_switch = constraint_switch;
            
            
            m_factored = false;
            // refactor for every solve
            m_refactor = true;
            
            // init residual
            res = std::numeric_limits<double>::infinity();
            
            
            this->a = a;
            this->b = b;
            
            Dv.resize(P.rows());
            
            stiffness_calculated = false;
            islinear = false;

            
            eigenfit_damping = true;
            
            
        }
        
        TimeStepperImplEigenFitSIImpl(const TimeStepperImplEigenFitSIImpl &toCopy) {
            
        }
        
        ~TimeStepperImplEigenFitSIImpl() { delete rhs;
            delete v_old;
            delete v_temp;
            delete q_old;
            delete q_temp;
        }
        
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
        
        bool matrix_fix_flag = true;
        
        // store stiffness if linear
        Eigen::SparseMatrix<double,Eigen::RowMajor> stiffness;
        
        
        Eigen::VectorXd Dv;
        
        
        std::string integrator;
        
        bool islinear;
        bool stiffness_calculated;
        bool eigenfit_damping;
        
        int m_constraint_switch;
    protected:
        
        //num modes to correct
        unsigned int m_num_modes;
        
        
        
        //Subspace Eigenvectors and eigenvalues from this coarse mesh
        std::pair<Eigen::MatrixXx<DataType>, Eigen::VectorXx<DataType> > m_coarseUs;
        // for SMW
        Eigen::MatrixXd Y;
        Eigen::MatrixXd Z;
        
        MatrixAssembler m_massMatrix;
        MatrixAssembler m_stiffnessMatrix;
        VectorAssembler m_forceVector;
        VectorAssembler m_fExt;
        
        
        Eigen::SparseMatrix<double> m_P;
        Eigen::SparseMatrix<double> m_P2;
        Eigen::SparseMatrix<double> m_S;
        
        //storage for lagrange multipliers
        typename VectorAssembler::MatrixType m_lagrangeMultipliers;
        
        bool m_factored, m_refactor;
        
        // iteration counter
        // residual
        double res, res_old, c1, c2;
        
        
        SolverPardiso<Eigen::SparseMatrix<DataType, Eigen::RowMajor> > m_pardiso;
        
    private:
    };
}




template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
template<typename World>
void TimeStepperImplEigenFitSIImpl<DataType, MatrixAssembler, VectorAssembler>::step(World &world, double dt, double t)
{
    
    
    int step_number =static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->step_number;
    
    
    MatrixAssembler &massMatrix = m_massMatrix;
    MatrixAssembler &stiffnessMatrix = m_stiffnessMatrix;
    VectorAssembler &forceVector = m_forceVector;
    VectorAssembler &fExt = m_fExt;
    
    
    ASSEMBLEMATINIT(massMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
    ASSEMBLELIST(massMatrix, world.getSystemList(), getMassMatrix);
    ASSEMBLEEND(massMatrix);
    
    if (step_number > m_constraint_switch && m_constraint_switch != -1) {
        m_P = m_P2;
    }
    
    (*massMatrix) = m_P*(*massMatrix)*m_P.transpose();
    
    
    //        if (step_number < constraint_switch) {
    // init rhs
    rhs = new double[m_P.rows()];
    Eigen::Map<Eigen::VectorXd> eigen_rhs(rhs,m_P.rows());
    eigen_rhs.setZero();
    // velocity from previous step (for calculating residual)
    v_old = new double[world.getNumQDotDOFs()];
    Eigen::Map<Eigen::VectorXd> eigen_v_old(v_old,world.getNumQDotDOFs());
    eigen_v_old.setZero();
    
    // velocity from previous step (for calculating residual)
    v_temp = new double[world.getNumQDotDOFs()];
    Eigen::Map<Eigen::VectorXd> eigen_v_temp(v_temp,world.getNumQDotDOFs());
    eigen_v_temp.setZero();
    
    q_old = new double[world.getNumQDotDOFs()];
    Eigen::Map<Eigen::VectorXd> eigen_q_old(q_old,world.getNumQDotDOFs());
    eigen_q_old.setZero();
    
    q_temp = new double[world.getNumQDotDOFs()];
    Eigen::Map<Eigen::VectorXd> eigen_q_temp(q_temp,world.getNumQDotDOFs());
    eigen_q_temp.setZero();
    
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
    
    if(!islinear || !stiffness_calculated)
    {
        //get stiffness matrix
        ASSEMBLEMATINIT(stiffnessMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
        ASSEMBLELIST(stiffnessMatrix, world.getSystemList(), getStiffnessMatrix);
        ASSEMBLELIST(stiffnessMatrix, world.getForceList(), getStiffnessMatrix);
        ASSEMBLEEND(stiffnessMatrix);
        
        
        //constraint Projection
        (*stiffnessMatrix) = m_P*(*stiffnessMatrix)*m_P.transpose();
        
        stiffness_calculated = true;
        stiffness = (*stiffnessMatrix);
    }
    
    if(islinear)
    {
        (*stiffnessMatrix) = stiffness;
    }
    //Need to filter internal forces seperately for this applicat
    ASSEMBLEVECINIT(forceVector, world.getNumQDotDOFs());
    ASSEMBLELIST(forceVector, world.getSystemList(), getImpl().getInternalForce);
    ASSEMBLEEND(forceVector);
    
    ASSEMBLEVECINIT(fExt, world.getNumQDotDOFs());
    ASSEMBLELIST(fExt, world.getSystemList(), getImpl().getBodyForce);
    ASSEMBLEEND(fExt);
    
    (*forceVector) = m_P*(*forceVector);
    
    
    //Eigendecomposition
    // if number of modes not equals to 0, use EigenFit
    if (m_num_modes != 0 ) {
            static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->calculateEigenFitData(q,m_massMatrix,stiffnessMatrix,m_coarseUs,Y,Z);
        
        
        //    Correct Forces
        (*forceVector) = (*forceVector) + Y*m_coarseUs.first.transpose()*(*forceVector);
        
        
        if(eigenfit_damping)
        {
            (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P *(qDot) + b * (Y*(Z * (m_P *qDot)));
        }
        else{
            (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P *(qDot);
        }
        
    }
    else
    {
        (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P *(qDot);
        
    }
    // add external force
    (*forceVector) = (*forceVector) + m_P*(*fExt);
    
    //setup RHS
    eigen_rhs = (*m_massMatrix)*m_P*(qDot-eigen_v_old) - dt*(*forceVector);
    
    
    Eigen::VectorXd x0;
    // last term is damping
    Eigen::SparseMatrix<DataType, Eigen::RowMajor> systemMatrix;
    
    systemMatrix = -(*m_massMatrix) + dt*dt*(*m_stiffnessMatrix) - dt * (a *(*m_massMatrix) + b * (*m_stiffnessMatrix));
    
    
    m_pardiso.symbolicFactorization(systemMatrix, m_num_modes);
    m_pardiso.numericalFactorization();
    
    m_pardiso.solve(eigen_rhs);
    x0 = m_pardiso.getX();
    
    if(!matrix_fix_flag )
    {
        cout<<"Ignoring change in stiffness matrix from EigenFit"<<endl;
    }
    else
    {
        Y = dt*Y;
        Z = dt*Z;
        m_pardiso.solve(Y);
        Eigen::MatrixXd APrime = Z*m_pardiso.getX();
        Eigen::VectorXd bPrime = Y*(Eigen::MatrixXd::Identity(m_num_modes,m_num_modes) + APrime).ldlt().solve(Z*x0);
        m_pardiso.solve(bPrime);
        
        x0 -= m_pardiso.getX();
        
        
    }
    
    Dv = m_P.transpose()*x0;
    
    eigen_v_temp = qDot;
    eigen_v_temp = eigen_v_temp + Dv;
    
    qDot = eigen_v_temp;
    
    
    q = eigen_q_old + dt * (qDot);
    
    
    
    
    
}

template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
using TimeStepperEigenFitSI = TimeStepper<DataType, TimeStepperImplEigenFitSIImpl<DataType, MatrixAssembler, VectorAssembler> >;







#endif /* TimeStepperEigenFitSI_h */
