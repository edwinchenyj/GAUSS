//
//  TimeStepperIM.h
//  Gauss
//
//  Created by Yu Ju Edwin Chen on 2018-12-10.
//

#ifndef TimeStepperIM_h
#define TimeStepperIM_h


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


//TODO Solver Interface
namespace Gauss {
    
    //Given Initial state, step forward in time using linearly implicit Euler Integrator
    template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
    class TimeStepperImplIMImpl
    {
    public:
        
        
        template<typename Matrix>
        TimeStepperImplIMImpl(Matrix &P, double a, double b) {
            
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
            
            this->a = 0.0;
            this->b = b;
            
            //            // construct the selection matrix
            int m = P.cols() - P.rows();
            int n = P.cols();
            Eigen::VectorXd one = Eigen::VectorXd::Ones(P.rows());
            Eigen::MatrixXd P_col_sum =   one.transpose() * P;
            std::vector<Eigen::Triplet<DataType> > triplets;
            Eigen::SparseMatrix<DataType> S;
            //            std::cout<<"P_col_sum size: "<< P_col_sum.rows()<< " " << P_col_sum.cols()<<std::endl;
            S.resize(m,n);
            //
            
            inv_mass_calculated = false;
            mass_calculated = false;
            mass_lumped_calculated = false;
            
            
            unsigned int row_index =0;
            for(unsigned int col_index = 0; col_index < n; col_index++) {
                
                //add triplet into matrix
                //                std::cout<<col_index<<std::endl;
                if (P_col_sum(0,col_index) == 0) {
                    triplets.push_back(Eigen::Triplet<DataType>(row_index,col_index,1));
                    row_index+=1;
                }
                
            }
            
            S.setFromTriplets(triplets.begin(), triplets.end());
            
            m_S = S;
            //            std::cout<<"m_S sum: "<< m_S.sum()<<std::endl;
            
        }
        
        
        TimeStepperImplIMImpl(const TimeStepperImplIMImpl &toCopy) {
            
        }
        
        ~TimeStepperImplIMImpl() { }
        
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
        
        //        int flag = 0;
    protected:
        
        //num modes to correct
        unsigned int m_numModes;
        
        //        //Ratios diagonal matrix, stored as vector
        Eigen::VectorXd m_R;
        
        
        Eigen::SparseMatrix<DataType> inv_mass;
        Eigen::VectorXx<DataType> mass_lumped;
        Eigen::VectorXx<DataType> mass_lumped_inv;
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
        Eigen::SparseMatrix<DataType> m_S;
        
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
        
        SolverPardiso<Eigen::SparseMatrix<DataType, Eigen::RowMajor> > m_pardiso;
        SolverPardiso<Eigen::SparseMatrix<DataType, Eigen::RowMajor> > m_pardiso_test;
        SolverPardiso<Eigen::SparseMatrix<DataType, Eigen::RowMajor> > m_pardiso_mass;
        //        SolverPardiso<Eigen::SparseMatrix<DataType, Eigen::RowMajor> > m_pardiso_res;
#else
#endif
        
    private:
    };
}

template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
template<typename World>
void TimeStepperImplIMImpl<DataType, MatrixAssembler, VectorAssembler>::step(World &world, double dt, double t) {
    
    //First two lines work around the fact that C++11 lambda can't directly capture a member variable.
    MatrixAssembler &massMatrix = m_massMatrix;
    MatrixAssembler &stiffnessMatrix = m_stiffnessMatrix;
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
    
    
    do {
        std::cout<<"it outer: " << it_outer<<std::endl;
        it_outer = it_outer + 1;
        if (it_outer > 20) {
            std::cout<< "warning: quasi-newton more than 20 iterations." << std::endl;
        }
        
        eigen_q_temp = eigen_q_old + 1.0/4.0 * dt * (eigen_v_old + qDot);
        
        // set the state
        q = eigen_q_temp;
        
        //        //get mass matrix
        //        ASSEMBLEMATINIT(massMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
        //        ASSEMBLELIST(massMatrix, world.getSystemList(), getMassMatrix);
        //        ASSEMBLEEND(massMatrix);
        //
        
        
        
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
        
        (*stiffnessMatrix) = m_P*(*stiffnessMatrix)*m_P.transpose();
        
        (*forceVector) = m_P*(*forceVector);
        
        // add damping
        (*forceVector) = (*forceVector) -  (a * (m_M) + b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
        
        
        // add external force
        (*forceVector) = (*forceVector) + m_P*(*fExt);
        
        //setup RHS
        eigen_rhs = (m_M)*m_P*(qDot-eigen_v_old) - dt*(*forceVector);
        
        res_old = 1.0/2.0 * dt * dt * (m_MInv*(*forceVector)).squaredNorm();
        
        // std::cout<<"res_old: "<<res_old << std::endl;
        
        Eigen::VectorXd x0;
        // last term is damping
        Eigen::SparseMatrix<DataType, Eigen::RowMajor> systemMatrix = -(m_M)+ 1.0/4.0*dt*dt*(*m_stiffnessMatrix)- 1.0/2.0 *dt * (a *(m_M) + b * (*m_stiffnessMatrix));
        
        
#ifdef GAUSS_PARDISO
        
        m_pardiso.symbolicFactorization(systemMatrix);
        m_pardiso.numericalFactorization();
        
        //    SMW update for Eigenfit here
        
        m_pardiso.solve(eigen_rhs);
        x0 = m_pardiso.getX();
        
#else
        //solve system (Need interface for solvers but for now just use Eigen LLt)
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > solver;
        
        if(m_refactor || !m_factored) {
            solver.compute(systemMatrix);
        }
        
        if(solver.info()!=Eigen::Success) {
            // decomposition failed
            assert(1 == 0);
            std::cout<<"Decomposition Failed \n";
            exit(1);
        }
        
        if(solver.info()!=Eigen::Success) {
            // solving failed
            assert(1 == 0);
            std::cout<<"Solve Failed \n";
            exit(1);
        }
        
        
        x0 = solver.solve((eigen_rhs));
        
#endif
        
        auto Dv = m_P.transpose()*x0;
        //        std::cout<<"m_S*Dv"<< m_S*Dv <<std::endl;
        eigen_v_temp = qDot;
        eigen_v_temp = eigen_v_temp + Dv*step_size;
        //update state
        //        std::cout<<"m_S*eigen_v_temp"<< m_S*eigen_v_temp <<std::endl;
        
        q = eigen_q_old + 1.0/4.0 * dt*(eigen_v_temp + eigen_v_old) ;
        
        
        //        std::cout<<"q "<<q.rows()<< std::endl;
        
        // calculate the residual. brute force for now. ugly
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
        //        (*massMatrix) = m_P*(*massMatrix)*m_P.transpose();
        (*stiffnessMatrix) = m_P*(*stiffnessMatrix)*m_P.transpose();
        
        (*forceVector) = m_P*(*forceVector);
        
        (*forceVector) = (*forceVector) -  (a * (m_M) + b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
        
        
        // add external force
        (*forceVector) = (*forceVector) + m_P*(*fExt);
        
        //        m_pardiso_mass.solve(*forceVector);
        std::cout << "res: " << 1.0/2.0 * (m_P*(eigen_v_temp - eigen_v_old) - dt*m_MInv*(*forceVector)).squaredNorm()<< std::endl;
        res  = 1.0/2.0 * (m_P*(eigen_v_temp - eigen_v_old) - dt*m_MInv*(*forceVector)).squaredNorm();
        
        qDot = eigen_v_temp;
        q = eigen_q_old + 1.0/2.0 * dt * (qDot + eigen_v_old);
        
    } while(res > 1e-6);
    it_outer = 0;
    //    std::cout<<"m_S*qDot"<< m_S*qDot <<std::endl;
    
    q = eigen_q_old + 1.0/2.0 * dt * (qDot + eigen_v_old);
}

template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
using TimeStepperIM = TimeStepper<DataType, TimeStepperImplIMImpl<DataType, MatrixAssembler, VectorAssembler> >;




#endif /* TimeStepperIM_h */
