//
//  TimeStepperEigenFitCoRotBE.h
//  Gauss
//
//  Created by Yu Ju Edwin Chen on 2018-12-13.
//

#ifndef TimeStepperEigenFitCoRotBE_h
#define TimeStepperEigenFitCoRotBE_h

#include <World.h>
#include <Assembler.h>
#include <TimeStepper.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <UtilitiesEigen.h>
#include <UtilitiesMATLAB.h>
#include <Eigen/SparseCholesky>
#include <SolverPardiso.h>
#include <EigenFitCoRot.h>
#include <limits>


//TODO Solver Interface
namespace Gauss {
    
    //Given Initial state, step forward in time using linearly implicit Euler Integrator
    template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
    class TimeStepperImplEigenFitCoRotBEImpl
    {
    public:
        
        template<typename Matrix>
        TimeStepperImplEigenFitCoRotBEImpl(Matrix &P, unsigned int numModes) {
            
            m_numModes = (numModes);
            
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
            
            simple_mass_flag = false;
            
        }
        
        TimeStepperImplEigenFitCoRotBEImpl(const TimeStepperImplEigenFitCoRotBEImpl &toCopy) {
            
        }
        
        ~TimeStepperImplEigenFitCoRotBEImpl() { }
        
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
        
        bool matrix_fix_flag = false;
        
        bool simple_mass_flag;
        bool mass_calculated;
        Eigen::VectorXx<double> mass_lumped;
        Eigen::VectorXx<double> mass_lumped_inv;
        
        Eigen::SparseMatrix<double,Eigen::RowMajor> MinvK;
        Eigen::SparseMatrix<DataType,Eigen::RowMajor> m_M;
        
        double update_step_size;
        Eigen::VectorXd update_step;
        Eigen::VectorXd prev_update_step;
    protected:
        
        //num modes to correct
        unsigned int m_numModes;
        
        //        //Ratios diagonal matrix, stored as vector
        Eigen::VectorXd m_R;
        
        
        //Subspace Eigenvectors and eigenvalues from the embedded fine mesh
        std::pair<Eigen::MatrixXx<DataType>, Eigen::VectorXx<DataType> > m_fineUs;
        //Subspace Eigenvectors and eigenvalues from this coarse mesh
        std::pair<Eigen::MatrixXx<DataType>, Eigen::VectorXx<DataType> > m_coarseUs;
        // for SMW
        Eigen::MatrixXd Y;
        Eigen::MatrixXd Z;
        
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
void TimeStepperImplEigenFitCoRotBEImpl<DataType, MatrixAssembler, VectorAssembler>::step(World &world, double dt, double t) {
    
    // TODO: should not be here... set the rayleigh damping parameter
    a = static_cast<EigenFitCoRot*>(std::get<0>(world.getSystemList().getStorage())[0])->a;
    b = static_cast<EigenFitCoRot*>(std::get<0>(world.getSystemList().getStorage())[0])->b;
    
    simple_mass_flag = static_cast<EigenFitCoRot*>(std::get<0>(world.getSystemList().getStorage())[0])->simple_mass_flag;
    
    //First two lines work around the fact that C++11 lambda can't directly capture a member variable.
    MatrixAssembler &massMatrix = m_massMatrix;
    MatrixAssembler &stiffnessMatrix = m_stiffnessMatrix;
    VectorAssembler &forceVector = m_forceVector;
    VectorAssembler &fExt = m_fExt;
    
    if(simple_mass_flag && !mass_calculated)
    {
        
        //get mass matrix
        ASSEMBLEMATINIT(massMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
        ASSEMBLELIST(massMatrix, world.getSystemList(), getMassMatrix);
        ASSEMBLEEND(massMatrix);
        
        (*massMatrix) = m_P*(*massMatrix)*m_P.transpose();
        
        Eigen::VectorXx<double> ones(m_P.rows());
        ones.setOnes();
        mass_lumped.resize(m_P.rows());
        mass_lumped_inv.resize(m_P.rows());
        mass_lumped = ((*massMatrix)*ones);
        mass_lumped_inv = mass_lumped.cwiseInverse();
        
        
        m_M.resize(mass_lumped.rows(),mass_lumped.rows());
        typedef Eigen::Triplet<double> T;
        std::vector<T> tripletList;
        tripletList.reserve(mass_lumped.rows());
        for(int i = 0; i < mass_lumped.rows(); i++)
        {
            tripletList.push_back(T(i,i,mass_lumped(i)));
        }
        m_M.setFromTriplets(tripletList.begin(),tripletList.end());
        
        (*massMatrix) = m_M;
        (*m_massMatrix) = m_M;
        mass_calculated = true;
    }
    else
    {
        //get mass matrix
        ASSEMBLEMATINIT(massMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
        ASSEMBLELIST(massMatrix, world.getSystemList(), getMassMatrix);
        ASSEMBLEEND(massMatrix);
        
        (*massMatrix) = m_P*(*massMatrix)*m_P.transpose();
    }
    
    if(simple_mass_flag)
    {
        (*massMatrix) = m_M;
        (*m_massMatrix) = m_M;
    }
    
    
    
    
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
    
    prev_update_step = dt * (qDot);

    
    cout<<"Newton iteration for implicit backward Euler..."<<endl;
    do {
        std::cout<<"Newton it outer: " << it_outer<<std::endl;
        it_outer = it_outer + 1;
        if (it_outer > 20) {
            std::cout<< "warning: quasi-newton more than 20 iterations." << std::endl;
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
        
        //constraint Projection
        (*stiffnessMatrix) = m_P*(*stiffnessMatrix)*m_P.transpose();
        
        (*forceVector) = m_P*(*forceVector);
        
        
        //Eigendecomposition
        
        // if number of modes not equals to 0, use EigenFit
        if (m_numModes != 0 && static_cast<EigenFitCoRot*>(std::get<0>(world.getSystemList().getStorage())[0])->ratio_recalculation_switch != 6) {
            
            try{
                if(static_cast<EigenFitCoRot*>(std::get<0>(world.getSystemList().getStorage())[0])->calculateEigenFitData(q,massMatrix,stiffnessMatrix,m_coarseUs,Y,Z)) throw 1;
            }catch(...)
            {
                std::cout<<"hausdorff distance check fail\n";
                static_cast<EigenFitCoRot*>(std::get<0>(world.getSystemList().getStorage())[0])->flag = 2;
                return;
            }
            
            //    Correct Forces
            (*forceVector) = (*forceVector) + Y*m_coarseUs.first.transpose()*(*forceVector);
            
            
            // add damping
            (*forceVector) = (*forceVector) -  (a * (*massMatrix) + b*(*stiffnessMatrix)) * m_P * (qDot);
            //            (*forceVector) = (*forceVector) -  (a * (*massMatrix) + b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot) + b * (Y * (Z * (m_P * 1.0 / 2.0 *(eigen_v_old + qDot))));
            
        }
        else
        {
            // add damping
            (*forceVector) = (*forceVector) -  (a * (*massMatrix) + b*(*stiffnessMatrix)) * m_P * (qDot);
        }
        // add external force
        (*forceVector) = (*forceVector) + m_P*(*fExt);
        
        
        
        //setup RHS
        eigen_rhs = (*massMatrix)*m_P*(qDot-eigen_v_old) - dt*(*forceVector);
        
        m_pardiso_mass.symbolicFactorization(*massMatrix);
        m_pardiso_mass.numericalFactorization();
        m_pardiso_mass.solve(*forceVector);
        
        //        res_old = 1.0/2.0 * dt * dt * ((m_pardiso_mass.getX()).transpose() * (m_pardiso_mass.getX()));
        res_old = 1.0/2.0 * dt * dt * ((m_pardiso_mass.getX()).transpose()).squaredNorm();
        
        //            std::cout<<"res_old: "<<res_old << std::endl;
        //        eigen_v_old = qDot;
        
        //            std::cout<< "qDot - eigen_v_old: "<<(qDot - eigen_v_old).norm() << std::endl;
        
        Eigen::VectorXd x0;
        // last term is damping
        Eigen::SparseMatrix<DataType, Eigen::RowMajor> systemMatrix;
        if(!simple_mass_flag)
        {
            systemMatrix = -(*m_massMatrix) + dt*dt*(*m_stiffnessMatrix) - dt * (a *(*m_massMatrix) + b * (*m_stiffnessMatrix));
        }
        else
        {
            systemMatrix = -(m_M) + dt*dt*(*m_stiffnessMatrix) - dt * (a *(*m_massMatrix) + b * (*m_stiffnessMatrix));
        }
#ifdef GAUSS_PARDISO
        
        m_pardiso.symbolicFactorization(systemMatrix, m_numModes);
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
        if(!matrix_fix_flag && m_numModes != 0 && static_cast<EigenFitCoRot*>(std::get<0>(world.getSystemList().getStorage())[0])->ratio_recalculation_switch != 6)
        {
            cout<<"Ignoring change in stiffness matrix from EigenFit"<<endl;
        }
        else
        {
            //
            if (m_numModes != 0 && static_cast<EigenFitCoRot*>(std::get<0>(world.getSystemList().getStorage())[0])->ratio_recalculation_switch != 6) {
                
                //            Y = (-1.0/4.0*dt*dt)*Y;
                Y = (-dt*dt+dt*b)*Y;
                // Y = (1.0/4.0*dt*dt)*Y;
                //            Z = 1/2*dt*Z;
                
#ifdef GAUSS_PARDISO
                
                m_pardiso.solve(Y);
                Eigen::MatrixXd APrime = Z*m_pardiso.getX();
                Eigen::VectorXd bPrime = Y*(Eigen::MatrixXd::Identity(m_numModes,m_numModes) + APrime).ldlt().solve(Z*x0);
                
                
#else
                Eigen::VectorXd bPrime = Y*(Eigen::MatrixXd::Identity(m_numModes,m_numModes) + Z*solver.solve(Y)).ldlt().solve(Z*x0);
                
#endif
                
                
#ifdef GAUSS_PARDISO
                
                m_pardiso.solve(bPrime);
                
                x0 -= m_pardiso.getX();
                
                m_pardiso.cleanup();
#else
                
                x0 -= solver.solve(bPrime);
                
#endif
            }
        }
        //        qDot = m_P.transpose()*x0;
        
        auto Dv = m_P.transpose()*x0;
        
        eigen_v_temp = qDot;
        eigen_v_temp = eigen_v_temp + Dv*step_size;
        //update state
        q = eigen_q_old + dt*(eigen_v_temp);
        
        
        //        std::cout<<"q "<<q.rows()<< std::endl;
        
        cout<<" calculate the residual."<<endl;  //brute force for now. ugly
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
        
        if (m_numModes != 0 && static_cast<EigenFitCoRot*>(std::get<0>(world.getSystemList().getStorage())[0])->ratio_recalculation_switch != 6) {
            
            //
            //
            //            static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->calculateEigenFitData(q,massMatrix,stiffnessMatrix,m_coarseUs,Y,Z);
            
            //    Correct Forces
            (*forceVector) = (*forceVector) + Y*m_coarseUs.first.transpose()*(*forceVector);
            
            // add damping
            (*forceVector) = (*forceVector) -  (a * (*massMatrix) + b*(*stiffnessMatrix)) * m_P * (qDot);
            //            (*forceVector) = (*forceVector) -  (a * (*massMatrix) + b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot) + b * (Y * (Z * (m_P * 1.0 / 2.0 *(eigen_v_old + qDot))));
            //             (*forceVector) = (*forceVector) -  (a * (*massMatrix) + b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
            
        }
        else
        {
            (*forceVector) = (*forceVector) -  (a * (*massMatrix) + b*(*stiffnessMatrix)) * m_P * (qDot);
            
        }
        // add external force
        (*forceVector) = (*forceVector) + m_P*(*fExt);
        
        m_pardiso_mass.solve(*forceVector);
        std::cout << "res: " << 1.0/2.0 * (m_P*(eigen_v_temp - eigen_v_old) - dt*m_pardiso_mass.getX()).squaredNorm()<< std::endl;
        res  = 1.0/2.0 * (m_P*(eigen_v_temp - eigen_v_old) - dt*m_pardiso_mass.getX()).squaredNorm();
        
        qDot = eigen_v_temp;
        q = eigen_q_old + dt * (qDot );
        
        update_step_size = (dt * (qDot) - prev_update_step).norm();
        prev_update_step = dt * (qDot);
        cout<<" step size: " << update_step_size<<endl;
    } while(res > 1e-6  && update_step_size > 1e-3);
    it_outer = 0;
    q = eigen_q_old + dt * (qDot );
}

template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
using TimeStepperEigenFitCoRotBE = TimeStepper<DataType, TimeStepperImplEigenFitCoRotBEImpl<DataType, MatrixAssembler, VectorAssembler> >;


#endif /* TimeStepperEigenFitCoRotBE_h */
