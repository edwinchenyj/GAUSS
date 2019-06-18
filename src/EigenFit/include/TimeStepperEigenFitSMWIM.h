//
//  TimeStepperEigenFitSMWIM.h
//  Gauss
//
//  Created by Edwin Chen on 2018-05-15.
//
//

#ifndef TimeStepperEigenFitSMWIM_h
#define TimeStepperEigenFitSMWIM_h



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
#include <ExponentialIMEX.h>
#include <Eigen/Eigenvalues>

#include <unsupported/Eigen/MatrixFunctions>

#include <math.h>
#include <algorithm>

//TODO Solver Interface
namespace Gauss {
    
    //Given Initial state, step forward in time using linearly implicit Euler Integrator
    template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
    class TimeStepperImplEigenFitSMWIMImpl
    {
    public:
        
        template<typename Matrix>
        TimeStepperImplEigenFitSMWIMImpl(Matrix P, Matrix P2, int constraint_switch, unsigned int num_modes, double a = 0.0, double b = -1e-2, std::string integrator = "IM") {
            
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
            
            
            
            m_M.resize(P.cols(),P.cols());
            m_M.reserve(P.cols()); //simple mass
            mass_lumped.resize(P.cols());
            mass_lumped_inv.resize(P.cols());
            
            //            (*m_massMatrix).resize(P.rows(),P.rows());
            
            it_outer = 0;
            it_inner = 0;
            // constants from Nocedal and Wright
            step_size = 1;
            c1 = 1e-4;
            c2 = 0.9;
            
            simple_mass_flag = false;
            mass_calculated = false;
            mass_factorized = false;
            
            this->a = a;
            this->b = b;
            
            step_success = true;
            Dv.resize(P.rows());
            
            stiffness_calculated = false;
            islinear = false;
#ifdef LINEAR
            islinear = true;
            stiffness.resize(P.rows(),P.rows());
#endif
            mass.resize(P.cols(),P.cols());
            eigenfit_damping = true;
            
            
        }
        
        TimeStepperImplEigenFitSMWIMImpl(const TimeStepperImplEigenFitSMWIMImpl &toCopy) {
            
        }
        
        ~TimeStepperImplEigenFitSMWIMImpl() { delete rhs;
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
        
        bool matrix_fix_flag = false;
        
        bool simple_mass_flag;
        bool mass_calculated;
        bool mass_factorized;
        Eigen::VectorXx<double> mass_lumped;
        Eigen::VectorXx<double> mass_lumped_inv;
        
        Eigen::SparseMatrix<DataType> inv_mass;
        Eigen::SparseMatrix<double,Eigen::RowMajor> MinvK;
        Eigen::SparseMatrix<double,Eigen::RowMajor> m_M;
        Eigen::SparseMatrix<double,Eigen::RowMajor> stiffness;
        Eigen::SparseMatrix<double,Eigen::RowMajor> mass;
        
        double update_step_size;
        Eigen::VectorXd update_step;
        Eigen::VectorXd prev_update_step;
        
        Eigen::VectorXd Dv;
        
        bool step_success;
        
        std::string integrator;
        
        bool islinear;
        bool stiffness_calculated;
        bool eigenfit_damping;
        
        int m_constraint_switch;
    protected:
        
        //num modes to correct
        unsigned int m_num_modes;
        
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
        
        
        Eigen::SparseMatrix<double> m_P;
        Eigen::SparseMatrix<double> m_P2;
        Eigen::SparseMatrix<double> m_S;
        
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
        SolverPardiso<Eigen::SparseMatrix<DataType, Eigen::RowMajor> > m_pardiso_mass;
        //        SolverPardiso<Eigen::SparseMatrix<DataType, Eigen::RowMajor> > m_pardiso_res;
#else
#endif
        
    private:
    };
}




template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
template<typename World>
void TimeStepperImplEigenFitSMWIMImpl<DataType, MatrixAssembler, VectorAssembler>::step(World &world, double dt, double t)
{
    
    simple_mass_flag = static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->simple_mass_flag;
    int step_number =static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->step_number;
    
    if (integrator.compare("IM")==0 || integrator.compare("BE")==0 || integrator.compare("SI")==0)
    {
        
        
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
            mass =  (*massMatrix);
       
        
        
        
        if(simple_mass_flag)
            //        if(simple_mass_flag && !mass_calculated)
        {
            
            Eigen::VectorXx<double> ones((*massMatrix).rows());
            ones.setOnes();
            mass_lumped.resize((*massMatrix).rows());
            mass_lumped_inv.resize((*massMatrix).rows());
            mass_lumped = ((*massMatrix)*ones);
            mass_lumped_inv = mass_lumped.cwiseInverse();
            
            
            m_M.resize((*massMatrix).rows(),(*massMatrix).rows());
            m_M.setZero();
            m_M.reserve((*massMatrix).rows());
            typedef Eigen::Triplet<double> T;
            std::vector<T> tripletList;
            tripletList.reserve((*massMatrix).rows());
            for(int i = 0; i < (*massMatrix).rows(); i++)
            {
                tripletList.push_back(T(i,i,mass_lumped(i)));
            }
            m_M.setFromTriplets(tripletList.begin(),tripletList.end());
            
            mass_calculated = false;
        }
        
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
        
        if (integrator.compare("IM") == 0) {
            prev_update_step = 1.0/2.0 * dt * (eigen_v_old + qDot);
        }
        else{ //BE or SI
            prev_update_step = eigen_q_old + dt*qDot;
        }
        
        
        
        //    cout<<"Newton iteration for implicit midpoint..."<<endl;
        do {
            std::cout<<"Newton it outer: " << it_outer<< ", ";
            it_outer = it_outer + 1;
            if (it_outer > 20) {
                std::cout<< "warning: quasi-newton more than 20 iterations." << std::endl;
                q = eigen_q_old;
                qDot = eigen_v_old;
                step_success = false;
                return;
            }
            
            if (integrator.compare("IM") == 0) {
                eigen_q_temp = eigen_q_old + 1.0/4.0 * dt * (eigen_v_old + qDot);
            }
            else {
                eigen_q_temp = eigen_q_old + dt * (qDot);
                
            }
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
            
            if(!simple_mass_flag)
            {
                //Eigendecomposition
                // if number of modes not equals to 0, use EigenFit
                if (m_num_modes != 0 ) {
                    
                    if (it_outer == 1) {
                        static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->calculateEigenFitData(q,m_massMatrix,stiffnessMatrix,m_coarseUs,Y,Z);
                    }
                    
                    //    Correct Forces
                    (*forceVector) = (*forceVector) + Y*m_coarseUs.first.transpose()*(*forceVector);
                    
                    // add damping
                    //                    (*forceVector) = (*forceVector) -  ( b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
                    
                    if(eigenfit_damping)
                    {
                        if (integrator.compare("IM") == 0) {
                            (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot) + b * (Y*(Z * (m_P *( 1.0 / 2.0 *(eigen_v_old + qDot)))));
                        }
                        else {
                            (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P *(qDot) + b * (Y*(Z * (m_P *qDot)));
                        }
                    }
                    else{
                        // add damping
                        if (integrator.compare("IM") == 0) {
                            (*forceVector) = (*forceVector) -  ( b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
                        }
                        else
                        {
                            (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P *(qDot);
                        }
                    }
                    
                }
                else
                {
                    // add damping
                    if (integrator.compare("IM") == 0) {
                        (*forceVector) = (*forceVector) -  ( b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
                    }
                    else
                    {
                        (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P *(qDot);
                    }
                    
                }
                // add external force
                (*forceVector) = (*forceVector) + m_P*(*fExt);
                
                //setup RHS
                eigen_rhs = (*m_massMatrix)*m_P*(qDot-eigen_v_old) - dt*(*forceVector);
                
                m_pardiso_mass.symbolicFactorization(*m_massMatrix);
                m_pardiso_mass.numericalFactorization();
                mass_factorized = true;
                
                m_pardiso_mass.solve(*forceVector);
                
                res_old = 1.0/2.0 * (m_P*(eigen_v_temp - eigen_v_old) - dt*m_pardiso_mass.getX()).squaredNorm();
                cout<<"res old: "<< res_old << ", ";
                
                Eigen::VectorXd x0;
                // last term is damping
                Eigen::SparseMatrix<DataType, Eigen::RowMajor> systemMatrix;
                
                if (integrator.compare("IM") == 0) {
                    systemMatrix = -(*m_massMatrix) + 1.0/4.0* dt*dt*(*m_stiffnessMatrix) - 1.0/2.0 * dt * (a *(*m_massMatrix) + b * (*m_stiffnessMatrix));
                }
                else{
                    systemMatrix = -(*m_massMatrix) + dt*dt*(*m_stiffnessMatrix) - dt * (a *(*m_massMatrix) + b * (*m_stiffnessMatrix));
                    
                }
#ifdef GAUSS_PARDISO
                
                m_pardiso.symbolicFactorization(systemMatrix, m_num_modes);
                m_pardiso.numericalFactorization();
                
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
                if(!matrix_fix_flag )
                {
                    //                cout<<"Ignoring change in stiffness matrix from EigenFit"<<endl;
                }
                else if(m_num_modes != 0)
                {
                    cout<<"Warning: stiffness fix not implemented"<<endl;
                }
                
                Dv = m_P.transpose()*x0;
                
                eigen_v_temp = qDot;
                eigen_v_temp = eigen_v_temp + Dv*step_size;
                //update state
                if (integrator.compare("IM") == 0) {
                    q = eigen_q_old + 1.0/4.0 * dt*(eigen_v_temp + eigen_v_old);
                }
                else{
                    q = eigen_q_old +  dt*(eigen_v_temp);
                }
                //            warning: if using stiffness from the beginning of the step for rayleigh damping, no recalculation needed (explicit damping force)
                //            //get stiffness matrix
                //                ASSEMBLEMATINIT(stiffnessMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
                //                ASSEMBLELIST(stiffnessMatrix, world.getSystemList(), getStiffnessMatrix);
                //                ASSEMBLELIST(stiffnessMatrix, world.getForceList(), getStiffnessMatrix);
                //                ASSEMBLEEND(stiffnessMatrix);
                
                //Need to filter internal forces seperately for this applicat
                ASSEMBLEVECINIT(forceVector, world.getNumQDotDOFs());
                ASSEMBLELIST(forceVector, world.getSystemList(), getImpl().getInternalForce);
                ASSEMBLEEND(forceVector);
                
                ASSEMBLEVECINIT(fExt, world.getNumQDotDOFs());
                ASSEMBLELIST(fExt, world.getSystemList(), getImpl().getBodyForce);
                ASSEMBLEEND(fExt);
                
                (*forceVector) = m_P*(*forceVector);
                
                if (m_num_modes != 0 ) {
                    //
                    //                    static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->calculateEigenFitData(q,massMatrix,stiffnessMatrix,m_coarseUs,Y,Z);
                    //
                    //    Correct Forces
                    (*forceVector) = (*forceVector) + Y*(m_coarseUs.first.transpose()*(*forceVector));
                    
                    // add damping
                    //                (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
                    if(eigenfit_damping)
                    {
                        if (integrator.compare("IM") == 0) {
                            (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot) + b * (Y*(Z * (m_P *( 1.0 / 2.0 *(eigen_v_old + qDot)))));
                        }
                        else {
                            (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P *(qDot) + b * (Y*(Z * (m_P *qDot)));
                        }
                    }
                    else{
                        // add damping
                        if (integrator.compare("IM") == 0) {
                            (*forceVector) = (*forceVector) -  ( b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
                        }
                        else
                        {
                            (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P *(qDot);
                        }
                    }
                }
                else
                {
                    if (integrator.compare("IM") == 0) {
                        (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
                    }
                    else
                    {
                        (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P *( qDot);
                    }
                }
                // add external force
                (*forceVector) = (*forceVector) + m_P*(*fExt);
                
                m_pardiso_mass.solve(*forceVector);
                res  = 1.0/2.0 * (m_P*(eigen_v_temp - eigen_v_old) - dt*m_pardiso_mass.getX()).squaredNorm();
                std::cout << "res: " << res<<endl;
                
            }
            else {
                
                //            cout<<"using simple mass"<<endl;
                //Eigendecomposition
                // if number of modes not equals to 0, use EigenFit
                if (m_num_modes != 0 ) {
                    
                    if (it_outer == 1) { //since the correction is small (not stiff), use explicit integration
                        static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->calculateEigenFitData(q,m_massMatrix,stiffnessMatrix,m_coarseUs,Y,Z);
                    }
                    //                cout<<"eigenfit data calculated "<<endl;
                    //    Correct Forces
                    (*forceVector) = (*forceVector) + Y*(m_coarseUs.first.transpose()*(*forceVector));
                    //                cout<<"force corrected"<<endl;
                    // add damping
                    //                (*forceVector) = (*forceVector) -  ( b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
                    //                cout<<"damping force added"<<endl;
                    if(eigenfit_damping)
                    {
                        if (integrator.compare("IM") == 0) {
                            (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot) + b * (Y*(Z * (m_P *( 1.0 / 2.0 *(eigen_v_old + qDot)))));
                        }
                        else {
                            (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P *(qDot) + b * (Y*(Z * (m_P *qDot)));
                        }
                    }
                    else{
                        // add damping
                        if (integrator.compare("IM") == 0) {
                            (*forceVector) = (*forceVector) -  ( b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
                        }
                        else
                        {
                            (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P *(qDot);
                        }
                    }
                    
                }
                else
                {
                    // add damping
                    if (integrator.compare("IM") == 0) {
                        
                        (*forceVector) = (*forceVector) -  (a * (m_M) + b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
                    }
                    else {
                        (*forceVector) = (*forceVector) -  (a * (m_M) + b*(*stiffnessMatrix)) * m_P * (qDot);
                        
                    }
                }
                // add external force
                (*forceVector) = (*forceVector) + m_P*(*fExt);
                //setup RHS
                eigen_rhs = (m_M)*m_P*(qDot-eigen_v_old) - dt*(*forceVector);
                
                m_pardiso_mass.symbolicFactorization(m_M);
                m_pardiso_mass.numericalFactorization();
                mass_factorized = true;
                
                m_pardiso_mass.solve(*forceVector);
                
                res_old = 1.0/2.0 * (m_P*(eigen_v_temp - eigen_v_old) - dt*m_pardiso_mass.getX()).squaredNorm();
                cout<<"res old: " << res_old<< ". ";
                Eigen::VectorXd x0;
                // last term is damping
                Eigen::SparseMatrix<DataType, Eigen::RowMajor> systemMatrix;
                
                if (integrator.compare("IM") == 0) {
                    
                    systemMatrix = -(m_M) + 1.0/4.0* dt*dt*(*m_stiffnessMatrix) - 1.0/2.0 * dt * (a *(m_M) + b * (*m_stiffnessMatrix));
                }
                else {
                    systemMatrix = -(m_M) + dt*dt*(*m_stiffnessMatrix) -  dt * (a *(m_M) + b * (*m_stiffnessMatrix));
                    
                }
#ifdef GAUSS_PARDISO
                
                m_pardiso.symbolicFactorization(systemMatrix, m_num_modes);
                m_pardiso.numericalFactorization();
                
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
                if(!matrix_fix_flag && m_num_modes != 0 )
                {
                    //                cout<<"Ignoring change in stiffness matrix from EigenFit"<<endl;
                }
                else if(m_num_modes != 0 )
                {
                    //                cout<<"Warning: stiffness fix not implemented"<<endl;
                }
                
                Dv = m_P.transpose()*x0;
                
                eigen_v_temp = qDot;
                eigen_v_temp = eigen_v_temp + Dv*step_size;
                //update state
                if (integrator.compare("IM") == 0) {
                    q = eigen_q_old + 1.0/4.0 * dt*(eigen_v_temp + eigen_v_old);
                }
                else
                {
                    q = eigen_q_old +  dt*(eigen_v_temp);
                    
                }
                
                //Need to filter internal forces seperately for this applicat
                ASSEMBLEVECINIT(forceVector, world.getNumQDotDOFs());
                ASSEMBLELIST(forceVector, world.getSystemList(), getImpl().getInternalForce);
                ASSEMBLEEND(forceVector);
                
                ASSEMBLEVECINIT(fExt, world.getNumQDotDOFs());
                ASSEMBLELIST(fExt, world.getSystemList(), getImpl().getBodyForce);
                ASSEMBLEEND(fExt);
                
                
                (*forceVector) = m_P*(*forceVector);
                
                if (m_num_modes != 0 ) {
                    
                    //                static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->calculateEigenFitData(q,massMatrix,stiffnessMatrix,m_coarseUs,Y,Z);
                    
                    //    Correct Forces
                    (*forceVector) = (*forceVector) + Y*(m_coarseUs.first.transpose()*(*forceVector));
                    
                    // add damping
                    //                (*forceVector) = (*forceVector) -  (a * (m_M) + b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
                    if(eigenfit_damping)
                    {
                        if (integrator.compare("IM") == 0) {
                            (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot) + b * (Y*(Z * (m_P *( 1.0 / 2.0 *(eigen_v_old + qDot)))));
                        }
                        else {
                            (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P *(qDot) + b * (Y*(Z * (m_P *qDot)));
                        }
                    }
                    else{
                        // add damping
                        if (integrator.compare("IM") == 0) {
                            (*forceVector) = (*forceVector) -  ( b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
                        }
                        else
                        {
                            (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P *(qDot);
                        }
                    }
                }
                else
                {
                    if (integrator.compare("IM") == 0) {
                        (*forceVector) = (*forceVector) -  (a * (m_M) + b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
                    }
                    else
                    {
                        (*forceVector) = (*forceVector) -  (a * (m_M) + b*(*stiffnessMatrix)) * m_P *(qDot);
                        
                    }
                }
                // add external force
                (*forceVector) = (*forceVector) + m_P*(*fExt);
                
                m_pardiso_mass.solve(*forceVector);
                res  = 1.0/2.0 * (m_P*(eigen_v_temp - eigen_v_old) - dt*m_pardiso_mass.getX()).squaredNorm();
                std::cout << "res: " << res <<endl;
                
            }
            
            qDot = eigen_v_temp;
            
            if (integrator.compare("IM") == 0) {
                q = eigen_q_old + 1.0/2.0 * dt * (qDot + eigen_v_old);
                
                update_step_size = (1.0/2.0 * dt * (qDot + eigen_v_old) - prev_update_step).norm();
                prev_update_step = 1.0/2.0 * dt * (qDot + eigen_v_old);
                cout<<" update size: " << update_step_size<<endl;
            }
            else
            {
                q = eigen_q_old + dt * (qDot);
                
                update_step_size = ( dt * (qDot) - prev_update_step).norm();
                prev_update_step =  dt * (qDot);
                cout<<" update size: " << update_step_size<<endl;
                
            }
            //    } while(res > 1e-6  && update_step_size > 1e-6);
            if (integrator.compare("SI") == 0) {
                break; // if it's IS, only need one step
            }
#ifdef NH
        } while(res > 1e-4 && update_step_size > 1e-4);
#endif
#ifdef LINEAR
    } while(false ); // only need one step for linear
#endif
#ifdef COROT
} while(res > 1e-4 && update_step_size > 1e-4 );
#endif
#ifdef ARAP
} while(res > 1e-4 && update_step_size > 1e-4);
#endif
it_outer = 0;

if (integrator.compare("IM") == 0) {
    q = eigen_q_old + 1.0/2.0 * dt * (qDot + eigen_v_old);
}
else
{
    q = eigen_q_old + dt * (qDot);
}
}


}

template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
using TimeStepperEigenFitSMWIM = TimeStepper<DataType, TimeStepperImplEigenFitSMWIMImpl<DataType, MatrixAssembler, VectorAssembler> >;







#endif /* TimeStepperEigenFitSMWIM_h */
