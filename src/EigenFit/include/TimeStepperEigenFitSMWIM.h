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
        TimeStepperImplEigenFitSMWIMImpl(Matrix P, unsigned int num_modes, double a = 0.0, double b = -1e-2, std::string integrator = "IM") {
            
            this->integrator = integrator;
            
            m_num_modes = (num_modes);
            
            m_P = P;
            m_factored = false;
            // refactor for every solve
            m_refactor = true;
            
            // init residual
            res = std::numeric_limits<double>::infinity();
            
            
            
            m_M.resize(P.rows(),P.rows());
            m_M.reserve(P.rows()); //simple mass
            mass_lumped.resize(P.rows());
            mass_lumped_inv.resize(P.rows());
            
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
        
        double update_step_size;
        Eigen::VectorXd update_step;
        Eigen::VectorXd prev_update_step;
        
        Eigen::VectorXd Dv;
        
        bool step_success;
        
        std::string integrator;
        
        bool islinear;
        bool stiffness_calculated;
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
void TimeStepperImplEigenFitSMWIMImpl<DataType, MatrixAssembler, VectorAssembler>::step(World &world, double dt, double t) {
    
    simple_mass_flag = static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->simple_mass_flag;
    
    
    if (integrator.compare("IM")==0 || integrator.compare("BE")==0 || integrator.compare("SI")==0)
    {
    //First two lines work around the fact that C++11 lambda can't directly capture a member variable.
    MatrixAssembler &massMatrix = m_massMatrix;
    MatrixAssembler &stiffnessMatrix = m_stiffnessMatrix;
    VectorAssembler &forceVector = m_forceVector;
    VectorAssembler &fExt = m_fExt;
    
    ASSEMBLEMATINIT(massMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
    ASSEMBLELIST(massMatrix, world.getSystemList(), getMassMatrix);
    ASSEMBLEEND(massMatrix);
    
    (*massMatrix) = m_P*(*massMatrix)*m_P.transpose();
    
    
    
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
        
        mass_calculated = true;
    }
    
    
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
            
            //constraint Projection
            (*stiffnessMatrix) = m_P*(*stiffnessMatrix)*m_P.transpose();
            
            (*forceVector) = m_P*(*forceVector);
            
            if(!simple_mass_flag)
            {
                //Eigendecomposition
                // if number of modes not equals to 0, use EigenFit
                if (m_num_modes != 0 && static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->ratio_recalculation_switch != 6) {
                    
                    if (it_outer == 1) { //since the correction is small (not stiff), use explicit integration
                        static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->calculateEigenFitData(q,m_massMatrix,stiffnessMatrix,m_coarseUs,Y,Z);
                    }
                    
                    //    Correct Forces
                    (*forceVector) = (*forceVector) + Y*m_coarseUs.first.transpose()*(*forceVector);
                    
                    // add damping
                    //                    (*forceVector) = (*forceVector) -  ( b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
                    // implicit
                    if (integrator.compare("IM") == 0) {
                        (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot) + b * (Y*(Z * (m_P *( 1.0 / 2.0 *(eigen_v_old + qDot)))));
                    }
                    else {
                        (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P *(qDot) + b * (Y*(Z * (m_P *qDot)));
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
                if(!matrix_fix_flag && m_num_modes != 0 && static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->ratio_recalculation_switch != 6)
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
                
                if (m_num_modes != 0 && static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->ratio_recalculation_switch != 6) {
                    //
                    //                    static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->calculateEigenFitData(q,massMatrix,stiffnessMatrix,m_coarseUs,Y,Z);
                    //
                    //    Correct Forces
                    (*forceVector) = (*forceVector) + Y*(m_coarseUs.first.transpose()*(*forceVector));
                    
                    // add damping
                    //                (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
                    if (integrator.compare("IM") == 0) {
                        (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot) + b * (Y*(Z * (m_P *( 1.0 / 2.0 *(eigen_v_old + qDot)))));
                    }
                    else{
                        (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P  * (qDot) + b * (Y*(Z * (m_P * qDot)));
                        
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
                if (m_num_modes != 0 && static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->ratio_recalculation_switch != 6) {
                    
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
                    if (integrator.compare("IM") == 0) {
                        (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot) + b * (Y*(Z * (m_P *( 1.0 / 2.0 *(eigen_v_old + qDot)))));
                    }
                    else
                    {
                        (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P *( qDot) + b * (Y*(Z * (m_P * qDot)));
                        
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
                if(!matrix_fix_flag && m_num_modes != 0 && static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->ratio_recalculation_switch != 6)
                {
                    //                cout<<"Ignoring change in stiffness matrix from EigenFit"<<endl;
                }
                else if(m_num_modes != 0 && static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->ratio_recalculation_switch != 6)
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
                
                if (m_num_modes != 0 && static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->ratio_recalculation_switch != 6) {
                    
                    //                static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->calculateEigenFitData(q,massMatrix,stiffnessMatrix,m_coarseUs,Y,Z);
                    
                    //    Correct Forces
                    (*forceVector) = (*forceVector) + Y*(m_coarseUs.first.transpose()*(*forceVector));
                    
                    // add damping
                    //                (*forceVector) = (*forceVector) -  (a * (m_M) + b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
                    if (integrator.compare("IM") == 0) {
                        (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot) + b * (Y*(Z * (m_P *( 1.0 / 2.0 *(eigen_v_old + qDot)))));
                    }
                    else{
                        (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P * ( qDot) + b * (Y*(Z * (m_P * qDot)));
                        
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
        } while(res > 1e-4);
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
    else if(integrator.compare("ERE") == 0)
    {
        
        //Grab the state
        Eigen::Map<Eigen::VectorXd> q = mapStateEigen<0>(world);
        Eigen::Map<Eigen::VectorXd> qDot = mapStateEigen<1>(world);
        
        MatrixAssembler &massMatrix = m_massMatrix;
        
        MatrixAssembler &stiffnessMatrix = m_stiffnessMatrix;
        VectorAssembler &forceVector = m_forceVector;
        VectorAssembler &fExt = m_fExt;
        
        //get mass matrix
        ASSEMBLEMATINIT(massMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
        ASSEMBLELIST(massMatrix, world.getSystemList(), getMassMatrix);
        ASSEMBLEEND(massMatrix);
        
        if(!islinear || !stiffness_calculated)
        {
            //get stiffness matrix
            ASSEMBLEMATINIT(stiffnessMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
            ASSEMBLELIST(stiffnessMatrix, world.getSystemList(), getStiffnessMatrix);
            ASSEMBLELIST(stiffnessMatrix, world.getForceList(), getStiffnessMatrix);
            ASSEMBLEEND(stiffnessMatrix);
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
        
        //constraint Projection
        (*massMatrix) = m_P*(*massMatrix)*m_P.transpose();

        if (!mass_calculated) {
            
            Eigen::VectorXx<DataType> ones(m_P.rows());
            ones.setOnes();
            mass_lumped = ((*massMatrix)*ones);
            mass_lumped_inv = mass_lumped.cwiseInverse();
            
            typedef Eigen::Triplet<double> T;
            std::vector<T> tripletList;
            tripletList.reserve((*massMatrix).rows());
            inv_mass.resize((*massMatrix).rows(),(*massMatrix).rows());
            MinvK.resize((*massMatrix).rows(),(*massMatrix).rows());
            inv_mass.reserve((*massMatrix).rows());
            
            for(int i = 0; i < (*massMatrix).rows(); i++)
            {
                tripletList.push_back(T(i,i,mass_lumped_inv(i)));
            }
            inv_mass.setFromTriplets(tripletList.begin(),tripletList.end());
            mass_calculated = true;
        }
        
        (*stiffnessMatrix) = m_P*(*stiffnessMatrix)*m_P.transpose();
        MinvK = inv_mass*(*stiffnessMatrix);
        (*forceVector) = m_P*(*forceVector);
        
        if (m_num_modes != 0 && static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->ratio_recalculation_switch != 6) {
                static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->calculateEigenFitData(q,m_massMatrix,stiffnessMatrix,m_coarseUs,Y,Z);
//          //    Correct Forces
            (*forceVector) = (*forceVector) + Y*(m_coarseUs.first.transpose()*(*forceVector));
            
                (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P *( qDot) + b * (Y*(Z * (m_P * qDot)));
            
            
        }
        else
        {
                (*forceVector) = (*forceVector) -  ( b*(*stiffnessMatrix)) * m_P * (qDot);
        }
        // add external force
        (*forceVector) = (*forceVector) + m_P*(*fExt);
        
        int N = m_P.rows();
        
        Eigen::VectorXx<DataType> du(2*N);
        du.head(N).noalias() = m_P * qDot;
        du.tail(N).noalias() = inv_mass*(*forceVector);
        //    Eigen::saveMarketVector(du,"duc.dat");
        //
        Eigen::VectorXx<DataType> state_free(2*N);
        state_free.head(N).noalias() = m_P * q;
        state_free.tail(N).noalias() = m_P * qDot;
        Eigen::VectorXx<DataType> g(2*N);
        Eigen::VectorXx<DataType> g2(2*N);
        //
        //    Eigen::saveMarketVector(state_free,"state_free.dat");
        //
        double eta = 1;
        //
        g.head(N) = du.head(N);
        g.head(N).noalias() -= m_P * qDot;
        //  efficient version
        g.tail(N) = du.tail(N);
        g.tail(N).noalias() -=  MinvK * m_P * q;
        g.tail(N).noalias() -= (-a) * m_P * qDot;
        g.tail(N).noalias() += b*MinvK * m_P * qDot;
       
        Eigen::VectorXx<DataType> u_tilde(2*N+1);
        u_tilde.head(2*N) = state_free;
        u_tilde(2*N) = 1.0/eta;
        //    Eigen::saveMarketVector(u_tilde,"u_tildec.dat");
        //
        Eigen::VectorXx<DataType> X(2*N+1);
        X.setZero();
        
        double tol = 1e-7;
        int m = 30;
        int n = u_tilde.rows();
        
        Eigen::VectorXx<DataType> ones(N);
        ones.setOnes();
        double anorm = ones.maxCoeff();
        //    cout<<"anorm: "<<anorm<<endl;
        double temp = (MinvK.cwiseAbs()*ones +  (b*(MinvK)).cwiseAbs()*ones + a * ones).maxCoeff();// +(a*mass_lumped.asDiagonal()+ ;
        //    cout<<"temp: "<< temp<<endl;
        anorm = std::max(anorm, temp);
        //    cout<<"anorm: "<<anorm<<endl;
        //        double anorm = (A.cwiseAbs()*ones).maxCoeff(); // infinity norm
        
        // some initialization
        int mxrej = 10;
        double btol  = 1.0e-7;
        double gamma = 0.9;
        double delta = 1.2;
        int mb    = m;
        double t_out   = abs(dt);
        int nstep = 0;
        double t_new   = 0;
        double t_now = 0;
        double s_error = 0;
        double eps = 2.2204e-16;
        double rndoff = anorm*eps;
        
        int k1 = 2;
        double xm = 1.0/m;
        double normv = u_tilde.norm();
        double beta = normv;
        double fact = (pow((m+1)/exp(1),(m+1)))*sqrt(2*M_PI*(m+1));
        t_new = (1.0/anorm)*pow((fact*tol)/(4*beta*anorm),xm);
        double s = pow(10,(floor(log10(t_new))-1));
        t_new = ceil(t_new/s)*s;
        int sgn = copysign(1,dt);
        nstep = 0;
        
        double avnorm = 1.0;
        Eigen::MatrixXx<DataType> F;
        F.setIdentity(m,m);
        double err_loc = 1.0;
        
        
        Eigen::VectorXx<DataType> w(u_tilde);
        double hump = normv;
        int stages = 0;
        while (t_now < t_out)
        {
            stages = stages + 1;
            nstep = nstep + 1;
            double t_step = std::min( t_out-t_now,t_new );
            Eigen::MatrixXx<DataType> V;
            V.setZero(n,m+1);
            Eigen::MatrixXx<DataType> H;
            H.setZero(m+2,m+2);
            
            V.col(0) = (1.0/beta)*w;
            //            saveMarketVector(V.col(0),"V0.dat");
            for(int j = 0; j < m; j++)
            {
                //            cout<<"j: "<<j<<endl;
                Eigen::VectorXx<DataType> p(2*N+1);
                Eigen::VectorXx<DataType> p2(2*N+1);
                //                p = A*V.col(j);
                //                saveMarket(V,"Vc.dat");
                //                saveMarketVector(V.col(j),"Vj.dat");
                //                saveMarketVector(V.col(j).segment(N,N),"VjN.dat");
                
                //                p.head(N) = (V.col(j).segment(N,N)) + (V.col(j)(2*N)) * eta*(g.head(N));
                //                efficient version
                p.head(N) = (V.col(j).segment(N,N));
                p.head(N).noalias() += (V.col(j)(2*N)) * eta*(g.head(N));
                //                p.segment(N,N).noalias() = mass_lumped_inv.asDiagonal()*((*stiffnessMatrix) * V.col(j).head(N));
                p.segment(N,N).noalias() = MinvK * V.col(j).head(N);
                p.segment(N,N).noalias() += (V.col(j)(2*N)) * eta*g.tail(N);
                p.segment(N,N).noalias() += (-a)* (V.col(j).segment(N,N));
                p.segment(N,N).noalias() += (-b)*(MinvK * (V.col(j).segment(N,N)));
                p(2*N) = 0;
                //                p = J_tilde*V.col(j);
                //            cout<<"p: "<<p<<endl;
                //                Eigen::saveMarket(V,"Vc.dat");
                //                saveMarketVector(p,"pc.dat");
                //                saveMarketVector(p2,"p2c.dat");
                //                cout<<"p - p2: "<<(p-p2).norm()<<endl;
                for(int  i = 0; i <= j; i++ )
                {
                    //                    cout<<"i: "<<i<<endl;
                    //                    cout<<"j: "<<j<<endl;
                    H(i,j) = V.col(i).transpose()*p;
                    //                    cout<<"H(i,j): "<< H(i,j)<<endl;
                    p.noalias() -= H(i,j)*V.col(i);
                    //                    saveMarketVector(p,"pc.dat");
                    
                }
                s = p.norm();
                //            cout<<"p: "<<p<<endl;
                //            cout<<"p norm: "<<p.norm()<<endl;
                //                            cout<<"s: "<<s<<endl;
                if(s < btol)
                {
                    k1 = 0;
                    mb = j;
                    t_step = t_out-t_now;
                    break;
                }
                //                cout<<"j: "<<j<<endl;
                H(j+1,j) = s;
                V.col(j+1) = (1.0/s)*p;
                //                saveMarketVector(V.col(j+1),"Vj1.dat");
            }
            if(k1 != 0)
            {
                H(m+1,m) = 1.0;
                avnorm = (V.col(m).segment(N,N) + (V.col(m)(2*N)) * eta*g.head(N)).squaredNorm();
                avnorm += (MinvK * V.col(m).head(N) + (V.col(m)(2*N)) * eta*(g.tail(N)) + (-a * V.col(m).segment(N,N) - b*(MinvK) *  V.col(m).segment(N,N))).squaredNorm();
                avnorm = sqrt(avnorm);
                //                avnorm = (J_tilde*V.col(m)).norm();
                //                cout<<"avnorm: "<<avnorm<<endl;
                //                cout<<"avnorm - avnorm2: "<<avnorm -avnorm2<<endl;
            }
            int ireject = 0;
            while (ireject <= mxrej)
            {
                int mx = mb + k1;
                //
                //                            cout<<"t_step: "<<t_step<<endl;
                //                            cout<<"mx: "<<mx<<endl;
                //                            cout<<"H: "<<H<<endl;
                //                            Eigen::MatrixXx<DataType> sp(mx,mx);
                //                            sp = H.topLeftCorner(mx,mx);
                //                            Eigen::saveMarket(H.topLeftCorner(mx,mx),"Hc.dat");
                //                Eigen::saveMarket(sgn*t_step*H.topLeftCorner(mx,mx),"expA.dat");
                F.noalias() = (sgn*t_step*H.topLeftCorner(mx,mx)).exp();
                //                Eigen::saveMarket(F,"Fc.dat");
                
                if (k1 == 0)
                {
                    err_loc = btol;
                    break;
                }
                else
                {
                    double phi1 = abs( beta*F(m,0) );
                    double phi2 = abs( beta*F(m+1,0) * avnorm );
                    //                    cout<<"phi1: "<<phi1<<endl;
                    //                    cout<<"phi2: "<<phi2<<endl;
                    
                    if(phi1 > 10*phi2){
                        
                        err_loc = phi2;
                        xm = 1.0/m;
                    }
                    else if( phi1 > phi2)
                    {
                        err_loc = (phi1*phi2)/(phi1-phi2);
                        xm = 1.0/m;
                    }
                    else
                    {
                        err_loc = phi1;
                        xm = 1.0/(m-1);
                    }
                }
                if (err_loc <= delta * t_step*tol)
                {
                    break;
                }
                else
                {
                    t_step = gamma * t_step * pow(t_step*tol/err_loc,xm);
                    s = pow(10,(floor(log10(t_step))-1));
                    t_step = ceil(t_step/s) * s;
                    if (ireject == mxrej)
                    {
                        printf ("The requested tolerance is too high.");
                        exit (EXIT_FAILURE);
                    }
                    ireject = ireject + 1;
                    cout<<"ireject: "<< ireject<<endl;
                    }
                    }
                    int mx = mb + std::max( 0,k1-1 );
                    w = V.leftCols(mx)*(beta*F.topLeftCorner(mx,1));
                    beta = ( w ).norm();
                    hump = std::max(hump,beta);
                    
                    t_now = t_now + t_step;
                    t_new = gamma * t_step * pow((t_step*tol/err_loc),xm);
                    s = pow(10,(floor(log10(t_new))-1));
                    t_new = ceil(t_new/s) * s;
                    
                    err_loc = std::max(err_loc,rndoff);
                    s_error = s_error + err_loc;
                    
                    }
                    X = w;
                    double err = s_error;
                    hump = hump / normv;
                    //    }
                    //    Eigen::saveMarketVector(X,"X.dat");
                    
                    //
                    q = m_P.transpose() * X.head(N);
                    qDot =  m_P.transpose() *( X.segment(N,N));
                    
    }
}

template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
using TimeStepperEigenFitSMWIM = TimeStepper<DataType, TimeStepperImplEigenFitSMWIMImpl<DataType, MatrixAssembler, VectorAssembler> >;







#endif /* TimeStepperEigenFitSMWIM_h */
