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


//TODO Solver Interface
namespace Gauss {
    
    //Given Initial state, step forward in time using linearly implicit Euler Integrator
    template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
    class TimeStepperImplEigenFitSMWIMImpl
    {
    public:
        
        template<typename Matrix>
        TimeStepperImplEigenFitSMWIMImpl(Matrix P, unsigned int num_modes, double a = 0.0, double b = -1e-2) {
            
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
        
        Eigen::SparseMatrix<double,Eigen::RowMajor> MinvK;
        Eigen::SparseMatrix<double,Eigen::RowMajor> m_M;
        
        double update_step_size;
        Eigen::VectorXd update_step;
        Eigen::VectorXd prev_update_step;
        
        Eigen::VectorXd Dv;
        
        bool step_success;
        
        
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
    
    prev_update_step = 1.0/2.0 * dt * (eigen_v_old + qDot);
    
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
        
        
        eigen_q_temp = eigen_q_old + 1.0/4.0 * dt * (eigen_v_old + qDot);
        
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
        
        if(!simple_mass_flag)
        {
            //Eigendecomposition
            // if number of modes not equals to 0, use EigenFit
            if (m_num_modes != 0 && static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->ratio_recalculation_switch != 6) {
                
                static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->calculateEigenFitData(q,m_massMatrix,stiffnessMatrix,m_coarseUs,Y,Z);
                
                //    Correct Forces
                (*forceVector) = (*forceVector) + Y*m_coarseUs.first.transpose()*(*forceVector);
                
                // add damping
//                (*forceVector) = (*forceVector) -  ( b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
                (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot) + b * (Y*(Z * (m_P *( 1.0 / 2.0 *(eigen_v_old + qDot)))));
                
            }
            else
            {
                // add damping
                (*forceVector) = (*forceVector) -  ( b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
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
            
            systemMatrix = -(*m_massMatrix) + 1.0/4.0* dt*dt*(*m_stiffnessMatrix) - 1.0/2.0 * dt * (a *(*m_massMatrix) + b * (*m_stiffnessMatrix));
            
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
            q = eigen_q_old + 1.0/4.0 * dt*(eigen_v_temp + eigen_v_old);
            
            //            warning: if using stiffness from the beginning of the step for rayleigh damping, no recalculation needed
//            //get stiffness matrix
//            ASSEMBLEMATINIT(stiffnessMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
//            ASSEMBLELIST(stiffnessMatrix, world.getSystemList(), getStiffnessMatrix);
//            ASSEMBLELIST(stiffnessMatrix, world.getForceList(), getStiffnessMatrix);
//            ASSEMBLEEND(stiffnessMatrix);
            
            //Need to filter internal forces seperately for this applicat
            ASSEMBLEVECINIT(forceVector, world.getNumQDotDOFs());
            ASSEMBLELIST(forceVector, world.getSystemList(), getImpl().getInternalForce);
            ASSEMBLEEND(forceVector);
            
            ASSEMBLEVECINIT(fExt, world.getNumQDotDOFs());
            ASSEMBLELIST(fExt, world.getSystemList(), getImpl().getBodyForce);
            ASSEMBLEEND(fExt);
            
            (*forceVector) = m_P*(*forceVector);
            
            if (m_num_modes != 0 && static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->ratio_recalculation_switch != 6) {
                
//                            static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->calculateEigenFitData(q,massMatrix,stiffnessMatrix,m_coarseUs,Y,Z);
                
                //    Correct Forces
                (*forceVector) = (*forceVector) + Y*m_coarseUs.first.transpose()*(*forceVector);
                
                // add damping
//                (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
                (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot) + b * (Y*(Z * (m_P *( 1.0 / 2.0 *(eigen_v_old + qDot)))));
            }
            else
            {
                (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
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
                
                static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->calculateEigenFitData(q,m_massMatrix,stiffnessMatrix,m_coarseUs,Y,Z);
//                cout<<"eigenfit data calculated "<<endl;
                //    Correct Forces
                (*forceVector) = (*forceVector) + Y*m_coarseUs.first.transpose()*(*forceVector);
//                cout<<"force corrected"<<endl;
                // add damping
//                (*forceVector) = (*forceVector) -  ( b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
//                cout<<"damping force added"<<endl;
                (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot) + b * (Y*(Z * (m_P *( 1.0 / 2.0 *(eigen_v_old + qDot)))));
            }
            else
            {
                // add damping
                (*forceVector) = (*forceVector) -  (a * (m_M) + b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
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
            
            systemMatrix = -(m_M) + 1.0/4.0* dt*dt*(*m_stiffnessMatrix) - 1.0/2.0 * dt * (a *(m_M) + b * (*m_stiffnessMatrix));
            
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
            q = eigen_q_old + 1.0/4.0 * dt*(eigen_v_temp + eigen_v_old);
            
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
                (*forceVector) = (*forceVector) + Y*m_coarseUs.first.transpose()*(*forceVector);
                
                // add damping
//                (*forceVector) = (*forceVector) -  (a * (m_M) + b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
                (*forceVector) = (*forceVector) -  (b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot) + b * (Y*(Z * (m_P *( 1.0 / 2.0 *(eigen_v_old + qDot)))));
            }
            else
            {
                (*forceVector) = (*forceVector) -  (a * (m_M) + b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
            }
            // add external force
            (*forceVector) = (*forceVector) + m_P*(*fExt);
            
            m_pardiso_mass.solve(*forceVector);
            res  = 1.0/2.0 * (m_P*(eigen_v_temp - eigen_v_old) - dt*m_pardiso_mass.getX()).squaredNorm();
            std::cout << "res: " << res <<endl;
            
        }
        
        
//        int inner_it = 0;
//        step_size = 1;
//        // armijo condition
//        while (res > res_old - c1 * step_size * res * 2)
//        {
//            cout<<"Armijo cond. failed, start backtracking, ";
//            res_old = res;
//            inner_it = inner_it + 1;
//            if (inner_it == 40)
//            {
//                cout<<"error: line search more than 40 iterations."<<endl;
//                step_success = false;
//                return;
//            }
//            // backtracking line search
//            step_size = 0.9 * step_size;
//
//            eigen_v_temp = qDot;
//            eigen_v_temp = eigen_v_temp + Dv*step_size;
//            //update state
//            q = eigen_q_old + 1.0/4.0 * dt*(eigen_v_temp + eigen_v_old);
//
//            //Need to filter internal forces seperately for this applicat
//            ASSEMBLEVECINIT(forceVector, world.getNumQDotDOFs());
//            ASSEMBLELIST(forceVector, world.getSystemList(), getImpl().getInternalForce);
//            ASSEMBLEEND(forceVector);
//
//            ASSEMBLEVECINIT(fExt, world.getNumQDotDOFs());
//            ASSEMBLELIST(fExt, world.getSystemList(), getImpl().getBodyForce);
//            ASSEMBLEEND(fExt);
//
//            (*forceVector) = m_P*(*forceVector);
//
//            // Warning: mass term removed in rayleigh damping, and use K from begining of the step
//            if (m_num_modes != 0 && static_cast<EigenFit*>(std::get<0>(world.getSystemList().getStorage())[0])->ratio_recalculation_switch != 6) {
//
//                //    Correct Forces
//                (*forceVector) = (*forceVector) + Y*m_coarseUs.first.transpose()*(*forceVector);
//
//                // add damping
//                (*forceVector) = (*forceVector) -  ( b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
//            }
//            else
//            {
//                (*forceVector) = (*forceVector) -  ( b*(*stiffnessMatrix)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
//            }
//            // add external force
//            (*forceVector) = (*forceVector) + m_P*(*fExt);
//
//            m_pardiso_mass.solve(*forceVector);
//            res  = 1.0/2.0 * (m_P*(eigen_v_temp - eigen_v_old) - dt*m_pardiso_mass.getX()).squaredNorm();
//            std::cout << "res: " << res << ", ";
//
//        }
//        cout<< "step size: " << step_size <<endl;
        
        qDot = eigen_v_temp;
        q = eigen_q_old + 1.0/2.0 * dt * (qDot + eigen_v_old);
        
        update_step_size = (1.0/2.0 * dt * (qDot + eigen_v_old) - prev_update_step).norm();
        prev_update_step = 1.0/2.0 * dt * (qDot + eigen_v_old);
        cout<<" update size: " << update_step_size<<endl;
//    } while(res > 1e-6  && update_step_size > 1e-6);
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
    q = eigen_q_old + 1.0/2.0 * dt * (qDot + eigen_v_old);
}

template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
using TimeStepperEigenFitSMWIM = TimeStepper<DataType, TimeStepperImplEigenFitSMWIMImpl<DataType, MatrixAssembler, VectorAssembler> >;







#endif /* TimeStepperEigenFitSMWIM_h */
