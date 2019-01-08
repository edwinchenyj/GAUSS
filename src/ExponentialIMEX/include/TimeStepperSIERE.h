//
//  TimeStepperSIERE.h
//  Gauss
//
//  Created by Yu Ju Edwin Chen on 2018-12-21.
//

#ifndef TimeStepperSIERE_h
#define TimeStepperSIERE_h


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
#include <Eigen/Sparse>
#include <GenEigsSolver.h>
#include <MatOp/SparseGenMatProd.h>
#include <GenEigsRealShiftSolver.h>
#include <SparseGenRealShiftSolvePardiso.h>

using namespace Eigen;

//namespace Eigen {
//    template class Eigen::Map<Eigen::SparseMatrix<double,Eigen::RowMajor,int>>;
//}

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
        if(norm(D(j,j)) > 1e-3)
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
    class TimeStepperImplSIEREImpl
    {
    public:
        
        template<typename Matrix>
        TimeStepperImplSIEREImpl(Matrix &P, double a, double b, int numModes, std::string integrator = "SIERE") {
            
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
            
            m_numModes = numModes;
            
            inv_mass_calculated = false;
            mass_calculated = false;
            mass_lumped_calculated = false;
            
            Eigen::VectorXd ones(m_P.rows());
            ones.setOnes();
            rayleigh_b_scaling.resize(m_P.rows()*2);
            rayleigh_b_scaling.setZero();
            rayleigh_b_scaling.head(m_P.rows()) = ones;
            rayleigh_b_scaling.tail(m_P.rows()) = -b*ones;
            
            // some initialization for speed
            //            Eigen::SparseMatrix<double> J12;
            J12.resize(m_P.rows()*2,m_P.rows()*2);
            J21.resize(m_P.rows()*2,m_P.rows()*2);
            J22.resize(m_P.rows()*2,m_P.rows()*2);
            
            MinvK.resize(m_P.rows(),m_P.rows());
            
            //            typedef Eigen::Triplet<double> T;
            std::vector<T> tripletListJ12;
            tripletListJ12.reserve(m_P.rows());
            for(int i = 0; i < m_P.rows(); i++)
            {
                tripletListJ12.push_back(T(i,i+m_P.rows(),1.0));
            }
            J12.setFromTriplets(tripletListJ12.begin(),tripletListJ12.end());
            
            
            
            U1.resize(m_P.rows()*2,numModes);
            V1.resize(m_P.rows()*2,numModes);
            U2.resize(m_P.rows()*2,numModes);
            V2.resize(m_P.rows()*2,numModes);
            
            U1.setZero();
            V1.setZero();
            U2.setZero();
            V2.setZero();
            
            
            dt_J_G_reduced.resize(numModes*2,numModes*2);
            
            vG.resize(m_P.rows());
            vH.resize(m_P.rows());
            
            fG.resize(m_P.rows());
            fH.resize(m_P.rows());
            
            A.resize((J12).rows(), (J12).cols());
            
            this->integrator = integrator;
            
            step_number = 0;
            
            step_success = true;
            
            int m = P.cols() - P.rows();
            int n = P.cols();
            Eigen::VectorXd one = Eigen::VectorXd::Ones(P.rows());
            Eigen::MatrixXd P_col_sum =   one.transpose() * P;
            std::vector<Eigen::Triplet<DataType> > triplets;
            Eigen::SparseMatrix<DataType> S;
            S.resize(m,n);
            //
            
            unsigned int row_index =0;
            for(unsigned int col_index = 0; col_index < n; col_index++) {
                
                if (P_col_sum(0,col_index) == 0) {
                    triplets.push_back(Eigen::Triplet<DataType>(row_index,col_index,1));
                    row_index+=1;
                }
                
            }
            
            S.setFromTriplets(triplets.begin(), triplets.end());
            
            m_S = S;
            
        }
        
        TimeStepperImplSIEREImpl(const TimeStepperImplSIEREImpl &toCopy) {
            
        }
        
        ~TimeStepperImplSIEREImpl() {
//            delete [] outer_ind_ptr; delete [] inner_ind; delete [] values;
            delete rhs;
            delete v_old;
            delete v_temp;
            delete q_old;
            delete q_temp;
        }
        
        //Methods
        //init() //initial conditions will be set at the begining
        template<typename World>
        void step(World &world, double dt, double t);
        
        template<typename World>
        void calculate_rest_stiffness(World &world);
        
        
        inline typename VectorAssembler::MatrixType & getLagrangeMultipliers() { return m_lagrangeMultipliers; }
        
        
        Eigen::SparseMatrix<double,Eigen::RowMajor> J12;
        Eigen::SparseMatrix<double,Eigen::RowMajor> J21;
        Eigen::SparseMatrix<double,Eigen::RowMajor> J22;

        
        
        double* rhs  = NULL;
        double* v_old  = NULL;
        double* v_temp = NULL;
        double* q_old = NULL;
        double* q_temp = NULL;
        
        
        // for damping
        double a;
        // negative for opposite sign for stiffness
        double b;
        
        int m_numModes;
        
        std::string integrator;
        
        Eigen::MatrixXx<double> U1;
        Eigen::MatrixXx<double> V1;
        Eigen::MatrixXx<double> U2;
        Eigen::MatrixXx<double> V2;
        
        std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > m_Us;
        std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > m_Us2;
        
        Eigen::VectorXd normalizing_const;
        
        Eigen::MatrixXx<double> dt_J_G_reduced;
        
        Eigen::VectorXx<double> vG;
        Eigen::VectorXx<double> vH;
        
        Eigen::VectorXx<double> fG;
        
        Eigen::VectorXx<double> fH;
        
        Eigen::SparseMatrix<double,Eigen::RowMajor> A;
        
        Eigen::SparseMatrix<double,Eigen::RowMajor> MinvK;
        Eigen::SparseMatrix<double,Eigen::RowMajor> Identity;
        
        std::vector<int> J21_J22_outer_ind_ptr;
        std::vector<int> J22i_outer_ind_ptr;
        std::vector<int> J21_inner_ind;
//        std::vector<int> J22_outer_ind_ptr;
        std::vector<int> J22_inner_ind;
        std::vector<int> J22i_inner_ind;
        std::vector<double> J22i_identity_val;
        std::vector<double> stiffness0_val;
        std::vector<int> stiffness0_outer_ind_ptr;
        std::vector<int> stiffness0_inner_ind;


        
        double update_step_size;
        Eigen::VectorXd update_step;
        Eigen::VectorXd prev_update_step;
        
        bool step_success;
        
        bool islinear;
        bool stiffness_calculated;
        
        int step_number;
        int it_print;
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
        
        
        Eigen::SparseMatrix<DataType> m_P;
        Eigen::SparseMatrix<double> m_S;

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
void TimeStepperImplSIEREImpl<DataType, MatrixAssembler, VectorAssembler>::calculate_rest_stiffness(World &world) {
    
    MatrixAssembler &stiffnessMatrix = m_stiffnessMatrix;
    
    //Grab the state
    Eigen::Map<Eigen::VectorXd> q = mapStateEigen<0>(world);
    Eigen::Map<Eigen::VectorXd> qDot = mapStateEigen<1>(world);
    
    Eigen::VectorXd copy_q = q;
    q.setZero();
    
    //get stiffness matrix
    ASSEMBLEMATINIT(stiffnessMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
    ASSEMBLELIST(stiffnessMatrix, world.getSystemList(), getStiffnessMatrix);
    ASSEMBLELIST(stiffnessMatrix, world.getForceList(), getStiffnessMatrix);
    ASSEMBLEEND(stiffnessMatrix);
    
    
    (*stiffnessMatrix) = m_P*(*stiffnessMatrix)*m_P.transpose();
    for (int i_nnz = 0; i_nnz < (*stiffnessMatrix).nonZeros(); i_nnz++)
    {
//        stiffness0_val.push_back(*((*stiffnessMatrix).innerIndexPtr()+i_nnz));
        stiffness0_val.push_back(*((*stiffnessMatrix).valuePtr()+i_nnz));
        stiffness0_inner_ind.push_back(*((*stiffnessMatrix).innerIndexPtr()+i_nnz));
    }
    
    for (int i_row = 0; i_row < (*stiffnessMatrix).rows() + 1; i_row++) {
        stiffness0_outer_ind_ptr.push_back(*((*stiffnessMatrix).outerIndexPtr()+i_row));
    }
    q = copy_q;
    
    
}

template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
template<typename World>
void TimeStepperImplSIEREImpl<DataType, MatrixAssembler, VectorAssembler>::step(World &world, double dt, double t) {
    
    cout<<"integrator: "<<integrator<<endl;
    if (integrator.compare("SIERE")==0) {
    
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
    
    Eigen::Map<Eigen::SparseMatrix<double,Eigen::RowMajor>> K0_map((*stiffnessMatrix).rows(), (*stiffnessMatrix).cols(), (*stiffnessMatrix).nonZeros(), stiffness0_outer_ind_ptr.data(), stiffness0_inner_ind.data(), stiffness0_val.data());
//        Eigen::Map<Eigen::SparseMatrix<double,Eigen::RowMajor>> K0_map((*stiffnessMatrix).rows(), (*stiffnessMatrix).cols(), (*stiffnessMatrix).nonZeros(), (*stiffnessMatrix).outerIndexPtr(), (*stiffnessMatrix).innerIndexPtr(), stiffness0_val.data());
        
        
    (*forceVector) = m_P*(*forceVector);
    
    // add damping
    (*forceVector).noalias() -= (a*(*m_massMatrix)+b*(K0_map)) * (m_P * ( qDot));
    
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
    //    m_Us = generalizedEigenvalueProblemNotNormalized((*stiffnessMatrix), m_M, m_numModes, 0.00);
    //    cout<<m_Us.second<<endl;
        Eigen::SparseMatrix<double,Eigen::RowMajor> MinvK0;
        MinvK0 = (1)*mass_lumped_inv.asDiagonal()*(K0_map);
    MinvK = (1)*mass_lumped_inv.asDiagonal()*(*stiffnessMatrix);
    
    //    Eigen::SparseMatrix<DataType,> MinvK = -A;
    
    //Spectra::SparseSymMassShiftSolve<DataType> Aop(K, M);
    //Aop.set_shift(shift);
    Spectra::SparseGenRealShiftSolvePardiso<DataType> op(MinvK);
    
    // Construct eigen solver object, requesting the smallest three eigenvalues
    Spectra::GenEigsRealShiftSolver<DataType, Spectra::LARGEST_MAGN, Spectra::SparseGenRealShiftSolvePardiso<DataType>> eigs(&op, m_numModes, 5*m_numModes,0.0);
    
    // Initialize and compute
    eigs.init();
    eigs.compute();
    
    if(eigs.info() == Spectra::SUCCESSFUL)
    {
        int neg_evals = 0;
        for (int i_eval = 0; i_eval < m_numModes; i_eval++) {
            if (eigs.eigenvalues().real()(i_eval) > 0) {
                neg_evals++;
            }
        }
        cout<<"there are " <<neg_evals<< " negative eigenvalues."<<endl;
        m_Us = std::make_pair(eigs.eigenvectors().real(), eigs.eigenvalues().real());
    }
    else{
        cout<<"eigen solve failed"<<endl;
        exit(1);
    }
    
    normalizing_const.noalias() = (m_Us.first.transpose() * mass_lumped.asDiagonal() * m_Us.first).diagonal();
    normalizing_const = normalizing_const.cwiseSqrt().cwiseInverse();
    
    m_Us.first = m_Us.first * (normalizing_const.asDiagonal());
    
    
//    if (J21_J22_outer_ind_ptr.empty()) {
        J21_J22_outer_ind_ptr.erase(J21_J22_outer_ind_ptr.begin(),J21_J22_outer_ind_ptr.end());
        J22i_outer_ind_ptr.erase(J22i_outer_ind_ptr.begin(),J22i_outer_ind_ptr.end());
        for (int i_row = 0; i_row < MinvK.rows(); i_row++) {
            J21_J22_outer_ind_ptr.push_back(0);
            J22i_outer_ind_ptr.push_back(0);
        }
        
        J22i_inner_ind.erase(J22i_inner_ind.begin(),J22i_inner_ind.end());
        J22i_identity_val.erase(J22i_identity_val.begin(),J22i_identity_val.end());
        for (int i_row = 0; i_row < MinvK.rows() + 1; i_row++) {
            J21_J22_outer_ind_ptr.push_back(*(MinvK.outerIndexPtr()+i_row));
            J22i_outer_ind_ptr.push_back(i_row);
            J22i_inner_ind.push_back(i_row + MinvK.rows());
            J22i_identity_val.push_back(1.0);
        }
//    }
    
//    if (J21_inner_ind.empty() || J22_inner_ind.empty()) {
        J21_inner_ind.erase(J21_inner_ind.begin(),J21_inner_ind.end());
        J22_inner_ind.erase(J22_inner_ind.begin(),J22_inner_ind.end());
        
        for (int i_nnz = 0; i_nnz < MinvK.nonZeros(); i_nnz++)
        {
            J21_inner_ind.push_back(*(MinvK.innerIndexPtr()+i_nnz));
            J22_inner_ind.push_back(*(MinvK.innerIndexPtr()+i_nnz) + MinvK.cols());
        }
        
//    }
    
    Eigen::Map<Eigen::SparseMatrix<double,Eigen::RowMajor>> J21_map((MinvK.rows())*2, (MinvK.cols())*2, MinvK.nonZeros(), J21_J22_outer_ind_ptr.data(), MinvK.innerIndexPtr(), (MinvK).valuePtr());
    Eigen::Map<Eigen::SparseMatrix<double,Eigen::RowMajor>> J22_map((MinvK.rows())*2, (MinvK.cols())*2, MinvK.nonZeros(), J21_J22_outer_ind_ptr.data(), J22_inner_ind.data(), (MinvK0).valuePtr());
        Eigen::Map<Eigen::SparseMatrix<double,Eigen::RowMajor>> J22i_map((MinvK.rows())*2, (MinvK.cols())*2, MinvK.cols(), J22i_outer_ind_ptr.data(), J22i_inner_ind.data(),J22i_identity_val.data());
    
#ifndef NDEBUG
//    cout<<"debug mode, printing matrices."<<endl;
//            Eigen::saveMarketDat(J22i_map,"J22i.dat");
        
//
    

#endif
    
    
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
    
    vG.noalias() = m_Us.first * (m_Us.first.transpose() * mass_lumped.asDiagonal() * (m_P * qDot));
    
    vH = -vG;
    vH.noalias() += m_P*qDot;
    
    //    Eigen::saveMarketVectorDat(vG,"vG.dat");
    //    Eigen::saveMarketVectorDat(vH,"vH.dat");
    
    fG.noalias() = (mass_lumped.asDiagonal() * m_Us.first ) * (m_Us.first.transpose() * (*forceVector));
    fH = (*forceVector) - fG;
    
    //    Eigen::saveMarketVectorDat(fG,"fG.dat");
    //    Eigen::saveMarketVectorDat(fH,"fH.dat");
    
    
    A.setIdentity();
//    A -= dt * (J);
    A -= dt * (J12 + J21_map -b*J22_map - a*J22i_map);
    
//#ifndef NDEBUG
//    Eigen::SparseMatrix<double, Eigen::RowMajor> A2;
//    A2.resize(A.rows(),A.cols());
//    A2.setIdentity();
//    A2 -= dt * (J12 + J21_map -b*J22_map);
//
//    Eigen::saveMarketDat(A,"A.dat");
//    Eigen::saveMarketDat(A2,"A2.dat");
//#endif
    
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
    rhs1.tail(m_P.rows()).noalias() = (-dt) * mass_lumped_inv.asDiagonal() * fH;
    
    //    Eigen::saveMarketVectorDat(rhs1,"rhs1c.dat");
    
    
    Eigen::VectorXx<double> reduced_vec;
    reduced_vec.resize(dt_J_G_reduced.cols());
    reduced_vec.head(dt_J_G_reduced.cols()/2).noalias() = m_Us.first.transpose() * (mass_lumped.asDiagonal() * (m_P * qDot));
    reduced_vec.tail(dt_J_G_reduced.cols()/2).noalias() = (m_Us.first.transpose() * (*forceVector));
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
    
    rhs2.noalias() = (-dt) * block_diag_eigv * phi_reduced * reduced_vec;
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
    y0 = x0;
    y0.noalias() -= x1 * yLHS.ldlt().solve(V1.transpose()*x0);
    //    Eigen::saveMarketVectorDat(y0,"y0c.dat");
    
    //
    Eigen::VectorXd y1;
    y1.resize(x2.rows());
    Eigen::MatrixXd yRHS2 = V1.transpose()*x2;
    //    m_pardiso_y.solve(yRHS2);
    x2.noalias() -= x1 * (yLHS.ldlt().solve(yRHS2));
    //    Eigen::saveMarketDat(x2,"y1c.dat");
    //    y1 = x2 - x1 * (yLHS.ldlt().solve(yRHS2));
    //
    Eigen::MatrixXd sol2LHS = Is + V2.transpose()*x2;
    //    m_pardiso_sol2.symbolicFactorization(sol2LHS);
    //    m_pardiso_sol2.numericalFactorization();
    Eigen::VectorXd sol2;
    Eigen::MatrixXd sol2RHS = V2.transpose()*y0;
    //    m_pardiso_sol2.solve(sol2RHS);
    y0.noalias() -= x2 * (sol2LHS).ldlt().solve(sol2RHS);
    //    Eigen::saveMarketDat(y0,"sol2.dat");
    //    sol2 = y0 - y1 * (Is + V2.transpose()*y1).ldlt().solve(V2.transpose()*y0);
    
    auto state = mapStateEigen(world);
    
    state -= m_P2.transpose()*y0;
    
#else
#endif
        
    }
    else if(integrator.compare("ERE") == 0)
    {
        
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
            inv_mass_calculated = true;
            mass_calculated = true;
        }
        
        (*stiffnessMatrix) = m_P*(*stiffnessMatrix)*m_P.transpose();
        
        Eigen::Map<Eigen::SparseMatrix<double,Eigen::RowMajor>> K0_map((*stiffnessMatrix).rows(), (*stiffnessMatrix).cols(), (*stiffnessMatrix).nonZeros(), stiffness0_outer_ind_ptr.data(), stiffness0_inner_ind.data(), stiffness0_val.data());
        Eigen::SparseMatrix<double,Eigen::RowMajor> MinvK0;
        MinvK0 = inv_mass*(K0_map);
        MinvK = inv_mass*(*stiffnessMatrix);
        (*forceVector) = m_P*(*forceVector);
        
        // add damping
        (*forceVector) = (*forceVector) -  (a * (*massMatrix) + b*(K0_map)) * m_P * ( qDot);
        
        // add external force
        (*forceVector) = (*forceVector) + m_P*(*fExt);
        
        int N = m_P.rows();
        
        Eigen::VectorXx<DataType> du(2*N);
        du.head(N).noalias() = m_P * qDot;
        du.tail(N).noalias() = inv_mass*(*forceVector);
        
        Eigen::VectorXx<DataType> state_free(2*N);
        state_free.head(N).noalias() = m_P * q;
        state_free.tail(N).noalias() = m_P * qDot;
        Eigen::VectorXx<DataType> g(2*N);
        Eigen::VectorXx<DataType> g2(2*N);
        
        double eta = 1;
        
//        Eigen::SparseMatrix<double,Eigen::RowMajor> iden;
//
//        iden.resize(MinvK.rows(),MinvK.cols());
//        iden.setIdentity();
        //
        //  efficient version
        g.head(N) = du.head(N);
        g.head(N).noalias() -= m_P * qDot;
        //  efficient version
        g.tail(N) = du.tail(N);
        g.tail(N).noalias() -=  MinvK * m_P * q;
        g.tail(N).noalias() -= (-a) * m_P * qDot;
        g.tail(N).noalias() += (b*MinvK0) * m_P * qDot;
        
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
        double temp = (MinvK.cwiseAbs()*ones +  (b*(MinvK0)).cwiseAbs()*ones + a * ones).maxCoeff();// +(a*mass_lumped.asDiagonal()+ ;
        //    cout<<"temp: "<< temp<<endl;
        anorm = std::max(anorm, temp);
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
                //                p.head(N) = (V.col(j).segment(N,N)) + (V.col(j)(2*N)) * eta*(g.head(N));
                //                efficient version
                p.head(N) = (V.col(j).segment(N,N));
                p.head(N).noalias() += (V.col(j)(2*N)) * eta*(g.head(N));
                //                p.segment(N,N).noalias() = mass_lumped_inv.asDiagonal()*((*stiffnessMatrix) * V.col(j).head(N));
                p.segment(N,N).noalias() = MinvK * V.col(j).head(N);
                p.segment(N,N).noalias() += (V.col(j)(2*N)) * eta*g.tail(N);
                p.segment(N,N).noalias() += (-a)* (V.col(j).segment(N,N));
                p.segment(N,N).noalias() += (-b)*(MinvK0 * (V.col(j).segment(N,N)));
                p(2*N) = 0;
                for(int  i = 0; i <= j; i++ )
                {
                    H(i,j) = V.col(i).transpose()*p;
                    p.noalias() -= H(i,j)*V.col(i);
                    
                }
                s = p.norm();
                if(s < btol)
                {
                    k1 = 0;
                    mb = j;
                    t_step = t_out-t_now;
                    break;
                }
                H(j+1,j) = s;
                V.col(j+1) = (1.0/s)*p;
                
            }
            if(k1 != 0)
            {
                H(m+1,m) = 1.0;
                avnorm = (V.col(m).segment(N,N) + (V.col(m)(2*N)) * eta*g.head(N)).squaredNorm();
                avnorm += (MinvK * V.col(m).head(N) + (V.col(m)(2*N)) * eta*(g.tail(N)) + (-a * V.col(m).segment(N,N) - b*(MinvK0) *  V.col(m).segment(N,N))).squaredNorm();
                avnorm = sqrt(avnorm);
                
            }
            int ireject = 0;
            while (ireject <= mxrej)
            {
                int mx = mb + k1;
                //
                F.noalias() = (sgn*t_step*H.topLeftCorner(mx,mx)).exp();
                
                if (k1 == 0)
                {
                    err_loc = btol;
                    break;
                }
                else
                {
                    double phi1 = abs( beta*F(m,0) );
                    double phi2 = abs( beta*F(m+1,0) * avnorm );
                    
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
        hump = hump / normv;
        Eigen::VectorXd q_s,qDot_s;
        q_s = m_S *q;
        qDot_s = m_S*qDot;
        q = m_P.transpose() * X.head(N) + m_S.transpose() * q_s;
        qDot =  m_P.transpose() *( X.segment(N,N))  + m_S.transpose() * qDot_s;
        //                    q += dt * qDot;
    }
    else
    {
        
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
        
        if (integrator.compare("IM") == 0) {
            prev_update_step = 1.0/2.0 * dt * (eigen_v_old + qDot);
        }
        else{ //BE or SI
            prev_update_step = eigen_q_old + dt*qDot;
        }
        
        do {
            std::cout<<"it outer: " << it_outer<<std::endl;std::cout<<"Newton it outer: " << it_outer<< ", ";
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

            Eigen::Map<Eigen::SparseMatrix<double,Eigen::RowMajor>> K0_map((*stiffnessMatrix).rows(), (*stiffnessMatrix).cols(), (*stiffnessMatrix).nonZeros(), stiffness0_outer_ind_ptr.data(), stiffness0_inner_ind.data(), stiffness0_val.data());
            
            (*forceVector) = m_P*(*forceVector);
            
            // add damping
            if (integrator.compare("IM") == 0) {
                (*forceVector) = (*forceVector) -  (a *(*m_massMatrix) + b*(K0_map)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
            }
            else
            {
//                cout<<"K0 size: " << K0_map.rows() << " " <<K0_map.cols()<<endl;
                (*forceVector) = (*forceVector) -  (a *(*m_massMatrix) + b*K0_map) * m_P *(qDot);
            }
            
            // add external force
            (*forceVector) = (*forceVector) + m_P*(*fExt);
            
            //setup RHS
            eigen_rhs = (m_M)*m_P*(qDot-eigen_v_old) - dt*(*forceVector);
            
            res_old = 1.0/2.0 * dt * dt * (m_MInv*(*forceVector)).squaredNorm();
            
            // std::cout<<"res_old: "<<res_old << std::endl;
            
            Eigen::VectorXd x0;
            // last term is damping
            
            Eigen::SparseMatrix<DataType, Eigen::RowMajor> systemMatrix;
            if (integrator.compare("IM") == 0) {
                systemMatrix = -(*m_massMatrix) + 1.0/4.0* dt*dt*(*m_stiffnessMatrix) - 1.0/2.0 * dt * (a *(*m_massMatrix) + b * (K0_map));
            }
            else{
//                cout<<"K0 size: " << K0_map.rows() << " " <<K0_map.cols()<<endl;
                systemMatrix = -(*m_massMatrix) + dt*dt*(*m_stiffnessMatrix) - dt * (a *(*m_massMatrix) + b * (K0_map));
                
            }
            
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
            eigen_v_temp = qDot;
            eigen_v_temp = eigen_v_temp + Dv*step_size;
            //update state
            if (integrator.compare("IM") == 0) {
                q = eigen_q_old + 1.0/4.0 * dt*(eigen_v_temp + eigen_v_old);
            }
            else{
                q = eigen_q_old +  dt*(eigen_v_temp);
            }
            
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
            (*stiffnessMatrix) = m_P*(*stiffnessMatrix)*m_P.transpose();
            
            (*forceVector) = m_P*(*forceVector);
            
            if (integrator.compare("IM") == 0) {
                (*forceVector) = (*forceVector) -  ( a *(*m_massMatrix) + b*(K0_map)) * m_P * 1.0 / 2.0 *(eigen_v_old + qDot);
            }
            else
            {
                (*forceVector) = (*forceVector) -  (a *(*m_massMatrix) + b*(K0_map)) * m_P *(qDot);
            }
            
            // add external force
            (*forceVector) = (*forceVector) + m_P*(*fExt);
            
            //        m_pardiso_mass.solve(*forceVector);
            std::cout << "res: " << 1.0/2.0 * (m_P*(eigen_v_temp - eigen_v_old) - dt*m_MInv*(*forceVector)).squaredNorm()<< std::endl;
            res  = 1.0/2.0 * (m_P*(eigen_v_temp - eigen_v_old) - dt*m_MInv*(*forceVector)).squaredNorm();
            
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
#ifdef STVK
} while(res > 1e-4 && update_step_size > 1e-4);
#endif

it_print = it_outer;
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
using TimeStepperSIERE = TimeStepper<DataType, TimeStepperImplSIEREImpl<DataType, MatrixAssembler, VectorAssembler> >;

#endif /* TimeStepperSIERE_h */
