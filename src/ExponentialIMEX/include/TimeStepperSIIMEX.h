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
#include <MatOp/SparseGenMatProd.h>
#include <GenEigsRealShiftSolver.h>
#include <SparseGenRealShiftSolvePardiso.h>



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
        TimeStepperImplSIIMEXImpl(Matrix &P, double a, double b, int numModes) {
            
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
            J21.resize(m_P.rows()*2,m_P.rows()*2);
            
            MinvK.resize(m_P.rows(),m_P.rows());
            
//            typedef Eigen::Triplet<double> T;
            std::vector<T> tripletListJ21;
            tripletListJ21.reserve(m_P.rows());
            for(int i = 0; i < m_P.rows(); i++)
            {
                tripletListJ21.push_back(T(i,i+m_P.rows(),1.0));
            }
            J21.setFromTriplets(tripletListJ21.begin(),tripletListJ21.end());
            
            
            
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
            
            A.resize((J21).rows(), (J21).cols());

            
            
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
        
        Eigen::SparseMatrix<double,Eigen::RowMajor> J21;
        
        // for damping
        double a;
        // negative for opposite sign for stiffness
        double b;
        
        int m_numModes;
        
        
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
//    m_Us = generalizedEigenvalueProblemNotNormalized((*stiffnessMatrix), m_M, m_numModes, 0.00);
//    cout<<m_Us.second<<endl;
    MinvK = (1)*mass_lumped_inv.asDiagonal()*(*stiffnessMatrix);
    
//    Eigen::SparseMatrix<DataType,> MinvK = -A;
    
    //Spectra::SparseSymMassShiftSolve<DataType> Aop(K, M);
    //Aop.set_shift(shift);
    Spectra::SparseGenRealShiftSolvePardiso<DataType> op(MinvK);
    
    // Construct eigen solver object, requesting the smallest three eigenvalues
    Spectra::GenEigsRealShiftSolver<DataType, Spectra::LARGEST_MAGN, Spectra::SparseGenRealShiftSolvePardiso<DataType>> eigs(&op, m_numModes, 5*m_numModes,0.0);
    
    
    //Spectra::GenEigsShiftSolver<double, Spectra::LARGEST_MAGN,  Spectra::SparseSymMassShiftSolve<DataType> > eigs(&Aop, numVecs, 5*numVecs, shift);
    
    // Initialize and compute
    eigs.init();
    eigs.compute();
    
//    Spectra::SparseGenMatProd<double> op(MinvK);
//
//    // Construct eigen solver object, requesting the largest three eigenvalues
//    Spectra::GenEigsRealShiftSolver< double, Spectra::LARGEST_MAGN, Spectra::SparseGenMatProd<double> > eigs(&op, 3, 6,0.0);
//
//    // Initialize and compute
//    eigs.init();
//    int nconv = eigs.compute();
//    Eigen::VectorXcd evalues;
    if(eigs.info() == Spectra::SUCCESSFUL)
    {
//        evalues = eigs.eigenvalues();
//    eigenvalueProblemNotNormalized(MinvK, m_numModes, 0.00);
//    cout<<m_Us2.second<<endl;
//        cout<<eigs.eigenvalues()<<endl;
        m_Us = std::make_pair(eigs.eigenvectors().real(), eigs.eigenvalues().real());
    }
    else{
        cout<<"eigen solve failed"<<endl;
        exit(1);
    }
//    cout<<m_Us2.second;
//    Eigen::saveMarketDat(m_M,"mass.dat");
//    Eigen::saveMarketDat(m_Us.first,"not_normalized.dat");
    
    normalizing_const.noalias() = (m_Us.first.transpose() * mass_lumped.asDiagonal() * m_Us.first).diagonal();
    normalizing_const = normalizing_const.cwiseSqrt().cwiseInverse();
    
    m_Us.first.noalias() = m_Us.first * (normalizing_const.asDiagonal());
    
//    normalizing_const.noalias() = (m_Us2.first.transpose() * mass_lumped.asDiagonal() * m_Us2.first).diagonal();
//    normalizing_const = normalizing_const.cwiseSqrt().cwiseInverse();
//
//    m_Us2.first.noalias() = m_Us2.first * (normalizing_const.asDiagonal());
    //
//        Eigen::saveMarketDat(m_Us.first,"normalized.dat");
//        Eigen::saveMarketDat(m_Us2.first,"normalized2.dat");
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

    (J) = mass_lumped_inv2.asDiagonal() * (J) * rayleigh_b_scaling.asDiagonal();
    J += J21;

    
//
//    q = m_P.transpose() * X.head(N);
//    qDot =  m_P.transpose() *( X.segment(N,N));
    
    
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
