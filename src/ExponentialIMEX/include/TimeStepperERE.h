    //
//  TimeStepperERE.h
//  Gauss
//
//  Created by Yu Ju Edwin Chen on 2018-11-04.
//

#ifndef TimeStepperERE_h
#define TimeStepperERE_h


#include <World.h>
#include <Assembler.h>
#include <TimeStepper.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <UtilitiesEigen.h>
#include <UtilitiesMATLAB.h>
#include <Eigen/SparseCholesky>
#include <SolverPardiso.h>
#include <ExponentialIMEX.h>
#include <limits>
#include <igl/speye.h>


//TODO Solver Interface
namespace Gauss {
    
    //Given Initial state, step forward in time using linearly implicit Euler Integrator
    template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
    class TimeStepperImplEREImpl
    {
    public:
        
        template<typename Matrix>
        TimeStepperImplEREImpl(Matrix &P, double a, double b) {
            
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
            
        }
        
        TimeStepperImplEREImpl(const TimeStepperImplEREImpl &toCopy) {
            
        }
        
        ~TimeStepperImplEREImpl() { }
        
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
        Eigen::SparseMatrix<DataType> MinvK;
        Eigen::VectorXx<DataType> mass_lumped;
        Eigen::VectorXx<DataType> mass_lumped_inv;
        
        bool inv_mass_calculated, mass_calculated;
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
void TimeStepperImplEREImpl<DataType, MatrixAssembler, VectorAssembler>::step(World &world, double dt, double t) {
    
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
    MinvK = inv_mass*(*stiffnessMatrix);
    (*forceVector) = m_P*(*forceVector);
    
    
    // add damping
    (*forceVector) = (*forceVector) -  (a * (*massMatrix) + b*(*stiffnessMatrix)) * m_P * ( qDot);
    
    // add external force
    (*forceVector) = (*forceVector) + m_P*(*fExt);
//
//    //solve mass (Need interface for solvers but for now just use Eigen LLt)
//    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > solver_mass;
//
//    if(m_refactor || !m_factored) {
//        solver_mass.compute(*massMatrix);
//    }
//
//    if(solver_mass.info()!=Eigen::Success) {
//        // decomposition failed
//        assert(1 == 0);
//        std::cout<<"Decomposition Failed \n";
//        exit(1);
//    }
//
//    if(solver_mass.info()!=Eigen::Success) {
//        // solving failed
//        assert(1 == 0);
//        std::cout<<"Solve Failed \n";
//        exit(1);
//    }
//
//    Eigen::saveMarket(*stiffnessMatrix,"stiffness.dat");
//    Eigen::saveMarket(*massMatrix,"mass.dat");
//    Eigen::saveMarket(m_P,"m_P.dat");
//    Eigen::saveMarketVector(q,"q.dat");
//    Eigen::saveMarketVector(qDot,"v.dat");
    int N = m_P.rows();

//    typedef Eigen::Triplet<DataType> T;
//    std::vector<T> tripletList;
//    tripletList.reserve(N + 2*(*stiffnessMatrix).nonZeros());
//    for(int i = 0; i < N; i++)
//    {
//        tripletList.push_back(T(i, i + N, 1.0));
//    }
//    for (int k=0; k<(mass_lumped.asDiagonal().inverse()*(*stiffnessMatrix)).outerSize(); ++k)
//    {
//        for (Eigen::SparseMatrix<double>::InnerIterator it(mass_lumped.asDiagonal().inverse()*(*stiffnessMatrix),k); it; ++it)
//        {
//            tripletList.push_back(T(it.row() + N, it.col(), it.value()));
//        }
//    }
//    for (int k=0; k<(mass_lumped.asDiagonal().inverse()*(*stiffnessMatrix)).outerSize(); ++k)
//    {
//        for (Eigen::SparseMatrix<double>::InnerIterator it(mass_lumped.asDiagonal().inverse()*(*stiffnessMatrix),k); it; ++it)
//        {
//            tripletList.push_back(T(it.row() + N, it.col()+N, -b*it.value()));
//        }
//    }
//    for(int i = 0; i < N; i++)
//    {
//        tripletList.push_back(T(i+N, i + N, -a));
//    }

//
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
    
//    Eigen::SparseMatrix<DataType> J(2*m_P.rows(),2*m_P.rows());
//    J.setFromTriplets(tripletList.begin(), tripletList.end());
    //
//    g.head(N) = du.head(N) - m_P * qDot;
//  efficient version
    g.head(N) = du.head(N);
    g.head(N).noalias() -= m_P * qDot;
    
//    g.tail(N) = du.tail(N) - mass_lumped.asDiagonal().inverse()*(*stiffnessMatrix) * m_P * q
//    - (-a) * m_P * qDot + b*mass_lumped.asDiagonal().inverse()*(*stiffnessMatrix) * m_P * qDot;
//  efficient version
    g.tail(N) = du.tail(N);
    g.tail(N).noalias() -=  MinvK * m_P * q;
    g.tail(N).noalias() -= (-a) * m_P * qDot;
    g.tail(N).noalias() += b*MinvK * m_P * qDot;
//    g = du - J * state_free;
//    cout<<"g - g2: "<<(g-g2).norm()<<endl;
    
//    for(int i = 0; i < 2*N; i++)
//    {
//        tripletList.push_back(T(i, 2*N, eta*g(i)));
//    }
//    Eigen::SparseMatrix<DataType> J_tilde(2*m_P.rows()+1,2*m_P.rows()+1);
//    J_tilde.setFromTriplets(tripletList.begin(), tripletList.end());
    
//    Eigen::saveMarket(J,"Jc.dat");
//    Eigen::saveMarket(J_tilde,"J_tildec.dat");
//
//    Eigen::saveMarketVector(g,"gc.dat");
//    Eigen::saveMarketVector(du,"du.dat");
//    Eigen::saveMarketVector(*forceVector,"forceVector.dat");
//    Eigen::saveMarketVector(mass_lumped,"mass_lumped.dat");
//
    Eigen::VectorXx<DataType> u_tilde(2*N+1);
    u_tilde.head(2*N) = state_free;
    u_tilde(2*N) = 1.0/eta;
//    Eigen::saveMarketVector(u_tilde,"u_tildec.dat");
    //
    Eigen::VectorXx<DataType> X(2*N+1);
    X.setZero();
    
    
    // matrix exponential
//    void expv(double t, MatrixType &A, Eigen::VectorXx<DataType> &v, Eigen::VectorXx<DataType> &out, double tol = 1e-7, int m = 30)
//    {
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
//                Eigen::VectorXx<DataType> temp_vec;
//                avnorm = (A*V.col(m)).norm();
//                saveMarket(V,"Vc.dat");
//                saveMarketVector(V.col(m),"Vm.dat");
//                saveMarketVector(V.col(m).segment(N,N),"VmN.dat");
//                saveMarket(H,"Hc.dat");
//                double avnorm2;
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

template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
using TimeStepperERE = TimeStepper<DataType, TimeStepperImplEREImpl<DataType, MatrixAssembler, VectorAssembler> >;



#endif /* TimeStepperERE_h */
