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
        bool inv_mass_calculated;
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
    
    std::cout<<"b: "<<b<<std::endl;
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
    (*stiffnessMatrix) = m_P*(*stiffnessMatrix)*m_P.transpose();
    
    (*forceVector) = m_P*(*forceVector);
    
    
    // add damping
    (*forceVector) = (*forceVector) -  (a * (*massMatrix) + b*(*stiffnessMatrix)) * m_P * ( qDot);
    
    // add external force
    (*forceVector) = (*forceVector) + m_P*(*fExt);
    
    //solve mass (Need interface for solvers but for now just use Eigen LLt)
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > solver_mass;
    
    if(m_refactor || !m_factored) {
        solver_mass.compute(*massMatrix);
    }
    
    if(solver_mass.info()!=Eigen::Success) {
        // decomposition failed
        assert(1 == 0);
        std::cout<<"Decomposition Failed \n";
        exit(1);
    }
    
    if(solver_mass.info()!=Eigen::Success) {
        // solving failed
        assert(1 == 0);
        std::cout<<"Solve Failed \n";
        exit(1);
    }
    
//    Eigen::saveMarket(*stiffnessMatrix,"stiffness.dat");
//    Eigen::saveMarket(*massMatrix,"mass.dat");
//    Eigen::saveMarket(m_P,"m_P.dat");
//    Eigen::saveMarketVector(q,"q.dat");
//    Eigen::saveMarketVector(qDot,"v.dat");
//    typedef Eigen::Triplet<DataType> T;
//    std::vector<T> tripletList;
    int N = m_P.rows();
    if(!inv_mass_calculated)
    {
        Eigen::SparseMatrix<DataType> I(N,N);
        I.setIdentity();
        
        //    Eigen::saveMarket(sp1*(*massMatrix),"inv_mass_t_mass.dat");
        inv_mass = solver_mass.solve(I);
        inv_mass_calculated = true;
        Eigen::VectorXx<DataType> ones(N);
        ones.setOnes();
        inv_mass_norm = (inv_mass.cwiseAbs()*ones).maxCoeff();
    }
    
//    Eigen::saveMarket(inv_mass,"inv_mass.dat");
//    Eigen::saveMarket(sp1*(*massMatrix),"inv_mass_t_mass.dat");
    
//    tripletList.reserve(N + 2*(*stiffnessMatrix).nonZeros() + inv_mass.nonZeros() + (*massMatrix).nonZeros() + 2*N+1);
//    for (int k=0; k<inv_mass.outerSize(); ++k)
//    {
//        for (Eigen::SparseMatrix<double>::InnerIterator it(inv_mass,k); it; ++it)
//        {
//            tripletList.push_back(T(it.row(), it.col() + N, it.value()));
//        }
//    }
//
//    for(int i = 0; i < N; i++)
//    {
//        tripletList.push_back(T(i, i + N, 1.0));
//    }
//    Eigen::Sparse J(2*m_P.rows(),2*m_P.rows());
    
//    J.bottomLeftCorner(m_P.rows(),m_P.rows()) = solver_mass.solve(*stiffnessMatrix);

//    Eigen::saveMarket(sp1,"sp1.dat");
//    for (int k=0; k<(*stiffnessMatrix).outerSize(); ++k)
//    {
//        for (Eigen::SparseMatrix<double>::InnerIterator it(*stiffnessMatrix,k); it; ++it)
//        {
//            tripletList.push_back(T(it.row() + N, it.col(), it.value()));
//        }
//    }
//
//    for (int k=0; k<(*stiffnessMatrix).outerSize(); ++k)
//    {
//        for (Eigen::SparseMatrix<double>::InnerIterator it(*stiffnessMatrix,k); it; ++it)
//        {
//            tripletList.push_back(T(it.row() + N, it.col() + N, b*it.value()));
//        }
//    }
//    for (int k=0; k<(*massMatrix).outerSize(); ++k)
//    {
//        for (Eigen::SparseMatrix<double>::InnerIterator it(*massMatrix,k); it; ++it)
//        {
//            tripletList.push_back(T(it.row() + N, it.col() + N, a*it.value()));
//        }
//    }
    
//    J.bottomRightCorner(m_P.rows(),m_P.rows()) = solver_mass.solve(b*(*stiffnessMatrix) + a*(*massMatrix));
//
    Eigen::VectorXx<DataType> du(2*N);
    du.head(N) = m_P * qDot;
    du.tail(N) = (*forceVector);
//
    Eigen::VectorXx<DataType> state_free(2*N);
    state_free.head(N) = m_P * q;
    state_free.tail(N) = (*massMatrix) * m_P * qDot;
    Eigen::VectorXx<DataType> g(2*N);
//    Eigen::saveMarketVector(state_free,"state_free.dat");
//
    double eta = 1;
//
//    Eigen::SparseMatrix<DataType> J_tilde(2*m_P.rows()+1,2*m_P.rows()+1);
//    J_tilde.setFro mTriplets(tripletList.begin(), tripletList.end());
//    g = du - J_tilde.topLeftCorner(2*N,2*N) * state_free;
    g.head(N) = du.head(N) - m_P * qDot;
    g.tail(N) = du.tail(N) + (*stiffnessMatrix) * m_P * q
    + (-a*(*massMatrix) - b*(*stiffnessMatrix)) * m_P * qDot;
//    Eigen::saveMarketVector(g,"g.dat");
//    Eigen::saveMarketVector(du,"du.dat");
//    Eigen::saveMarketVector(*forceVector,"forceVector.dat");
    
//
//    for (int i = 0; i < 2*N; i++) {
//        J_tilde.insert(i,2*N) = eta*g(i);
//    }
//    J_tilde.topLeftCorner(2*m_P.rows(),2*m_P.rows()) = J;
//    J_tilde.topRightCorner(2*m_P.rows(),1) = eta*g;
//
    Eigen::VectorXx<DataType> u_tilde(2*N+1);
    u_tilde.head(2*N) = state_free;
    u_tilde(2*N) = 1.0/eta;
//    Eigen::saveMarket(J_tilde,"J_tilde.dat");
//    Eigen::saveMarketVector(u_tilde,"u_tilde.dat");
//
    Eigen::VectorXx<DataType> X(2*N+1);
    X.setZero();
//    Eigen::MatrixXx<DataType> dJ;
//    dJ = Eigen::MatrixXx<DataType>(J_tilde);
//        Eigen::saveMarket(J_tilde,"J_tilde.dat");
//        Eigen::saveMarketVector(u_tilde,"u_tilde.dat");
//    cout<<"dt: "<<dt<<endl;
//    expv(dt,J_tilde,u_tilde,X);
    
    
    // matrix exponential
//    void expv(double t, MatrixType &A, Eigen::VectorXx<DataType> &v, Eigen::VectorXx<DataType> &out, double tol = 1e-7, int m = 30)
//    {
        //    Eigen::saveMarket(A,"test_a.dat");
        //    Eigen::saveMarketVector(v,"test_v.dat");
    double tol = 1e-7;
    int m = 30;
        int n = u_tilde.rows();
        
    Eigen::VectorXx<DataType> ones(N);
    ones.setOnes();
    double anorm = inv_mass_norm;
    cout<<"anorm: "<<anorm<<endl;
    double temp = ((*stiffnessMatrix).cwiseAbs()*ones  +(b*(*stiffnessMatrix)).cwiseAbs()*ones).maxCoeff();
    cout<<"temp: "<< temp<<endl;
    anorm = std::max(anorm, temp);
    cout<<"anorm: "<<anorm<<endl;
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
        t_new = (1/anorm)*pow((fact*tol)/(4*beta*anorm),xm);
        double s = pow(10,(floor(log10(t_new))-1));
        t_new = ceil(t_new/s)*s;
        int sgn = copysign(1,dt);
        nstep = 0;
        
        double avnorm = 1;
        Eigen::MatrixXx<DataType> F;
        F.setIdentity(m,m);
        double err_loc = 1;
        
        
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
            
            V.col(0) = (1/beta)*w;
            //        cout<<"w: "<<w<<endl;
            for(int j = 0; j < m; j++)
            {
                //            cout<<"j: "<<j<<endl;
                Eigen::VectorXx<DataType> p(2*N+1);
//                p = A*V.col(j);
//                saveMarket(V,"Vc.dat");
//                saveMarketVector(V.col(j),"Vj.dat");
//                saveMarketVector(V.col(j).segment(N,N),"VjN.dat");
                
                p.head(N) = solver_mass.solve(V.col(j).segment(N,N)) + (V.col(j)(2*N)) * eta*(g.head(N));
                p.segment(N,N) = (*stiffnessMatrix) * V.col(j).head(N);
                p.segment(N,N) += V.col(j)(2*N) * eta*g.segment(N,N);
                p.segment(N,N) += (-a)* V.col(j).segment(N,N);
                p.segment(N,N) += (- b*(*stiffnessMatrix)) * solver_mass.solve(V.col(j).segment(N,N));
                
                p(2*N) = 0;
                //            cout<<"p: "<<p<<endl;
//                Eigen::saveMarket(V,"Vc.dat");
//                Eigen::saveMarketVector(p,"p.dat");
                for(int  i = 0; i <= j; i++ )
                {
                    H(i,j) = V.col(i).transpose()*p;
                    p = p-H(i,j)*V.col(i);
                }
                s = p.norm();
                //            cout<<"p: "<<p<<endl;
                //            cout<<"p norm: "<<p.norm()<<endl;
                //            cout<<"s: "<<s<<endl;
                if(s < btol)
                {
                    k1 = 0;
                    mb = j;
                    t_step = t_out-t_now;
                    break;
                }
                H(j+1,j) = s;
                V.col(j+1) = (1/s)*p;
            }
            if(k1 != 0)
            {
                H(m+1,m) = 1;
//                Eigen::VectorXx<DataType> temp_vec;
//                avnorm = (A*V.col(m)).norm();
                avnorm = ((solver_mass.solve(V.col(m).segment(N,N)) + V.col(m)(2*N) * eta*g.head(N)).squaredNorm());
                avnorm += ((*stiffnessMatrix) * V.col(m).head(N) + V.col(m)(2*N) * eta*g.segment(N,N) + (-a*(*massMatrix) - b*(*stiffnessMatrix)) * solver_mass.solve( V.col(m).segment(N,N))).squaredNorm();
                avnorm = sqrt(avnorm);
//                cout<<"avnorm: "<<avnorm<<endl;
            }
            int ireject = 0;
            while (ireject <= mxrej)
            {
                int mx = mb + k1;
                //
                //            cout<<"t_step: "<<t_step<<endl;
                //            cout<<"mx: "<<mx<<endl;
                //            cout<<"H: "<<H<<endl;
                //            Eigen::MatrixXx<DataType> sp(mx,mx);
                //            sp = H.topLeftCorner(mx,mx);
                //            Eigen::saveMarket(H.topLeftCorner(mx,mx),"Hc.dat");
                F = (sgn*t_step*H.topLeftCorner(mx,mx)).exp();
                //            Eigen::saveMarket(F,"Fc.dat");
                
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
        double err = s_error;
        hump = hump / normv;
//    }
//    Eigen::saveMarketVector(X,"X.dat");
    
//
    q = m_P.transpose() * X.head(N);
    qDot =  m_P.transpose() * solver_mass.solve( X.segment(N,N));
}

template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
using TimeStepperERE = TimeStepper<DataType, TimeStepperImplEREImpl<DataType, MatrixAssembler, VectorAssembler> >;



#endif /* TimeStepperERE_h */
