
//  ExponentialIMEX.h
//  Gauss
//
//  Created by Yu Ju Edwin Chen on 2018-11-01.
//

#ifndef ExponentialIMEX_h
#define ExponentialIMEX_h

#include <math.h>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/SparseExtra>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// transcribed from expv in expokit
//[w, err, hump] = expv( t, A, v, tol, m )
//EXPV computes an approximation of w = exp(t*A)*v for a
//general matrix A using Krylov subspace  projection techniques.
//It does not compute the matrix exponential in isolation but instead,
//it computes directly the action of the exponential operator on the
//operand vector. This way of doing so allows for addressing large
//sparse problems. The matrix under consideration interacts only
//via matrix-vector products (matrix-free method).
//
//w = expv( t, A, v )
//computes w = exp(t*A)*v using a default tol = 1.0e-7 and m = 30.
//
//[w, err] = expv( t, A, v )
//renders an estimate of the error on the approximation.
//
//[w, err] = expv( t, A, v, tol )
//overrides default tolerance.
//
//[w, err, hump] = expv( t, A, v, tol, m )
//overrides default tolerance and dimension of the Krylov subspace,
//and renders an approximation of the `hump'.
//
//The hump is defined as:
//        hump = max||exp(sA)||, s in [0,t]  (or s in [t,0] if t < 0).
//It is used as a measure of the conditioning of the matrix exponential
//problem. The matrix exponential is well-conditioned if hump = 1,
//whereas it is poorly-conditioned if hump >> 1. However the solution
//can still be relatively fairly accurate even when the hump is large
//(the hump is an upper bound), especially when the hump and
//||w(t)||/||v|| are of the same order of magnitude (further details in
//                                                      reference below).

using std::cout;
using std::endl;

//template<typename MatrixType>
//void phi(MatrixType &A, MatrixType &out)
//{
//    
//    Eigen::EigenSolver<MatrixType> es(A);
//    MatrixType D;
//    D.resize(A.rows(),A.cols());
//    D.setZero();
//    D = es.eigenvalues().real();
//    for (int i = 0; i < D.rows(); i++) {
//        D(i,i) = (exp(D(i,i)) - 1)/D(i,i);
//    }
//    out = es.eigenvectors().real()*D* es.eigenvectors().real().inverse();
//}

template<typename DataType, typename MatrixType>
void expv(double t, MatrixType &A, Eigen::VectorXx<DataType> &v, Eigen::VectorXx<DataType> &out, double tol = 1e-7, int m = 30)
{
//    Eigen::saveMarket(A,"test_a.dat");
//    Eigen::saveMarketVector(v,"test_v.dat");
    int n = v.rows();
    
    Eigen::VectorXx<DataType> ones(n);
    ones.setOnes();
    double anorm = (A.cwiseAbs()*ones).maxCoeff(); // infinity norm
    
    // some initialization
    int mxrej = 10;
    double btol  = 1.0e-7;
    double gamma = 0.9;
    double delta = 1.2;
    int mb    = m;
    double t_out   = abs(t);
    int nstep = 0;
    double t_new   = 0;
    double t_now = 0;
    double s_error = 0;
    double eps = 2.2204e-16;
    double rndoff = anorm*eps;
    
    int k1 = 2;
    double xm = 1.0/m;
    double normv = v.norm();
    double beta = normv;
    double fact = (pow((m+1)/exp(1),(m+1)))*sqrt(2*M_PI*(m+1));
    t_new = (1/anorm)*pow((fact*tol)/(4*beta*anorm),xm);
    double s = pow(10,(floor(log10(t_new))-1));
    t_new = ceil(t_new/s)*s;
    int sgn = copysign(1,t);
    nstep = 0;
    
    double avnorm = 1;
    Eigen::MatrixXx<DataType> F;
    F.setIdentity(m,m);
    double err_loc = 1;
    
    
    Eigen::VectorXx<DataType> w(v);
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
            Eigen::VectorXx<DataType> p;
            p = A*V.col(j);
//            cout<<"p: "<<p<<endl;
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
            avnorm = (A*V.col(m)).norm();
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
    out = w;
    double err = s_error;
    hump = hump / normv;
}
//
//// matrix free version
//template<typename DataType>
//void expv_matrixfree(double t, Eigen::VectorXd (&A)(Eigen::VectorXd), Eigen::VectorXx<DataType> &v, Eigen::VectorXx<DataType> &out, double anorm, double tol = 1e-7, int m = 30)
//{
//    //    Eigen::saveMarket(A,"test_a.dat");
//    //    Eigen::saveMarketVector(v,"test_v.dat");
//    int n = v.rows();
//
//    Eigen::VectorXx<DataType> ones(n);
//    ones.setOnes();
////    double anorm = (A.cwiseAbs()*ones).maxCoeff(); // infinity norm
//
//    // some initialization
//    int mxrej = 10;
//    double btol  = 1.0e-7;
//    double gamma = 0.9;
//    double delta = 1.2;
//    int mb    = m;
//    double t_out   = abs(t);
//    int nstep = 0;
//    double t_new   = 0;
//    double t_now = 0;
//    double s_error = 0;
//    double eps = 2.2204e-16;
//    double rndoff = anorm*eps;
//
//    int k1 = 2;
//    double xm = 1.0/m;
//    double normv = v.norm();
//    double beta = normv;
//    double fact = (pow((m+1)/exp(1),(m+1)))*sqrt(2*M_PI*(m+1));
//    t_new = (1/anorm)*pow((fact*tol)/(4*beta*anorm),xm);
//    double s = pow(10,(floor(log10(t_new))-1));
//    t_new = ceil(t_new/s)*s;
//    int sgn = copysign(1,t);
//    nstep = 0;
//
//    double avnorm = 1;
//    Eigen::MatrixXx<DataType> F;
//    F.setIdentity(m,m);
//    double err_loc = 1;
//
//
//    Eigen::VectorXx<DataType> w(v);
//    double hump = normv;
//    int stages = 0;
//    while (t_now < t_out)
//    {
//        stages = stages + 1;
//        nstep = nstep + 1;
//        double t_step = std::min( t_out-t_now,t_new );
//        Eigen::MatrixXx<DataType> V;
//        V.setZero(n,m+1);
//        Eigen::MatrixXx<DataType> H;
//        H.setZero(m+2,m+2);
//
//        V.col(0) = (1/beta)*w;
//        //        cout<<"w: "<<w<<endl;
//        for(int j = 0; j < m; j++)
//        {
//            //            cout<<"j: "<<j<<endl;
//            Eigen::VectorXx<DataType> p;
//            p = A(V.col(j));
//            //            cout<<"p: "<<p<<endl;
//            for(int  i = 0; i <= j; i++ )
//            {
//                H(i,j) = V.col(i).transpose()*p;
//                p = p-H(i,j)*V.col(i);
//            }
//            s = p.norm();
//            //            cout<<"p: "<<p<<endl;
//            //            cout<<"p norm: "<<p.norm()<<endl;
//            //            cout<<"s: "<<s<<endl;
//            if(s < btol)
//            {
//                k1 = 0;
//                mb = j;
//                t_step = t_out-t_now;
//                break;
//            }
//            H(j+1,j) = s;
//            V.col(j+1) = (1/s)*p;
//        }
//        if(k1 != 0)
//        {
//            H(m+1,m) = 1;
//            avnorm = (A(V.col(m))).norm();
//        }
//        int ireject = 0;
//        while (ireject <= mxrej)
//        {
//            int mx = mb + k1;
//            //
//            //            cout<<"t_step: "<<t_step<<endl;
//            //            cout<<"mx: "<<mx<<endl;
//            //            cout<<"H: "<<H<<endl;
//            //            Eigen::MatrixXx<DataType> sp(mx,mx);
//            //            sp = H.topLeftCorner(mx,mx);
//            //            Eigen::saveMarket(H.topLeftCorner(mx,mx),"Hc.dat");
//            F = (sgn*t_step*H.topLeftCorner(mx,mx)).exp();
//            //            Eigen::saveMarket(F,"Fc.dat");
//
//            if (k1 == 0)
//            {
//                err_loc = btol;
//                break;
//            }
//            else
//            {
//                double phi1 = abs( beta*F(m,0) );
//                double phi2 = abs( beta*F(m+1,0) * avnorm );
//                if(phi1 > 10*phi2){
//
//                    err_loc = phi2;
//                    xm = 1.0/m;
//                }
//                else if( phi1 > phi2)
//                {
//                    err_loc = (phi1*phi2)/(phi1-phi2);
//                    xm = 1.0/m;
//                }
//                else
//                {
//                    err_loc = phi1;
//                    xm = 1.0/(m-1);
//                }
//            }
//            if (err_loc <= delta * t_step*tol)
//            {
//                break;
//            }
//            else
//            {
//                t_step = gamma * t_step * pow(t_step*tol/err_loc,xm);
//                s = pow(10,(floor(log10(t_step))-1));
//                t_step = ceil(t_step/s) * s;
//                if (ireject == mxrej)
//                {
//                    printf ("The requested tolerance is too high.");
//                    exit (EXIT_FAILURE);
//                }
//                ireject = ireject + 1;
//                cout<<"ireject: "<< ireject<<endl;
//            }
//        }
//        int mx = mb + std::max( 0,k1-1 );
//        w = V.leftCols(mx)*(beta*F.topLeftCorner(mx,1));
//        beta = ( w ).norm();
//        hump = std::max(hump,beta);
//
//        t_now = t_now + t_step;
//        t_new = gamma * t_step * pow((t_step*tol/err_loc),xm);
//        s = pow(10,(floor(log10(t_new))-1));
//        t_new = ceil(t_new/s) * s;
//
//        err_loc = std::max(err_loc,rndoff);
//        s_error = s_error + err_loc;
//
//    }
//    out = w;
//    double err = s_error;
//    hump = hump / normv;
//}
//
//template<typename DataType>
//void expv_sparse(double t, Eigen::SparseMatrix<DataType> &A, Eigen::VectorXx<DataType> &v, Eigen::VectorXx<DataType> &out, double tol = 1e-7, int m = 30)
//{
//    //    Eigen::saveMarket(A,"test_a.dat");
//    //    Eigen::saveMarketVector(v,"test_v.dat");
//    int n = v.rows();
//    
//    Eigen::VectorXx<DataType> ones(n);
//    ones.setOnes();
//    double anorm = (A.cwiseAbs()*ones).maxCoeff(); // infinity norm
//    
//    // some initialization
//    int mxrej = 10;
//    double btol  = 1.0e-7;
//    double gamma = 0.9;
//    double delta = 1.2;
//    int mb    = m;
//    double t_out   = abs(t);
//    int nstep = 0;
//    double t_new   = 0;
//    double t_now = 0;
//    double s_error = 0;
//    double eps = 2.2204e-16;
//    double rndoff = anorm*eps;
//    
//    int k1 = 2;
//    double xm = 1.0/m;
//    double normv = v.norm();
//    double beta = normv;
//    double fact = (pow((m+1)/exp(1),(m+1)))*sqrt(2*M_PI*(m+1));
//    t_new = (1/anorm)*pow((fact*tol)/(4*beta*anorm),xm);
//    double s = pow(10,(floor(log10(t_new))-1));
//    t_new = ceil(t_new/s)*s;
//    int sgn = copysign(1,t);
//    nstep = 0;
//    
//    double avnorm = 1;
//    Eigen::MatrixXx<DataType> F;
//    F.setIdentity(m,m);
//    double err_loc = 1;
//    
//    
//    Eigen::VectorXx<DataType> w(v);
//    double hump = normv;
//    int stages = 0;
//    while (t_now < t_out)
//    {
//        stages = stages + 1;
//        nstep = nstep + 1;
//        double t_step = std::min( t_out-t_now,t_new );
//        Eigen::MatrixXx<DataType> V;
//        V.setZero(n,m+1);
//        Eigen::MatrixXx<DataType> H;
//        H.setZero(m+2,m+2);
//        
//        V.col(0) = (1/beta)*w;
//        //        cout<<"w: "<<w<<endl;
//        for(int j = 0; j < m; j++)
//        {
////            cout<<"j: "<<j<<endl;
//            Eigen::VectorXx<DataType> p;
//            p = A*V.col(j);
//            //            cout<<"p: "<<p<<endl;
//            for(int  i = 0; i <= j; i++ )
//            {
//                H(i,j) = V.col(i).transpose()*p;
//                p = p-H(i,j)*V.col(i);
//            }
//            s = p.norm();
//            //            cout<<"p: "<<p<<endl;
//            //            cout<<"p norm: "<<p.norm()<<endl;
//            //            cout<<"s: "<<s<<endl;
//            if(s < btol)
//            {
//                k1 = 0;
//                mb = j;
//                t_step = t_out-t_now;
//                break;
//            }
//            H(j+1,j) = s;
//            V.col(j+1) = (1/s)*p;
//        }
//        if(k1 != 0)
//        {
//            H(m+1,m) = 1;
//            avnorm = (A*V.col(m)).norm();
//        }
//        int ireject = 0;
//        while (ireject <= mxrej)
//        {
//            int mx = mb + k1;
//            
//            Eigen::MatrixXx<double> sp(mx,mx);
//            sp = sgn*t_step*H.topLeftCorner(mx,mx);
//            F = (sp).exp();
//            if (k1 == 0)
//            {
//                err_loc = btol;
//                break;
//            }
//            else
//            {
//                double phi1 = abs( beta*F(m,0) );
//                double phi2 = abs( beta*F(m+1,0) * avnorm );
//                if(phi1 > 10*phi2){
//                    
//                    err_loc = phi2;
//                    xm = 1/m;
//                }
//                else if( phi1 > phi2)
//                {
//                    err_loc = (phi1*phi2)/(phi1-phi2);
//                    xm = 1/m;
//                }
//                else
//                {
//                    err_loc = phi1;
//                    xm = 1/(m-1);
//                }
//            }
//            if (err_loc <= delta * t_step*tol)
//            {
//                break;
//            }
//            else
//            {
//                t_step = gamma * t_step * pow(t_step*tol/err_loc,xm);
//                s = pow(10,(floor(log10(t_step))-1));
//                t_step = ceil(t_step/s) * s;
//                if (ireject == mxrej)
//                {
//                    printf ("The requested tolerance is too high.");
//                    exit (EXIT_FAILURE);
//                }
//                ireject = ireject + 1;
//            }
//        }
//        int mx = mb + std::max( 0,k1-1 );
//        w = V.leftCols(mx)*(beta*F.topLeftCorner(mx,1));
//        beta = ( w ).norm();
//        hump = std::max(hump,beta);
//        
//        t_now = t_now + t_step;
//        t_new = gamma * t_step * pow((t_step*tol/err_loc),xm);
//        s = pow(10,(floor(log10(t_new))-1));
//        t_new = ceil(t_new/s) * s;
//        
//        err_loc = std::max(err_loc,rndoff);
//        s_error = s_error + err_loc;
//        
//    }
//    out = w;
//    double err = s_error;
//    hump = hump / normv;
//}
#endif /* ExponentialIMEX_h */
