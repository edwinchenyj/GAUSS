#include <functional>

#include <Qt3DIncludes.h>
#include <GaussIncludes.h>
#include <FEMIncludes.h>

//Any extra things I need such as constraints
#include <ConstraintFixedPoint.h>
#include <TimeStepperERE.h>
#include <TimeStepperSIIMEX.h>
#include <ExponentialIMEX.h>
#include <TimeStepperEigenFitSMWIM.h>
#include <EigenFit.h>
#include <fstream>
#include <igl/boundary_facets.h>

#include <unsupported/Eigen/MatrixFunctions>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

using namespace Gauss;
using namespace FEM;
using namespace ParticleSystem; //For Force Spring

/* Tetrahedral finite elements */
typedef PhysicalSystemFEM<double, NeohookeanTet> FEMLinearTets;

typedef World<double, std::tuple<FEMLinearTets *>,
std::tuple<ForceSpringFEMParticle<double> *, ForceParticlesGravity<double> *>,
std::tuple<ConstraintFixedPoint<double> *> > MyWorld;

typedef TimeStepperERE<double, AssemblerParallel<double, AssemblerEigenSparseMatrix<double>>, AssemblerParallel<double, AssemblerEigenVector<double>> > MyTimeStepper;

typedef Scene<MyWorld, MyTimeStepper> MyScene;

//
//std::vector<ConstraintFixedPoint<double> *> movingConstraints;
//std::vector<ConstraintFixedPoint<double> *> fixedConstraints;
//Eigen::VectorXi movingVerts;
//Eigen::VectorXi fixedVerts;
//Eigen::MatrixXd V, Vtemp;
//Eigen::MatrixXi F;
//Eigen::MatrixXi surfF;
//Eigen::MatrixXi surfFf;

Eigen::MatrixXd V;
Eigen::MatrixXi F;

char **arg_list;
unsigned int istep;
//
//typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
//typedef Eigen::Triplet<double> T;
//void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int n);

void preStepCallback(MyWorld &world) {
}

int main(int argc, char **argv) {
    //
    //    // block testing matrix exponential
    //    int n_t = 10;  // size of the image
    //    int m_t = n_t*n_t;  // number of unknows (=number of pixels)
    //    // Assembly:
    //    std::vector<T> coefficients;            // list of non-zeros coefficients
    //    Eigen::VectorXd b_t(m_t);                   // the right hand side-vector resulting from the constraints
    //    buildProblem(coefficients, b_t, n_t);
    //    SpMat A(m_t,m_t);
    //    A.setFromTriplets(coefficients.begin(), coefficients.end());
    //    // Solving:
    //    Eigen::SimplicialCholesky<SpMat> chol(A);  // performs a Cholesky factorization of A
    //    Eigen::VectorXd x = chol.solve(b_t);         // use the factorization to solve for the given right hand side
    //    // Export the result to a file:
    //    //    saveAsBitmap(x, n, argv[1]);
    //    //    cout<<"A: \n"<<A<<endl;
    //    //    cout<<"x: \n"<<x<<endl;
    //    Eigen::MatrixXd dA;
    //    dA = Eigen::MatrixXd(A);
    //    //    cout<<"exp(A)x: \n"<<dA.exp()*x<<endl;
    //    Eigen::VectorXd res(m_t);
    //    res.setZero();
    //    expv(1,dA,x,res);
    //    //    cout<<res<<endl;
    //    cout<<"err: "<<(res-dA.exp()*x).norm()<<endl;
    //    res.setZero();
    //    expv(1,A,x,res);
    //    cout<<"err sparse: "<<(res-dA.exp()*x).norm()<<endl;
    //    //
    
    std::cout<<"Test Neohookean FEM\n";
    
    //Setup Physics
    MyWorld world;
    
    arg_list = argv;
    
    //    define the file separator base on system
    const char kPathSeparator =
#ifdef _WIN32
    '\\';
#else
    '/';
#endif
    
    using std::cout;
    using std::endl;
    
    //    default example meshes
    std::string cmeshname = "/meshesTetWild/brick/brick_surf/brick_surf_5";
    
    
    //    readTetgen(V, F, dataDir()+cmeshname+".node", dataDir()+cmeshname+".ele");
//    readTetgen(V, F, dataDir()+"/meshesTetgen/Beam/Beam.node", dataDir()+"/meshesTetgen/Beam/Beam.ele");
    readTetgen(V, F, dataDir()+"/meshesTetWild/arma/arma_4.node", dataDir()+"/meshesTetWild/arma/arma_4.ele");
    
    
    //    default parameters
    double youngs = 2e6;
    double poisson = 0.45;
    int constraint_dir = 0; // constraint direction. 0 for x, 1 for y, 2 for z
    double constraint_tol = 1e-2;
    double a = 0.0;
    double b = -0.0001;
    
    
    FEMLinearTets *test = new FEMLinearTets(V,F);
    
    
    
    // set material
    for(unsigned int iel=0; iel<test->getImpl().getF().rows(); ++iel) {
        
        test->getImpl().getElement(iel)->setParameters(youngs, poisson);
        
    }
    world.addSystem(test);
    
    
    world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
    
    auto q = mapStateEigen(world);
    
    
    
    std::vector<ConstraintFixedPoint<double> *> movingConstraints;
    //    default constraint
    fixDisplacementMin(world, test,constraint_dir,constraint_tol);
    
    // construct the projection matrix for stepper
    Eigen::VectorXi indices = minVertices(test, constraint_dir,constraint_tol);
    Eigen::SparseMatrix<double> P = fixedPointProjectionMatrix(indices, *test,world);
    
    for(unsigned int ii=0; ii<indices.rows(); ++ii) {
        movingConstraints.push_back(new ConstraintFixedPoint<double>(&test->getQ()[indices[ii]], Eigen::Vector3d(0,0,0)));
        world.addConstraint(movingConstraints[ii]);
    }
    
    //    default to zero deformation
    q.setZero();
    
    
    MyTimeStepper stepper(0.01,P, a, b);
    
    Eigen::MatrixXi surfF;
    igl::boundary_facets(F,surfF);
    
    unsigned int file_ind = 0;
    struct stat buf;
    unsigned int idxc;
    
    for(istep=0; istep<1000 ; ++istep)
    {
        stepper.step(world);
//        cout<<"Moving constrained vertices using mouse motion"<<endl;
//        Eigen::VectorXd Xvel;
//        if(!Eigen::loadMarketVector(Xvel, "data/mouseXvel.mtx"))
//        {
//            cout<<"fail loading mouse x motion"<<endl;
//        }
//        Eigen::VectorXd Yvel;
//        if(!Eigen::loadMarketVector(Yvel, "data/mouseYvel.mtx"))
//            cout<<"fail loading mouse y motion"<<endl;
//        Eigen::VectorXd Zvel;
//        if(!Eigen::loadMarketVector(Zvel, "data/mouseZvel.mtx"))
//            cout<<"fail loading mouse z motion"<<endl;
//
//        for(unsigned int jj=0; jj<movingConstraints.size(); ++jj) {
//            auto v_q = mapDOFEigen(movingConstraints[jj]->getDOF(0), world.getState());
//            if ((istep) < 250) {
//                if(Xvel(istep) <= 0){   v_q(0) += 0.5*std::max(Xvel(istep),-0.005);}
//                else{ v_q(0) += 0.5*std::min(Xvel(istep),0.005);}
//                if(Yvel(istep) <= 0){   v_q(1) += 0.5*std::max(Yvel(istep),-0.005);}
//                else{ v_q(1) += 0.5*std::min(Yvel(istep),0.005);}
//                if(Zvel(istep) <= 0){   v_q(2) += 0.5*std::max(Zvel(istep),-0.005);}
//                else{ v_q(2) += 0.5*std::min(Zvel(istep),0.005);}
//            }
//        }
        std::ofstream ofile;
        //output data stream into text
        ofile.open("PE.txt", std::ios::app); //app is append which means it will put the text at the end
        ofile << std::get<0>(world.getSystemList().getStorage())[0]->getImpl().getPotentialEnergy(world.getState()) << std::endl;
        ofile.close();
        
        //output data stream into text
        ofile.open("Hamiltonian.txt", std::ios::app); //app is append which means it will put the text at the end
        ofile << std::get<0>(world.getSystemList().getStorage())[0]->getImpl().getEnergy(world.getState()) << std::endl;
        ofile.close();
        
        ofile.open("KE.txt", std::ios::app); //app is append which means it will put the text at the end
        ofile << std::get<0>(world.getSystemList().getStorage())[0]->getImpl().getEnergy(world.getState())  - std::get<0>(world.getSystemList().getStorage())[0]->getImpl().getPotentialEnergy(world.getState())<< std::endl;
        ofile.close();
        
        
        // check if the file already exist
        std::string filename = "surfpos" + std::to_string(file_ind) + ".obj";
        while (stat(filename.c_str(), &buf) != -1)
        {
            file_ind++;
            filename = "surfpos" + std::to_string(file_ind) + ".obj";
        }
        
        
        
        // rest pos for the coarse mesh getGeometry().first is V
        q = mapStateEigen(world);
        idxc = 0;
        Eigen::MatrixXd V_disp = std::get<0>(world.getSystemList().getStorage())[0]->getGeometry().first;
        
        
        // get the mesh position
        for(unsigned int vertexId=0;  vertexId < std::get<0>(world.getSystemList().getStorage())[0]->getGeometry().first.rows(); ++vertexId) {
            
            V_disp(vertexId,0) += q(idxc);
            idxc++;
            V_disp(vertexId,1) += q(idxc);
            idxc++;
            V_disp(vertexId,2) += q(idxc);
            idxc++;
        }
        
        
        //             output mesh position with elements
        igl::writeOBJ("pos" + std::to_string(file_ind) + ".obj",V_disp,std::get<0>(world.getSystemList().getStorage())[0]->getGeometry().second);
        
        // output mesh position with only surface mesh
        igl::writeOBJ("surfpos" + std::to_string(file_ind) + ".obj",V_disp,surfF);
        //
    }
//    //Display
//    QGuiApplication app(argc, argv);
//
//    MyScene *scene = new MyScene(&world, &stepper, preStepCallback);
//    GAUSSVIEW(scene);
//
//    return app.exec();
    
    
}


//
//
//
//typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
//typedef Eigen::Triplet<double> T;
//void insertCoefficient(int id, int i, int j, double w, std::vector<T>& coeffs,
//                       Eigen::VectorXd& b, const Eigen::VectorXd& boundary)
//{
//    int n = int(boundary.size());
//    int id1 = i+j*n;
//    if(i==-1 || i==n) b(id) -= w * boundary(j); // constrained coefficient
//    else  if(j==-1 || j==n) b(id) -= w * boundary(i); // constrained coefficient
//    else  coeffs.push_back(T(id,id1,w));              // unknown coefficient
//}
//void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int n)
//{
//    b.setZero();
//    Eigen::ArrayXd boundary = Eigen::ArrayXd::LinSpaced(n, 0,M_PI).sin().pow(2);
//    for(int j=0; j<n; ++j)
//    {
//        for(int i=0; i<n; ++i)
//        {
//            int id = i+j*n;
//            insertCoefficient(id, i-1,j, -1, coefficients, b, boundary);
//            insertCoefficient(id, i+1,j, -1, coefficients, b, boundary);
//            insertCoefficient(id, i,j-1, -1, coefficients, b, boundary);
//            insertCoefficient(id, i,j+1, -1, coefficients, b, boundary);
//            insertCoefficient(id, i,j,    4, coefficients, b, boundary);
//        }
//    }
//}

//
//void saveAsBitmap(const Eigen::VectorXd& x, int n, const char* filename)
//{
//    Eigen::Array<unsigned char,Eigen::Dynamic,Eigen::Dynamic> bits = (x*255).cast<unsigned char>();
//    QImage img(bits.data(), n,n,QImage::Format_Indexed8);
//    img.setColorCount(256);
//    for(int i=0;i<256;i++) img.setColor(i,qRgb(i,i,i));
//    img.save(filename);
//}
//
