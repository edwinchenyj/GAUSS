#include <functional>

#include <Qt3DIncludes.h>
#include <GaussIncludes.h>
#include <FEMIncludes.h>

//Any extra things I need such as constraints
#include <ConstraintFixedPoint.h>
#include <TimeStepperERE.h>
#include <TimeStepperSI.h>
#include <TimeStepperIM.h>
#include <TimeStepperBE.h>
#include <TimeStepperSIERE.h>
#include <ExponentialIMEX.h>
#include <fstream>
#include <igl/boundary_facets.h>

#include <unsupported/Eigen/MatrixFunctions>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include <sys/types.h>
#include <sys/stat.h>

using namespace Gauss;
using namespace FEM;
using namespace ParticleSystem; //For Force Spring

/* Tetrahedral finite elements */
typedef PhysicalSystemFEM<double, NeohookeanTet> FEMLinearTets;

typedef World<double, std::tuple<FEMLinearTets *>,
std::tuple<ForceSpringFEMParticle<double> *, ForceParticlesGravity<double> *>,
std::tuple<ConstraintFixedPoint<double> *> > MyWorld;

typedef TimeStepperERE<double, AssemblerParallel<double, AssemblerEigenSparseMatrix<double>>, AssemblerParallel<double, AssemblerEigenVector<double>> > MyTimeStepperERE;


typedef TimeStepperIM<double, AssemblerParallel<double, AssemblerEigenSparseMatrix<double>>, AssemblerParallel<double, AssemblerEigenVector<double>> > MyTimeStepperIM;


typedef TimeStepperSIERE<double, AssemblerParallel<double, AssemblerEigenSparseMatrix<double>>, AssemblerParallel<double, AssemblerEigenVector<double>> > MyTimeStepperSIERE;


typedef TimeStepperSI<double, AssemblerParallel<double, AssemblerEigenSparseMatrix<double>>, AssemblerParallel<double, AssemblerEigenVector<double>> > MyTimeStepperSI;


typedef TimeStepperBE<double, AssemblerParallel<double, AssemblerEigenSparseMatrix<double>>, AssemblerParallel<double, AssemblerEigenVector<double>> > MyTimeStepperBE;

Eigen::MatrixXd V;
Eigen::MatrixXi F;

char **arg_list;
unsigned int istep;

void preStepCallback(MyWorld &world) {
}

int main(int argc, char **argv) {

    
    std::cout<<"Test Neohookean FEM\n";
    
    //Setup Physics
    MyWorld world;
    
    arg_list = argv;
    clock_t t; // time used for clock
    
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
    std::string meshname = "/meshesTetWild/brick_surf/brick_surf_5";
    
    
    //    default parameters
    double youngs = 2e6;
    double poisson = 0.45;
    int const_profile = 0;
    double const_tol = 1e-2;
    std::string initial_def = "0";
    int num_steps = 10;
    double step_size = 1e-2;
    int const_dir = 0; // constraint direction. 0 for x, 1 for y, 2 for z
    double a = 0.0;
    double b = -0.0001;
    std::string integrator = "SIERE";
    //    parameters
    if(argc > 1)
        meshname = argv[1];
    if(argc > 2)
        youngs = atof(argv[2]);
    if(argc > 3)
        const_profile = atoi(argv[3]);
    if(argc > 4)
        const_tol = atof(argv[4]);
    if(argc > 5)
        initial_def = (argv[5]);
    if(argc > 6)
        num_steps = atoi(argv[6]);
    if(argc > 7)
        step_size = atof(argv[7]);
    if(argc > 8)
        const_dir = atoi(argv[8]); // constraint direction. 0 for x, 1 for y, 2 for z
    if(argc > 9)
        a = atof(argv[9]);
    if(argc > 10)
        b = atof(argv[10]);
    
    readTetgen(V, F, dataDir()+meshname +".node", dataDir()+meshname+".ele");
    
    std::string::size_type found = meshname.find_last_of(kPathSeparator);
    //    acutal name for the mesh, no path
    std::string meshnameActual = meshname.substr(found+1);
    
    FEMLinearTets *test = new FEMLinearTets(V,F);
    Eigen::MatrixXi surfF;
    igl::boundary_facets(F,surfF);
    
    // projection matrix for constraints
    Eigen::SparseMatrix<double> P;
    
    
    world.addSystem(test);
    
    std::vector<ConstraintFixedPoint<double> *> movingConstraints;
    
    // constraint switch
    if ((const_profile) == 0) {
        cout<<"Setting zero gravity..."<<endl;
        //            zero gravity
        Eigen::Vector3x<double> g;
        g(0) = 0;
        g(1) = 0;
        g(2) = 0;
        
        for(unsigned int iel=0; iel<test->getImpl().getF().rows(); ++iel) {
            
            test->getImpl().getElement(iel)->setGravity(g);
            
        }
        
        world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
        
        //            set the projection matrix to identity because there is no constraint to project
        //            Eigen::SparseMatrix<double> P;
        P.resize(V.rows()*3,V.rows()*3);
        P.setIdentity();
        //            std::cout<<P.rows();
        //            no constraints
    }
    else if(const_profile == 1)
    {
        cout<<"Building fix constraint projection matrix"<<endl;
        //    default constraint
        fixDisplacementMin(world, test,const_dir,const_tol);
        world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
        
        Eigen::VectorXi indices;
        // construct the projection matrix for stepper
        std::string constraint_file_name = "data/" + meshnameActual + "_const" + std::to_string(const_profile) + "_" +std::to_string(const_dir)+"_"+std::to_string(const_tol)+".mtx";
        if(!Eigen::loadMarketVector(indices,constraint_file_name))
        {
            cout<<"File does not exist, creating new file..."<<endl;
            indices = minVertices(test, const_dir,const_tol);
            Eigen::saveMarketVector(indices,constraint_file_name);
        }
        
        P = fixedPointProjectionMatrix(indices, *test,world);
    }
    else if(const_profile == 2)
    {
        //            zero gravity
        cout<<"Setting zero gravity..."<<endl;
        Eigen::Vector3x<double> g;
        g(0) = 0;
        g(1) = 0;
        g(2) = 0;
        
        for(unsigned int iel=0; iel<test->getImpl().getF().rows(); ++iel) {
            
            test->getImpl().getElement(iel)->setGravity(g);
            
        }
        
        world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
        
        Eigen::VectorXi indices;
        // construct the projection matrix for stepper
        std::string constraint_file_name = "data/" + meshnameActual + "_const" + std::to_string(const_profile) + "_" +std::to_string(const_dir)+"_"+std::to_string(const_tol)+".mtx";
        cout<<"Loading moving vertices and setting projection matrix..."<<endl;
        if(!Eigen::loadMarketVector(indices,constraint_file_name))
        {
            cout<<"File does not exist, creating new file..."<<endl;
            indices = minVertices(test, const_dir,const_tol);
            Eigen::saveMarketVector(indices,constraint_file_name);
        }
        P = fixedPointProjectionMatrix(indices, *test,world);
        
        for(unsigned int ii=0; ii<indices.rows(); ++ii) {
            movingConstraints.push_back(new ConstraintFixedPoint<double>(&test->getQ()[indices[ii]], Eigen::Vector3d(0,0,0)));
            world.addConstraint(movingConstraints[ii]);
        }
        
        fixDisplacementMin(world, test,const_dir,const_tol);
        
    }
    else if (const_profile == 4 || const_profile == 5 || const_profile == 6 || const_profile == 7 || const_profile == 8)
    {
        //            zero gravity
        cout<<"Setting zero gravity..."<<endl;
        Eigen::Vector3x<double> g;
        g(0) = 0;
        g(1) = 0;
        g(2) = 0;
        
        for(unsigned int iel=0; iel<test->getImpl().getF().rows(); ++iel) {
            
            test->getImpl().getElement(iel)->setGravity(g);
            
        }
        
        world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
        
        Eigen::VectorXi indices;
        // construct the projection matrix for stepper
        std::string constraint_file_name = "data/" + meshnameActual + "_const" + std::to_string(const_profile) + "_" +std::to_string(const_dir)+"_"+std::to_string(const_tol)+".mtx";
        cout<<"Setting moving constraints and constrainting projection matrix"<<endl;
        cout<<"Loading moving vertices and setting projection matrix..."<<endl;
        if(!Eigen::loadMarketVector(indices,constraint_file_name))
        {
            cout<<"File does not exist, creating new file..."<<endl;
            indices = minVertices(test, const_dir,const_tol);
            Eigen::saveMarketVector(indices,constraint_file_name);
        }
        
        P = fixedPointProjectionMatrix(indices, *test,world);
        
        //            movingVerts = minVertices(test, const_dir, const_tol);//indices for moving parts
        
        for(unsigned int ii=0; ii<indices.rows(); ++ii) {
            movingConstraints.push_back(new ConstraintFixedPoint<double>(&test->getQ()[indices[ii]], Eigen::Vector3d(0,0,0)));
            world.addConstraint(movingConstraints[ii]);
        }
        fixDisplacementMin(world, test,const_dir,const_tol);
        
        
    }
    else
    {
        std::cout<<"warning: wrong constraint profile\n";
    }
    
    
    // set material
    for(unsigned int iel=0; iel<test->getImpl().getF().rows(); ++iel) {
        
        test->getImpl().getElement(iel)->setParameters(youngs, poisson);
        
    }
    
    
    auto q = mapStateEigen(world);
    
    //    default to zero deformation
    q.setZero();
    cout<<"Setting initial deformation..."<<endl;
    if (strcmp(argv[5],"0")==0) {
        // if specified no initial deformation
        cout<<"No initial deformation"<<endl;
        q.setZero();
    }
    else
    {
        cout<<"Loading initial deformation"<<endl;
        // load the initial deformation (and velocity) from file)
        std::string qfileName(initial_def);
        Eigen::MatrixXd Vtemp = V;
        Eigen::VectorXd  tempv;
        cout<<"Loading initial deformation"<<qfileName<<endl;
        if(Eigen::loadMarketVector(tempv,qfileName))
        {
            std::cout<<"original state size "<<q.rows()<<"\nloaded state size "<<tempv.rows()<<endl;;
            q = tempv;
            
            unsigned int idxc = 0;
            
            // get the mesh position
            for(unsigned int vertexId=0;  vertexId < std::get<0>(world.getSystemList().getStorage())[0]->getGeometry().first.rows(); ++vertexId) {
                
                Vtemp(vertexId,0) += q(idxc);
                idxc++;
                Vtemp(vertexId,1) += q(idxc);
                idxc++;
                Vtemp(vertexId,2) += q(idxc);
                idxc++;
            }
            
            igl::writeOBJ("loadedpos.obj",Vtemp,surfF);
        }
        else{
            cout<<"can't load initial deformation\n";
            exit(1);
        }
    }
    
    
    MyTimeStepperERE stepper(step_size,P, a, b);
    
    unsigned int file_ind = 0;
    struct stat buf;
    unsigned int idxc;
    clock_t dt;
    clock_t total_t = 0.0;
        for(istep=0; istep<num_steps ; ++istep)
        {
            t = clock();
            stepper.step(world);
            dt = clock() - t;
            total_t += dt;
            // acts like the "callback" block for moving constraint
            if (const_profile == 2)
            {
                // constraint profile 2 will move some vertices
                //script some motion
                cout<<"Moving constrained vertices in y..."<<endl;
                for(unsigned int jj=0; jj<movingConstraints.size(); ++jj) {
                    
                    auto v_q = mapDOFEigen(movingConstraints[jj]->getDOF(0), world.getState());
                    Eigen::Vector3d new_q = (istep)*Eigen::Vector3d(0.0,-1.0/100,0.0);
                    v_q = new_q;
                    
                }
            }
            else if (const_profile == 4 )
            {
                cout<<"Moving constrained vertices in x..."<<endl;
                for(unsigned int jj=0; jj<movingConstraints.size(); ++jj) {
                    
                    auto v_q = mapDOFEigen(movingConstraints[jj]->getDOF(0), world.getState());
                    //
                    if ((istep) < 50) {
                        Eigen::Vector3d new_q = (istep)*Eigen::Vector3d(-1.0/100,0.0,0.0);
                        v_q = new_q;
                    }
                    
                }
            }
            else if (const_profile == 5)
            {
                cout<<"Moving constraint vertices in y..."<<endl;
                for(unsigned int jj=0; jj<movingConstraints.size(); ++jj) {
                    
                    auto v_q = mapDOFEigen(movingConstraints[jj]->getDOF(0), world.getState());
                    //
                    if ((istep) < 50) {
                        Eigen::Vector3d new_q = (istep)*Eigen::Vector3d(0.0,-1.0/100,0.0);
                        v_q = new_q;
                    }
                    
                }
            }else if (const_profile == 6)
            {
                cout<<"Moving constrained vertices in z..."<<endl;
                for(unsigned int jj=0; jj<movingConstraints.size(); ++jj) {
                    
                    auto v_q = mapDOFEigen(movingConstraints[jj]->getDOF(0), world.getState());
                    //
                    if ((istep) < 50) {
                        Eigen::Vector3d new_q = (istep)*Eigen::Vector3d(0.0,0.0,-1.0/100);
                        v_q = new_q;
                    }
                    
                    
                }
            }else if (const_profile == 7)
            {
                cout<<"Moving constrained vertices using mouse motion"<<endl;
                Eigen::VectorXd Xvel;
                if(!Eigen::loadMarketVector(Xvel, "data/mouseXvel.mtx"))
                {
                    cout<<"fail loading mouse x motion"<<endl;
                }
                Eigen::VectorXd Yvel;
                if(!Eigen::loadMarketVector(Yvel, "data/mouseYvel.mtx"))
                    cout<<"fail loading mouse y motion"<<endl;
                Eigen::VectorXd Zvel;
                if(!Eigen::loadMarketVector(Zvel, "data/mouseZvel.mtx"))
                    cout<<"fail loading mouse z motion"<<endl;
                
                for(unsigned int jj=0; jj<movingConstraints.size(); ++jj) {
                    
                    auto v_q = mapDOFEigen(movingConstraints[jj]->getDOF(0), world.getState());
                    //
                    if ((istep) < 250) {
                        //                        Eigen::Vector3d new_q = (istep)*Eigen::Vector3d(0.0,0.0,-1.0/100);
                        v_q(0) += 0.1*Xvel(istep);
                        v_q(1) += 0.1*Yvel(istep);
                        v_q(2) += 0.1*Zvel(istep);
                    }
                    
                    
                }
            }else if (const_profile == 8)
            {
                cout<<"Moving constrained vertices using mouse motion"<<endl;
                Eigen::VectorXd Xvel;
                if(!Eigen::loadMarketVector(Xvel, "data/mouseXvel.mtx"))
                {
                    cout<<"fail loading mouse x motion"<<endl;
                }
                Eigen::VectorXd Yvel;
                if(!Eigen::loadMarketVector(Yvel, "data/mouseYvel.mtx"))
                    cout<<"fail loading mouse y motion"<<endl;
                Eigen::VectorXd Zvel;
                if(!Eigen::loadMarketVector(Zvel, "data/mouseZvel.mtx"))
                    cout<<"fail loading mouse z motion"<<endl;
                
                for(unsigned int jj=0; jj<movingConstraints.size(); ++jj) {
                    auto v_q = mapDOFEigen(movingConstraints[jj]->getDOF(0), world.getState());
                    if ((istep) < 250) {
                        if(Xvel(istep) <= 0){   v_q(0) += 0.5*std::max(Xvel(istep),-0.005);}
                        else{ v_q(0) += 0.5*std::min(Xvel(istep),0.005);}
                        if(Yvel(istep) <= 0){   v_q(1) += 0.5*std::max(Yvel(istep),-0.005);}
                        else{ v_q(1) += 0.5*std::min(Yvel(istep),0.005);}
                        if(Zvel(istep) <= 0){   v_q(2) += 0.5*std::max(Zvel(istep),-0.005);}
                        else{ v_q(2) += 0.5*std::min(Zvel(istep),0.005);}
                    }
                }
            }
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

    std::ofstream total_stepper_time;
    total_stepper_time.open ("total_stepper_time.txt");
    total_stepper_time<<total_t<<endl;
    total_stepper_time.close();
    
    
}


