#include <functional>

#include <Qt3DIncludes.h>
#include <GaussIncludes.h>
#include <FEMIncludes.h>

//Any extra things I need such as constraints
#include <ConstraintFixedPoint.h>
#include <TimeStepperEigenFitLinearSMWIM.h>
#include <EigenFitLinear.h>
#include <fstream>
#include <igl/boundary_facets.h>
#include <igl/volume.h>

using namespace Gauss;
using namespace FEM;
using namespace ParticleSystem; //For Force Spring

/* Tetrahedral finite elements */

//typedef physical entities I need

//typedef scene
//typedef PhysicalSystemFEM<double, NeohookeanHFixedTet> FEMLinearTets;
typedef PhysicalSystemFEM<double, LinearTet> FEMLinearTets;

typedef World<double, std::tuple<FEMLinearTets *>,
std::tuple<ForceSpringFEMParticle<double> *, ForceParticlesGravity<double> *>,
std::tuple<ConstraintFixedPoint<double> *> > MyWorld;

//typedef World<double, std::tuple<FEMLinearTets *,PhysicalSystemParticleSingle<double> *>,
//                      std::tuple<ForceSpringFEMParticle<double> *>,
//                      std::tuple<ConstraintFixedPoint<double> *> > MyWorld;
//typedef TimeStepperEigenFitSMW<double, AssemblerEigenSparseMatrix<double>, AssemblerEigenVector<double>> MyTimeStepper;
typedef TimeStepperEigenFitLinearSMWIM<double, AssemblerParallel<double, AssemblerEigenSparseMatrix<double>>, AssemblerParallel<double, AssemblerEigenVector<double>> > MyTimeStepper;

typedef Scene<MyWorld, MyTimeStepper> MyScene;


//typedef TimeStepperEigenFitSI<double, AssemblerParallel<double, AssemblerEigenSparseMatrix<double> >,
//AssemblerParallel<double, AssemblerEigenVector<double> > > MyTimeStepper;

//typedef Scene<MyWorld, MyTimeStepper> MyScene;

// used for preStepCallback. should be delete
std::vector<ConstraintFixedPoint<double> *> movingConstraints;
std::vector<ConstraintFixedPoint<double> *> fixedConstraints;
Eigen::VectorXi movingVerts;
Eigen::VectorXi fixedVerts;
Eigen::MatrixXd V, Vtemp, Vtemp2;
Eigen::MatrixXi F, Ftemp2;
Eigen::MatrixXi surfF;
Eigen::MatrixXi surfFf;

char **arg_list;
unsigned int istep;

void preStepCallback(MyWorld &world) {
}

int main(int argc, char **argv) {
    //    arg list
    //    1: full path to coarse mesh
    //    2: full path to fine mesh
    //    3: youngs modulus (SI unit)
    //    4: constraint threshold (for defualt constraint profile)
    //    5: constraint profile switch
    //    6: name of the initial deformation profile
    //    7. number of time steps
    //    8. flag for using hausdorff distance
    //    9. number of modes to modifies
    //    10. constraint direction
    //    11. step size
    
    
    std::cout<<"Test Linear FEM EigenFit with implicit midpoint solver \n";
    
    //Setup Physics
    MyWorld world;
    
    arg_list = argv;
    
    Eigen::MatrixXd Vf;
    Eigen::MatrixXi Ff;
    
    
    //    define the file separator base on system
    const char kPathSeparator =
#ifdef _WIN32
    '\\';
#else
    '/';
#endif
    
    if (argc > 1) {
        // must supply all 9 parameters
        
        std::string cmeshname = argv[1];
        std::string fmeshname = argv[2];
        
        readTetgen(V, F, dataDir()+cmeshname+".node", dataDir()+cmeshname+".ele");
        readTetgen(Vf, Ff, dataDir()+fmeshname+".node", dataDir()+fmeshname+".ele");
        
        //        Eigen::saveMarketVector(V, "V.mtx");
        
        
        Vtemp = V;
        //        find the surface mesh
        igl::boundary_facets(F,surfF);
        igl::boundary_facets(Ff,surfFf);
        
        std::string::size_type found = cmeshname.find_last_of(kPathSeparator);
        //    acutal name for the mesh, no path
        std::string cmeshnameActual = cmeshname.substr(found+1);
        
        //    acutal name for the mesh, no path
        std::string fmeshnameActual = fmeshname.substr(found+1);
        
        cout<<"Using coarse mesh "<<cmeshname<<endl;
        cout<<"Using fine mesh "<<fmeshname<<endl;
        //    parameters
        double youngs = atof(argv[3]);
        double poisson = 0.45;
        double constraint_tol = atof(argv[4]);
        int const_profile = atoi(argv[5]);
        int initial_def = atoi(argv[6]);
        int numSteps = atoi(argv[7]);
        bool hausdorff = atoi(argv[8]);
        int numModes = atoi(argv[9]);
        int constraint_dir = atoi(argv[10]); // constraint direction. 0 for x, 1 for y, 2 for z
        double step_size = atof(argv[11]);
        int dynamic_flag = atoi(argv[12]);
        double a = atof(argv[13]);
        double b = atof(argv[14]);
        std::string ratio_manual_file = (argv[15]);
        int compute_frequency = atoi(argv[16]);
        
        cout<<"Simulation parameters..."<<endl;
        cout<<"Youngs: "<<youngs<<endl;
        cout<<"Poisson: "<<poisson<<endl;
        cout<<"Constraint direction: "<<constraint_dir<<endl;
        cout<<"Constraint threshold: "<<constraint_tol<<endl;
        cout<<"Step size: "<<step_size<<endl;
        cout<<"Number of steps:"<<numSteps<<endl;
        cout<<"dynamic_flag: "<<dynamic_flag<<endl;
        cout<<"Rayleigh a: "<<a<<endl;
        cout<<"Rayleigh b: "<<b<<endl;
        cout<<"Number of modes: "<<numModes<<endl;
        cout<<"Constraint profile: "<<const_profile<<endl;
        cout<<"Initial deformation: "<<initial_def<<endl;
        cout<< "ratio manual file/feval manual file: "<< ratio_manual_file;
        cout<< "compute frequency (only used for dynamics): "<< compute_frequency;
        //
        // send the constraint switch in as well, or the fine embedded mesh. ugly
        // the flag indicate whether to recalculated or not
        // need to pass the material and constraint parameters to embedding too. need to do it again below. ugly
        // also use the last two args to determine how many modes to fix. have to put it here now. ugly
        Eigen::VectorXd ratio_manual(numModes);
        
        if(!Eigen::loadMarketVector(ratio_manual,"data/" + ratio_manual_file))
            ratio_manual.setZero();
        
        EigenFitLinear *test = new EigenFitLinear(V,F,Vf,Ff,dynamic_flag,youngs,poisson,constraint_dir,constraint_tol, const_profile,hausdorff,numModes,cmeshnameActual,fmeshnameActual,ratio_manual,compute_frequency);
        
        // TODO: set rayleigh damping. should not be here...
        test->a = a;
        test->b = b;
        
        world.addSystem(test);
        
        
        // projection matrix for constraints
        Eigen::SparseMatrix<double> P;
        
        
        
        
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
            fixDisplacementMin(world, test,constraint_dir,constraint_tol);
            world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
            
            Eigen::VectorXi indices;
            // construct the projection matrix for stepper
            std::string constraint_file_name = "data/" + cmeshnameActual + "_const" + std::to_string(const_profile) + "_" +std::to_string(constraint_dir)+"_"+std::to_string(constraint_tol)+".mtx";
            if(!Eigen::loadMarketVector(indices,constraint_file_name))
            {
                cout<<"File does not exist, creating new file..."<<endl;
                indices = minVertices(test, constraint_dir,constraint_tol);
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
            std::string constraint_file_name = "data/" + cmeshnameActual + "_const" + std::to_string(const_profile) + "_" +std::to_string(constraint_dir)+"_"+std::to_string(constraint_tol)+".mtx";
            cout<<"Loading moving vertices and setting projection matrix..."<<endl;
            if(!Eigen::loadMarketVector(indices,constraint_file_name))
            {
                cout<<"File does not exist, creating new file..."<<endl;
                indices = minVertices(test, constraint_dir,constraint_tol);
                Eigen::saveMarketVector(indices,constraint_file_name);
            }
            P = fixedPointProjectionMatrix(indices, *test,world);
            
            for(unsigned int ii=0; ii<indices.rows(); ++ii) {
                movingConstraints.push_back(new ConstraintFixedPoint<double>(&test->getQ()[indices[ii]], Eigen::Vector3d(0,0,0)));
                world.addConstraint(movingConstraints[ii]);
            }
            
            fixDisplacementMin(world, test,constraint_dir,constraint_tol);
            
            
            
            
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
            std::string constraint_file_name = "data/" + cmeshnameActual + "_const" + std::to_string(const_profile) + "_" +std::to_string(constraint_dir)+"_"+std::to_string(constraint_tol)+".mtx";
            cout<<"Setting moving constraints and constrainting projection matrix"<<endl;
            cout<<"Loading moving vertices and setting projection matrix..."<<endl;
            if(!Eigen::loadMarketVector(indices,constraint_file_name))
            {
                cout<<"File does not exist, creating new file..."<<endl;
                indices = minVertices(test, constraint_dir,constraint_tol);
                Eigen::saveMarketVector(indices,constraint_file_name);
            }
            
            P = fixedPointProjectionMatrix(indices, *test,world);
            
            //            movingVerts = minVertices(test, constraint_dir, constraint_tol);//indices for moving parts
            
            for(unsigned int ii=0; ii<indices.rows(); ++ii) {
                movingConstraints.push_back(new ConstraintFixedPoint<double>(&test->getQ()[indices[ii]], Eigen::Vector3d(0,0,0)));
                world.addConstraint(movingConstraints[ii]);
            }
            fixDisplacementMin(world, test,constraint_dir,constraint_tol);
            
            
        }
        else
        {
            std::cout<<"warning: wrong constraint profile\n";
        }
        
        // set material
        cout<<"Setting Youngs and Poisson..."<<endl;
        for(unsigned int iel=0; iel<test->getImpl().getF().rows(); ++iel) {
            
            test->getImpl().getElement(iel)->setParameters(youngs, poisson);
            
        }
        
        // initialize the state (position and velocity)
        auto q = mapStateEigen(world);
        
        // if static, should calculate the ratios here (before loading the deformation)
        // or if DAC (dynamic_flag == 6), calculate the first ratio
        if(dynamic_flag == 6 || (dynamic_flag == 0 && numModes != 0))
        {
            auto q_pos = mapStateEigen<0>(world);
            q_pos.setZero();
            
            //First two lines work around the fact that C++11 lambda can't directly capture a member variable.
            AssemblerParallel<double, AssemblerEigenSparseMatrix<double>> massMatrix;
            AssemblerParallel<double, AssemblerEigenSparseMatrix<double>> stiffnessMatrix;
            
            //get mass matrix
            ASSEMBLEMATINIT(massMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
            ASSEMBLELIST(massMatrix, world.getSystemList(), getMassMatrix);
            ASSEMBLEEND(massMatrix);
            
            
            //get stiffness matrix
            ASSEMBLEMATINIT(stiffnessMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
            ASSEMBLELIST(stiffnessMatrix, world.getSystemList(), getStiffnessMatrix);
            ASSEMBLELIST(stiffnessMatrix, world.getForceList(), getStiffnessMatrix);
            ASSEMBLEEND(stiffnessMatrix);
            
            (*massMatrix) = P*(*massMatrix)*P.transpose();
            (*stiffnessMatrix) = P*(*stiffnessMatrix)*P.transpose();
            
            
            //Subspace Eigenvectors and eigenvalues from this coarse mesh
            std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > m_coarseUs;
            // for SMW
            Eigen::MatrixXd Y;
            Eigen::MatrixXd Z;
            cout<<"calculating static ratio"<<endl;
            test->calculateEigenFitData(q_pos,massMatrix,stiffnessMatrix,m_coarseUs,Y,Z);
            cout<<"static ratio calculated"<<endl;
        }
        if(dynamic_flag == 6)
        {
            double DAC_scalar = test->m_R(0);
            // set material
            cout<<"Resetting Youngs using DAC..."<<endl;
            for(unsigned int iel=0; iel<test->getImpl().getF().rows(); ++iel) {
                
                test->getImpl().getElement(iel)->setParameters(DAC_scalar * youngs, poisson);
                
            }
            
        }
        cout<<"Setting initial deformation..."<<endl;
        if (strcmp(argv[6],"0")==0) {
            // if specified no initial deformation
            cout<<"No initial deformation"<<endl;
            q.setZero();
        }
        else
        {
            cout<<"Loading initial deformation"<<endl;
            // load the initial deformation (and velocity) from file)
            std::string qfileName(argv[6]);
            Eigen::VectorXd  tempv;
            cout<<"Loading initial deformation"<<qfileName<<endl;
            if(Eigen::loadMarketVector(tempv,qfileName))
            {
            std::cout<<"original state size "<<q.rows()<<"\nloaded state size "<<tempv.rows();
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
        
        MyTimeStepper stepper(step_size,P,numModes);
        
        // rayleigh damping. should not be here but this will do for now
        //         the number of steps to take
        
        unsigned int file_ind = 0;
        unsigned int mode = 0;
        unsigned int vert_idx = 0;
        //        Eigen::MatrixXd coarse_eig_def;
        //        Eigen::VectorXd fine_eig_def;
        
        struct stat buf;
        unsigned int idxc;
        
        for(istep=0; istep<numSteps ; ++istep)
        {
            if(dynamic_flag != 0)
            {
                test->ratio_calculated = false;
                cout<<"reset ratio calculation flag"<<endl;
                
                if(dynamic_flag == 3)
                {
                    //                    test->m_feval_manual =
                    //                    Eigen::VectorXd eigv;
                    if(~Eigen::loadMarketVector(test->m_feval_manual, "data/" + ratio_manual_file +std::to_string(istep) + ".mtx"))
                    {
                        //                        test->m_feval_manual.resize(numModes);
                        cout<<"ERROR: can't load eigenvalue from fine mesh for dynamics EigenFit.";
                        exit(1);
                    }
                }
                else if(dynamic_flag == 4)
                {
                    test->m_finepos_manual = ratio_manual_file;
                }
            }
            
            try{
                stepper.step(world);
                
                throw test->flag;
            }
            catch(int e)
            {
                if(e == 2)
                {
                    std::cout<<"hausdorff distance check failed\n";
                    return 1;
                }
            }
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
            }//output data stream into text
            // the following ones append one number to an opened file along the simulation
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
            // output mesh position with only surface mesh
            // igl::writeOBJ("surfpos_rest" + std::to_string(file_ind) + ".obj",V_disp,surfF);
            
            
            // get the mesh position
            for(unsigned int vertexId=0;  vertexId < std::get<0>(world.getSystemList().getStorage())[0]->getGeometry().first.rows(); ++vertexId) {
                
                V_disp(vertexId,0) += q(idxc);
                idxc++;
                V_disp(vertexId,1) += q(idxc);
                idxc++;
                V_disp(vertexId,2) += q(idxc);
                idxc++;
            }
            
            test->m_Vc_current = V_disp;
            
            //             output mesh position with elements
            igl::writeOBJ("pos" + std::to_string(file_ind) + ".obj",V_disp,std::get<0>(world.getSystemList().getStorage())[0]->getGeometry().second);
            //
            //            std::string cdeffilename = "cdef"+ std::to_string(file_ind) + "_" + std::to_string(youngs) + "_" + std::to_string(poisson) + "_" + std::to_string(const_profile) + "_" + std::to_string(constraint_dir) + "_" + std::to_string(constraint_tol) + ".mtx";
            //            Eigen::saveMarket(V_disp,cdeffilename);
            //
            //
            //            igl::readOBJ("pos33.obj",Vtemp2, Ftemp2);
            //
            //            cout<<Vtemp2.rows()<<endl;
            //            cout<<Vtemp2.cols()<<endl;
            //            cout<<Ftemp2.rows()<<endl;
            //            cout<<Ftemp2.cols()<<endl;
            //
            //            cout<<V.rows()<<endl;
            //            cout<<V.cols()<<endl;
            //            cout<<F.rows()<<endl;
            //            cout<<F.cols()<<endl;
            
            // output mesh position with only surface mesh
            igl::writeOBJ("surfpos" + std::to_string(file_ind) + ".obj",V_disp,surfF);
            //
            Eigen::VectorXd coarse_eig_def;
            //            if(dynamic_flag == 2)
            //            {
            //                V_disp = test->V_reset;
            //                for(int mode = 0; mode < numModes; mode++)
            //                {
            //                    V_disp = test->V_reset;
            //                    igl::writeOBJ("restart_coarse_restshape_" + std::to_string(mode) + "_" + std::to_string(file_ind) + ".obj",V_disp,surfF);
            //
            //                    idxc = 0;
            //                    coarse_eig_def = (P.transpose()*((test->m_coarseUs_restart).first.col(mode))).transpose();
            //                    // get the mesh position
            //                    for(unsigned int vertexId=0;  vertexId < std::get<0>(world.getSystemList().getStorage())[0]->getGeometry().first.rows(); ++vertexId) {
            //
            //                        V_disp(vertexId,0) += coarse_eig_def(idxc);
            //                        idxc++;
            //                        V_disp(vertexId,1) += coarse_eig_def(idxc);
            //                        idxc++;
            //                        V_disp(vertexId,2) += coarse_eig_def(idxc);
            //                        idxc++;
            //                    }
            //                    igl::writeOBJ("restart_coarse_eigenmode" + std::to_string(mode) + "_" + std::to_string(file_ind) + ".obj",V_disp,surfF);
            //                }
            //
            //
            //            }
            
            // output eigenvalues
            Eigen::saveMarketVector(test->coarseEig.second, "eigenvalues" + std::to_string(file_ind)+ ".mtx");
            // output eigenvalues
            Eigen::saveMarketVector(test->fineEig.second, "fineeigenvalues" + std::to_string(file_ind)+ ".mtx");
            //
            //            if(dynamic_flag == 2)
            //            {
            //                cout<<"Writing restart coarse eigenvalues into files"<<endl;
            //                // output eigenvalues
            //                Eigen::saveMarketVector(test->m_coarseUs_restart.second, "restart_eigenvalues" + std::to_string(file_ind)+ ".mtx");
            //                // output eigenvalues
            //                Eigen::saveMarketVector(test->m_fineUs_restart.second, "restart_fineeigenvalues" + std::to_string(file_ind)+ ".mtx");
            //            }
            
            Eigen::MatrixXd Vf_disp;
            if (numModes != 0)
            {// declare variable for fine mesh rest pos
                // embedded V
                auto fine_q = mapStateEigen<0>(test->getFineWorld());
                fine_q = (*(test->N)) * q.head(q.rows()/2);
                idxc = 0; // reset index counter
                Vf_disp = std::get<0>(test->getFineWorld().getSystemList().getStorage())[0]->getGeometry().first;
                // output mesh position with only surface mesh
                for(unsigned int vertexId=0;  vertexId < std::get<0>(test->getFineWorld().getSystemList().getStorage())[0]->getGeometry().first.rows(); ++vertexId) {
                    
                    Vf_disp(vertexId,0) += fine_q(idxc);
                    idxc++;
                    Vf_disp(vertexId,1) += fine_q(idxc);
                    idxc++;
                    Vf_disp(vertexId,2) += fine_q(idxc);
                    idxc++;
                }
                
                igl::writeOBJ("finepos" + std::to_string(file_ind) + ".obj",Vf_disp,std::get<0>(test->getFineWorld().getSystemList().getStorage())[0]->getGeometry().second);
                // output mesh position with only surface mesh
                igl::writeOBJ("finesurfpos" + std::to_string(file_ind) + ".obj",Vf_disp,surfFf);
                test->m_Vf_current = Vf_disp;
            }
            
            //
            //            Eigen::VectorXd fine_eig_def;
            //            if(dynamic_flag == 2)
            //            {
            //                Vf_disp = test->Vf_reset;
            //                for(int mode = 0; mode < numModes; mode++)
            //                {
            //                    Vf_disp = test->Vf_reset;
            //                    igl::writeOBJ("restart_fine_restshape_" + std::to_string(mode) + "_" + std::to_string(file_ind) + ".obj",Vf_disp,surfFf);
            //
            //                    idxc = 0;
            //                    fine_eig_def = (test->m_fineP.transpose()*((test->m_fineUs_restart).first.col(mode))).transpose();
            //                    // get the mesh position
            //                    for(unsigned int vertexId=0;  vertexId < std::get<0>(test->getFineWorld().getSystemList().getStorage())[0]->getGeometry().first.rows(); ++vertexId) {
            //
            //                        Vf_disp(vertexId,0) += fine_eig_def(idxc);
            //                        idxc++;
            //                        Vf_disp(vertexId,1) += fine_eig_def(idxc);
            //                        idxc++;
            //                        Vf_disp(vertexId,2) += fine_eig_def(idxc);
            //                        idxc++;
            //                    }
            //                    igl::writeOBJ("restart_fine_eigenmode" + std::to_string(mode) + "_" + std::to_string(file_ind) + ".obj",Vf_disp,surfFf);
            //                }
            //            }
            // update step number;
            test->step_number++;
            cout<<"simulated frame: " << test->step_number<<endl;
        }
        
    }
    else
    {
        // using all default paramters for eigenfit
        
        //    default example meshes
        std::string cmeshname = "/meshesTetWild/brick_surf/brick_surf_4";
        std::string fmeshname = "/meshesTetWild/brick_surf/brick_surf_3";
        
        readTetgen(V, F, dataDir()+cmeshname+".node", dataDir()+cmeshname+".ele");
        readTetgen(Vf, Ff, dataDir()+fmeshname+".node", dataDir()+fmeshname+".ele");
        
        std::string::size_type found = cmeshname.find_last_of(kPathSeparator);
        //    acutal name for the mesh, no path
        std::string cmeshnameActual = cmeshname.substr(found+1);
        
        
        //    default parameters
        double youngs = 2e5;
        double poisson = 0.45;
        int constraint_dir = 0; // constraint direction. 0 for x, 1 for y, 2 for z
        double constraint_tol = 1e-2;
        int dynamic_flag = false;
        int const_profile = 0;
        bool hausdorff = false;
        int numModes = 3;
        double a = 0.0;
        double b = -0.01;
        
        
        // no constraint switch so just create the eigenfit obj with constraint switch set to 1
        // the flag indicate whether to recalculated or not
        // need to pass the material and constraint parameters to embedding too. need to do it again below. ugly
        // also use the last two args to determine how many modes to fix. default not using hausdorff distance, and use 10 modes. have to put it here now.  ugly
        //        EigenFit *test = new EigenFit(V,F,Vf,Ff,dynamic_flag,youngs,poisson,constraint_dir,constraint_tol, const_profile,hausdorff,numModes,cmeshnameActual,fmeshnameActual);
        //        std::string::size_type found = cmeshname.find_last_of(kPathSeparator);
        //    acutal name for the mesh, no path
        //        std::string cmeshnameActual = cmeshname.substr(found+1);
        
        //    acutal name for the mesh, no path
        std::string fmeshnameActual = fmeshname.substr(found+1);
        Eigen::VectorXd ratio_manual(numModes);
        ratio_manual.setZero();
        int compute_frequency = 0;
        //        EigenFit *test = new EigenFit(V,F,Vf,Ff,dynamic_flag,youngs,poisson,constraint_dir,constraint_tol, const_profile,hausdorff,numModes," "," ");
        EigenFitLinear *test = new EigenFitLinear(V,F,Vf,Ff,dynamic_flag,youngs,poisson,constraint_dir,constraint_tol, const_profile,hausdorff,numModes,cmeshnameActual,fmeshnameActual,ratio_manual,compute_frequency);
        
        
        // set material
        for(unsigned int iel=0; iel<test->getImpl().getF().rows(); ++iel) {
            
            test->getImpl().getElement(iel)->setParameters(youngs, poisson);
            
        }
        
        world.addSystem(test);
        
        world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
        // IMPORTANT, need to finalized before fix boundary
        
        //    default constraint
        fixDisplacementMin(world, test,constraint_dir,constraint_tol);
        
        // construct the projection matrix for stepper
        Eigen::VectorXi indices = minVertices(test, constraint_dir,constraint_tol);
        Eigen::SparseMatrix<double> P = fixedPointProjectionMatrix(indices, *test,world);
        
        
        auto q = mapStateEigen(world);
        
        //    default to zero deformation
        q.setZero();
        
        
        MyTimeStepper stepper(0.01,P,numModes);
        
        //Display
        QGuiApplication app(argc, argv);
        
        MyScene *scene = new MyScene(&world, &stepper, preStepCallback);
        GAUSSVIEW(scene);
        
        return app.exec();
        
    }
    
}