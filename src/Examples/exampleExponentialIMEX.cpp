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
#include <resultsUtilities.h>
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

//typedef physical entities I need

////typedef scene
////build specific principal stretch material
template<typename DataType, typename ShapeFunction>
using  EnergyPSNHHFixed = EnergyPrincipalStretchHFixed<DataType, ShapeFunction, PSNeohookean>;

template<typename DataType, typename ShapeFunction>
using  EnergyPSARAPHFixed = EnergyPrincipalStretchHFixed<DataType, ShapeFunction, PSARAP>;

template<typename DataType, typename ShapeFunction>
using  EnergyPSCoRotHFixed = EnergyPrincipalStretchHFixed<DataType, ShapeFunction, PSCorotatedLinear>;

//
///* Tetrahedral finite elements */
template<typename DataType>
using FEMPSCoRotTet = FEMPrincipalStretchTet<DataType, EnergyPSCoRotHFixed>; //Change EnergyPSCoRot to any other energy defined above to try out other marterials

template<typename DataType>
using FEMPSARAPTet = FEMPrincipalStretchTet<DataType, EnergyPSARAPHFixed>; //Change EnergyPSCoRot

template<typename DataType>
using FEMPSNHTet = FEMPrincipalStretchTet<DataType, EnergyPSNHHFixed>; //Change EnergyPSCoRot



#ifdef NH
typedef PhysicalSystemFEM<double, NeohookeanHFixedTet> FEMLinearTets;
#endif

#ifdef COROT
typedef PhysicalSystemFEM<double, FEMPSCoRotTet> FEMLinearTets;
#endif

#ifdef ARAP
typedef PhysicalSystemFEM<double, FEMPSARAPTet> FEMLinearTets;
#endif

#ifdef LINEAR
typedef PhysicalSystemFEM<double, LinearTet> FEMLinearTets;
#endif

#ifdef STVK
typedef PhysicalSystemFEM<double, StvkTet> FEMLinearTets;
#endif

typedef World<double, std::tuple<FEMLinearTets *>,
std::tuple<ForceSpringFEMParticle<double> *, ForceParticlesGravity<double> *>,
std::tuple<ConstraintFixedPoint<double> *> > MyWorld;

typedef TimeStepperSIERE<double, AssemblerParallel<double, AssemblerEigenSparseMatrix<double>>, AssemblerParallel<double, AssemblerEigenVector<double>> > MyTimeStepper;

typedef Scene<MyWorld, MyTimeStepper> MyScene;

Eigen::MatrixXd V;
Eigen::MatrixXi F;

char **arg_list;
unsigned int istep;

void preStepCallback(MyWorld &world) {
}

int main(int argc, char **argv) {
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
    int const_profile = 1;
    double const_tol = 2e-1;
    std::string initial_def = "0";
    int num_steps = 5;
    int num_modes = 10; // used for SIERE only
    double step_size = 1e-2;
    int const_dir = 0; // constraint direction. 0 for x, 1 for y, 2 for z
    double a = 0.0;
    double b = -0.0001;
    std::string integrator = "SIERE";
    
    std::string hete_filename = "0";
    double hete_falloff_ratio = 1.0;
    
    double motion_multiplier = 1.0;

    
    //    parameters
    
    parse_input(argc, argv, meshname, youngs, const_tol, const_profile, initial_def, num_steps, num_modes, const_dir, step_size, a, b, integrator, hete_filename, hete_falloff_ratio, motion_multiplier);
    
    
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
        
        P.resize(V.rows()*3,V.rows()*3);
        P.setIdentity();
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
    else if (const_profile < 30)
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
    
    
    
    std::vector<double> stiffness_ratio;
    
    if (hete_filename != "0") {
        std::ifstream ifile(hete_filename, std::ios::in);
        //check to see that the file was opened correctly:
        if (!ifile.is_open()) {
            std::cerr << "There was a problem opening the input hete file!\n";
            exit(1);//exit or do additional error checking
        }
        
        double num = 0.0;
        //keep storing values from the text file so long as data exists:
        while (ifile >> num) {
            stiffness_ratio.push_back(num);
        }
        
#ifndef NDEBUG
        //verify that the scores were stored correctly:
        for (int i = 0; i < stiffness_ratio.size(); ++i) {
            std::cout << stiffness_ratio[i] << std::endl;
        }
#endif
        
        
        double low_stiffness = youngs/hete_falloff_ratio;
        
        if(stiffness_ratio.size() != test->getImpl().getF().rows())
        {
            std::cerr << "Hete file need the same number of tets!\n";
            exit(1);
        }
        
        //        // set material
        cout<<"Setting Youngs and Poisson..."<<endl;
        for(unsigned int iel=0; iel<test->getImpl().getF().rows(); ++iel)
        {
#ifdef NH
            test->getImpl().getElement(iel)->setParameters(low_stiffness + stiffness_ratio[iel] * (youngs - low_stiffness), poisson);
#endif
#ifdef COROT
            test->getImpl().getElement(iel)->setParameters(low_stiffness + stiffness_ratio[iel] * (youngs - low_stiffness), poisson);
#endif
#ifdef STVK
            test->getImpl().getElement(iel)->setParameters(low_stiffness + stiffness_ratio[iel] * (youngs - low_stiffness), poisson);
#endif
#ifdef LINEAR
            test->getImpl().getElement(iel)->setParameters(low_stiffness + stiffness_ratio[iel] * (youngs - low_stiffness), poisson);
#endif
#ifdef ARAP
            test->getImpl().getElement(iel)->setParameters(low_stiffness + stiffness_ratio[iel] * (youngs - low_stiffness));
#endif
            
        }
        
        
    }
    else
    {
        
        //        // set material
        cout<<"Setting Youngs and Poisson..."<<endl;
        for(unsigned int iel=0; iel<test->getImpl().getF().rows(); ++iel)
        {
#ifdef NH
            test->getImpl().getElement(iel)->setParameters(youngs, poisson);
#endif
#ifdef COROT
            test->getImpl().getElement(iel)->setParameters(youngs, poisson);
#endif
#ifdef LINEAR
            test->getImpl().getElement(iel)->setParameters(youngs, poisson);
#endif
#ifdef ARAP
            test->getImpl().getElement(iel)->setParameters(youngs);
#endif
            
        }
    }
    
    
    auto q = mapStateEigen(world);
    
    //    default to zero deformation
    q.setZero();
    cout<<"Setting initial deformation..."<<endl;
    if (initial_def=="0") {
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
    
    
    MyTimeStepper stepper(step_size,P, a, b,num_modes,integrator);
    stepper.getImpl().calculate_rest_stiffness(world);
    
    unsigned int file_ind = 0;
    struct stat buf;
    unsigned int idxc;
    clock_t dt, actual_t;
    clock_t total_t = 0.0;
    for(istep=0; istep<num_steps ; ++istep)
    {
        stepper.getImpl().step_number++;
        cout<<"simulating frame #" << stepper.getImpl().step_number<<endl;
        
        t = clock();
        stepper.step(world);
        dt = clock() - t;
        total_t += dt;
        actual_t = (double)((double)total_t)/CLOCKS_PER_SEC;
        
        if(!stepper.getImpl().step_success)
        {
            cout<<"Error: stepper fail at frame "<< stepper.getImpl().step_number <<" with parameters: "<<endl;
            cout<<"Using mesh: "<<meshname<<endl;
            cout<<"Using Youngs: "<<youngs<<endl;
            cout<<"Using constraint tolerance: "<<const_tol<<endl;
            cout<<"Using constriant profile: "<<const_profile<<endl;
            cout<<"Using initial deformation: "<<initial_def<<endl;
            cout<<"Using number of steps: "<< num_steps<<endl;
            cout<<"Using number of modes: "<<num_modes<<endl;
            cout<<"Using constraint direction: "<<const_dir<<endl;
            cout<<"Using step size: "<<step_size<<endl;
            cout<<"Using a: "<<a<<endl;
            cout<<"Using b: "<<b<<endl;
            cout<<"Using integrator: "<< integrator<<endl;
            std::ofstream myfile;
            myfile.open ("error_log.txt");
            
            myfile<<"Error: stepper fail at frame "<< stepper.getImpl().step_number <<" with parameters: "<<endl;
            myfile<<"Using mesh: "<<meshname<<endl;
            myfile<<"Using Youngs: "<<youngs<<endl;
            myfile<<"Using constraint tolerance: "<<const_tol<<endl;
            myfile<<"Using constriant profile: "<<const_profile<<endl;
            myfile<<"Using initial deformation: "<<initial_def<<endl;
            myfile<<"Using number of steps: "<< num_steps<<endl;
            myfile<<"Using number of modes: "<<num_modes<<endl;
            myfile<<"Using constraint direction: "<<const_dir<<endl;
            myfile<<"Using step size: "<<step_size<<endl;
            myfile<<"Using a: "<<a<<endl;
            myfile<<"Using b: "<<b<<endl;
            myfile<<"Using integrator: "<< integrator<<endl;
            myfile.close();
            
            return 1;
        }
        
        apply_moving_constraint(const_profile, world.getState(), movingConstraints, istep, motion_multiplier);
        
        std::ofstream ofile;
        //output data stream into text
        ofile.open("PE.txt", std::ios::app); //app is append which means it will put the text at the end
        ofile << std::get<0>(world.getSystemList().getStorage())[0]->getImpl().getPotentialEnergy(world.getState()) << std::endl;
        ofile.close();
        
        //output data stream into text
        ofile.open("Hamiltonian.txt", std::ios::app); //app is append which means it will put the text at the end
        ofile << std::get<0>(world.getSystemList().getStorage())[0]->getImpl().getEnergy(world.getState()) << std::endl;
        ofile.close();
//
//        ofile.open("KE.txt", std::ios::app); //app is append which means it will put the text at the end
//        ofile << std::get<0>(world.getSystemList().getStorage())[0]->getImpl().getEnergy(world.getState())  - std::get<0>(world.getSystemList().getStorage())[0]->getImpl().getPotentialEnergy(world.getState())<< std::endl;
//        ofile.close();
        
        
        ofile.open("it_count.txt", std::ios::app); //app is append which means it will put the text at the end
        ofile << stepper.getImpl().it_print<< std::endl;
        ofile.close();

        ofile.open("stepper_time_per_step.txt", std::ios::app); //app is append which means it will put the text at the end
        ofile << dt<< std::endl;
        ofile.close();

        
        file_ind = istep;
//        }
        
        // rest pos for the coarse mesh getGeometry().first is V
        Eigen::VectorXd q = mapStateEigen(world);
        idxc = 0;
        Eigen::MatrixXd V_disp = std::get<0>(world.getSystemList().getStorage())[0]->getGeometry().first;
        q_state_to_position(q,V_disp);
        
        // output mesh position with only surface mesh
        igl::writeOBJ(filename_number_padded("surfpos", file_ind,"obj"),V_disp,surfF);
        
        if(integrator == "SIERE")
        {
            Eigen::saveMarketVectorDat(stepper.getImpl().m_Us.second, filename_number_padded("eigenvalues",file_ind,"dat"));
        }
    }
    
    //    output the total time spent in the stepper
    std::ofstream total_stepper_time;
    total_stepper_time.open ("total_stepper_time.txt");
    total_stepper_time<<total_t<<endl;
    total_stepper_time<<actual_t<<endl;
    total_stepper_time.close();
    
}


