//
//  exampleEigenFitCoRotRIM.cpp
//  Base
//
//  Created by Yu Ju Edwin Chen on 2018-12-12.
//

#include <functional>

#include <Qt3DIncludes.h>
#include <GaussIncludes.h>
#include <FEMIncludes.h>

//Any extra things I need such as constraints
#include <ConstraintFixedPoint.h>
#include <TimeStepperEigenFitSMWIM.h>
#include <resultsUtilities.h>
#include <EigenFit.h>
#include <fstream>
#include <igl/boundary_facets.h>
#include <igl/volume.h>

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



typedef World<double, std::tuple<FEMLinearTets *>,
std::tuple<ForceSpringFEMParticle<double> *, ForceParticlesGravity<double> *>,
std::tuple<ConstraintFixedPoint<double> *> > MyWorld;



typedef TimeStepperEigenFitSMWIM<double, AssemblerParallel<double, AssemblerEigenSparseMatrix<double>>, AssemblerParallel<double, AssemblerEigenVector<double>> > MyTimeStepper;

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
    
    
    
    std::cout<<"Test Neohookean FEM EigenFit with implicit midpoint solver\n";
    
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
    
    std::string cmeshname = "/meshesTetWild/brick_surf/brick_surf_5";
    std::string fmeshname = "/meshesTetWild/brick_surf/brick_surf_4";
    
    //    parameters
    double youngs = 2e6;
    double poisson = 0.45;
    double constraint_tol = 1e-2;
    int const_profile = 1;
    std::string initial_def = "0";
    int numSteps = 10;
    bool hausdorff = false;
    int num_modes = 5;
    int constraint_dir = 0; // constraint direction. 0 for x, 1 for y, 2 for z
    double step_size = 1e-2;
    int dynamic_flag = 0;
    double a = 0;
    double b = -1e-3;
    //        std::string ratio_manual_file = (argv[15]);
    int compute_frequency = 1; // not used anymore
    bool output_data_flag = false;
    bool simple_mass_flag = true;
    double mode_matching_tol = 0;
    parse_input(argc, argv, cmeshname, fmeshname, youngs, constraint_tol, const_profile, initial_def, numSteps, hausdorff, num_modes, constraint_dir, step_size, dynamic_flag, a, b, output_data_flag, simple_mass_flag, mode_matching_tol);
    
    
    readTetgen(V, F, dataDir()+cmeshname+".node", dataDir()+cmeshname+".ele");
    readTetgen(Vf, Ff, dataDir()+fmeshname+".node", dataDir()+fmeshname+".ele");
    
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
    
    EigenFit *test = new EigenFit(V,F,Vf,Ff,dynamic_flag,youngs,poisson,constraint_dir,constraint_tol, const_profile,hausdorff,num_modes,cmeshnameActual,fmeshnameActual,simple_mass_flag,mode_matching_tol);
    
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
    
    //        // set material
    cout<<"Setting Youngs and Poisson..."<<endl;
    for(unsigned int iel=0; iel<test->getImpl().getF().rows(); ++iel) {
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
    
    // initialize the state (position and velocity)
    auto q = mapStateEigen(world);
    q.setZero();
    // if static, should calculate the ratios here (before loading the deformation)
    // or if DAC (dynamic_flag == 6), calculate the first ratio
    if( num_modes != 0)
    {
        auto q = mapStateEigen<0>(world);
        //            cout<<"setting random perturbation to vertices"<<endl;
        q.setZero();
#ifdef COROT
//        q.setRandom();
////        cout<<"use wiggle "<<argv[argc-1]<<endl;
//        q *= 1e-4;
#endif
#ifdef ARAP
//        q.setRandom();
//        //        cout<<"use wiggle "<<argv[argc-1]<<endl;
//        q *= 1e-5;
#endif

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
        test->calculateEigenFitData(q,massMatrix,stiffnessMatrix,m_coarseUs,Y,Z);
        cout<<"static ratio calculated"<<endl;
        q.setZero();
    }
    if(dynamic_flag == 6)
    {
        double DAC_scalar = test->m_R(0);
        // set material
        cout<<"Resetting Youngs using DAC..."<<endl;
        for(unsigned int iel=0; iel<test->getImpl().getF().rows(); ++iel) {
#ifdef NH
            test->getImpl().getElement(iel)->setParameters(DAC_scalar * youngs, poisson);
#endif
#ifdef COROT
            test->getImpl().getElement(iel)->setParameters(DAC_scalar * youngs, poisson);
#endif
#ifdef LINEAR
            test->getImpl().getElement(iel)->setParameters(DAC_scalar * youngs, poisson);
#endif
#ifdef ARAP
            test->getImpl().getElement(iel)->setParameters(DAC_scalar * youngs);
#endif

            
        }
    }
    
    cout<<"Setting initial deformation..."<<endl;
    if (initial_def == "0") {
        // if specified no initial deformation
        cout<<"No initial deformation"<<endl;
        q.setZero();
    }
    else
    {
        cout<<"Loading initial deformation"<<endl;
        // load the initial deformation (and velocity) from file)
        std::string qfileName(initial_def);
        Eigen::VectorXd  tempv;
        cout<<"Loading initial deformation "<<qfileName<<endl;
        if(Eigen::loadMarketVector(tempv,qfileName))
        {
            
            std::cout<<"original state size "<<q.rows()<<"\nloaded state size "<<tempv.rows()<<endl;;
            q = tempv;
            
            unsigned int idxc = 0;
            
            q_state_to_position(tempv, Vtemp);
            
            igl::writeOBJ("loadedpos.obj",Vtemp,surfF);
        }
        else{
            cout<<"can't load initial deformation\n";
            exit(1);
        }
    }
    
    MyTimeStepper stepper(step_size,P,num_modes,a,b);
    
    // rayleigh damping. should not be here but this will do for now
    //         the number of steps to take
    
    unsigned int file_ind = 0;
    //        Eigen::MatrixXd coarse_eig_def;
    //        Eigen::VectorXd fine_eig_def;
    
    struct stat buf;
    unsigned int idxc;
    
    for(istep=0; istep<numSteps ; ++istep)
    {
        stepper.step(world);
        
        apply_moving_constraint(const_profile, world.getState(), movingConstraints, istep);
        // acts like the "callback" block for moving constraint
        
        //output data stream into text
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
        
        // check if the file already exist
        std::string filename = filename_number_padded("surfpos", file_ind,"obj");
        while (stat(filename.c_str(), &buf) != -1)
        {
            file_ind++;
            filename = filename_number_padded("surfpos", file_ind,"obj");
        }
        
        
        
        // rest pos for the coarse mesh getGeometry().first is V
        Eigen::VectorXd q = mapStateEigen(world);
        idxc = 0;
        Eigen::MatrixXd V_disp = std::get<0>(world.getSystemList().getStorage())[0]->getGeometry().first;
        // output mesh position with only surface mesh
        
        q_state_to_position(q,V_disp);
        
        
        // output mesh position with only surface mesh
        igl::writeOBJ(filename_number_padded("surfpos", file_ind,"obj"),V_disp,surfF);
        //
        
        // output eigenvalues
//        Eigen::saveMarketVector(test->coarseEig.second, filename_number_padded("eigenvalues",file_ind,"mtx"));
        // output eigenvalues
        Eigen::saveMarketVectorDat(test->coarseEig.second, filename_number_padded("eigenvalues",file_ind,"dat"));
        Eigen::saveMarketVectorDat(test->fineEig.second,filename_number_padded("feigenvalues",file_ind,"dat"));

        // output state
//        Eigen::saveMarketVectorDat(q, filename_number_padded("def",file_ind,"dat"));
//        Eigen::saveMarketVector(q, filename_number_padded("def",file_ind,"mtx"));

        
        if(num_modes != 0)
        {
            Eigen::saveMarketVectorDat(test->m_R, filename_number_padded("m_R",file_ind , "dat"));
            
            Eigen::saveMarketDat(test->coarseEig.first,filename_number_padded("coarse_def_modes", file_ind,"dat"));
        }
        
        // update step number;
        test->step_number++;
        cout<<"simulated frame: " << test->step_number<<endl;
        
    }
}




