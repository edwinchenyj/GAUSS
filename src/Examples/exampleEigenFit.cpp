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
#include <igl/writePLY.h>
#include <time.h>

#include <fstream>
#include <vector>
#include <cstdlib>
#include <iostream>

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
    //Setup Physics
    MyWorld world;
    
    arg_list = argv;
    
    Eigen::MatrixXd Vf;
    Eigen::MatrixXi Ff;
    clock_t t; // time used for clock
    
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
    double youngs = 2e5;
    double poisson = 0.45;
    double const_tol = 1e-2;
    int const_profile = 1;
    std::string initial_def = "0";
    int num_steps = 10;
    bool haus = false;
    int num_modes = 5;
    int const_dir = 0; // constraint direction. 0 for x, 1 for y, 2 for z
    double step_size = 1e-2;
    int dynamic_flag = 0;
    double a = 0;
    double b = -1e-2;
    //        std::string ratio_manual_file = (argv[15]);
    bool output_data_flag = false;
    bool simple_mass_flag = true;
    double mode_matching_tol = 0.4;
    int calculate_matching_data_flag = 1;
    double init_mode_matching_tol = 0.4;
    bool init_eigenvalue_criteria= false;
    int init_eigenvalue_criteria_factor = 4;
    bool eigenfit_damping = true;
    
    std::string integrator = "IM";
    std::string hete_filename = "0";
    double hete_falloff_ratio = 1.0;
    double motion_multiplier = 1.0;
    
    parse_input(argc, argv, cmeshname, fmeshname, youngs, const_tol, const_profile, initial_def, num_steps, haus, num_modes, const_dir, step_size, dynamic_flag, a, b, output_data_flag, simple_mass_flag, mode_matching_tol, calculate_matching_data_flag, init_mode_matching_tol, init_eigenvalue_criteria, init_eigenvalue_criteria_factor, integrator, eigenfit_damping, hete_filename, hete_falloff_ratio, motion_multiplier);
    
    std::ofstream simfile;
    simfile.open ("sim_log.txt");
    
    simfile<<"Simulation with parameters: "<<endl;
    simfile<<"Using coarse mesh: "<<cmeshname<<endl;
    simfile<<"Using fine mesh: "<<fmeshname<<endl;
    simfile<<"Using Youngs: "<<youngs<<endl;
    simfile<<"Using constraint tolerance: "<<const_tol<<endl;
    simfile<<"Using constriant profile: "<<const_profile<<endl;
    simfile<<"Using initial deformation: "<<initial_def<<endl;
    simfile<<"Using number of steps: "<< num_steps<<endl;
    simfile<<"Using haus: "<<haus<<endl;
    simfile<<"Using number of modes: "<<num_modes<<endl;
    simfile<<"Using constraint direction: "<<const_dir<<endl;
    simfile<<"Using step size: "<<step_size<<endl;
    simfile<<"Using dynamic_flag: "<<dynamic_flag<<endl;
    simfile<<"Using a: "<<a<<endl;
    simfile<<"Using b: "<<b<<endl;
    simfile<<"Using output data flag: "<<output_data_flag<<endl;
    simfile<<"Using simple mass flag: "<<simple_mass_flag<<endl;
    simfile<<"Using mode matching tol: "<<mode_matching_tol<<endl;
    simfile<<"Using calculate matching data flag: "<<calculate_matching_data_flag<<endl;
    simfile<<"Using init mode matching tol: "<<init_mode_matching_tol<<endl;
    simfile<<"Using init eigenvalue criteria: "<<init_eigenvalue_criteria<<endl;
    simfile<<"Using init eigenvalue criteria factor: "<<init_eigenvalue_criteria_factor<<endl;
    simfile<<"Using integrator: "<< integrator<<endl;
    simfile<<"Using eigenfit damping: "<< eigenfit_damping<<endl;
    simfile<<"Using hete filename: "<< hete_filename<<endl;
    simfile<<"Using hete falloff ratio: "<< hete_falloff_ratio<<endl;
    simfile.close();
    
    
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
    
    
    if(hete_filename != "0")
    {
        cout<<"replacing "<< hete_filename << " to ";
        size_t index = 0;
        while (true) {
            /* Locate the substring to replace. */
            index = hete_filename.find(cmeshnameActual, index);
            if (index == std::string::npos) break;
            
            /* Make the replacement. */
            hete_filename.replace(index, cmeshnameActual.size(), fmeshnameActual);
            
            /* Advance index forward so the next iteration doesn't pick it up as well. */
            index += cmeshnameActual.size();
        }
        cout<< hete_filename<<endl;
    }
    
    
    EigenFit *test = new EigenFit(V,F,Vf,Ff,dynamic_flag,youngs,poisson,const_dir,const_tol, const_profile,haus,num_modes,cmeshnameActual,fmeshnameActual,simple_mass_flag,mode_matching_tol,hete_filename,hete_falloff_ratio);

    //  TODO: clean up here...   additional parameter goes here..
    // TODO: set rayleigh damping. should not be here...
    test->a = a;
    test->b = b;
    test->calculate_matching_data_flag = calculate_matching_data_flag;
    test->init_mode_matching_tol = init_mode_matching_tol;
    test->init_eigenvalue_criteria = init_eigenvalue_criteria;
    test->init_eigenvalue_criteria_factor = init_eigenvalue_criteria_factor;
    
    
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
        fixDisplacementMin(world, test,const_dir,const_tol);
        world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
        
        Eigen::VectorXi indices;
        // construct the projection matrix for stepper
        std::string constraint_file_name = "data/" + cmeshnameActual + "_const" + std::to_string(const_profile) + "_" +std::to_string(const_dir)+"_"+std::to_string(const_tol)+".mtx";
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
        g(1) = -9.8 * motion_multiplier;
        g(2) = 0;
        
        for(unsigned int iel=0; iel<test->getImpl().getF().rows(); ++iel) {
            
            test->getImpl().getElement(iel)->setGravity(g);
            
        }
        cout<<"Building fix constraint projection matrix"<<endl;
        //    default constraint
        fixDisplacementMin(world, test,const_dir,const_tol);
        world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
        
        Eigen::VectorXi indices;
        // construct the projection matrix for stepper
        std::string constraint_file_name = "data/" + cmeshnameActual + "_const" + std::to_string(const_profile) + "_" +std::to_string(const_dir)+"_"+std::to_string(const_tol)+".mtx";
        if(!Eigen::loadMarketVector(indices,constraint_file_name))
        {
            cout<<"File does not exist, creating new file..."<<endl;
            indices = minVertices(test, const_dir,const_tol);
            Eigen::saveMarketVector(indices,constraint_file_name);
        }
        
        P = fixedPointProjectionMatrix(indices, *test,world);
        
    }else if(const_profile == 3)
    {
        //            zero gravity
        cout<<"Setting zero gravity..."<<endl;
        Eigen::Vector3x<double> g;
        g(0) = -9.8 * motion_multiplier;
        g(1) = 0;
        g(2) = 0;
        
        for(unsigned int iel=0; iel<test->getImpl().getF().rows(); ++iel) {
            
            test->getImpl().getElement(iel)->setGravity(g);
            
        }
        cout<<"Building fix constraint projection matrix"<<endl;
        //    default constraint
        fixDisplacementMin(world, test,const_dir,const_tol);
        world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
        
        Eigen::VectorXi indices;
        // construct the projection matrix for stepper
        std::string constraint_file_name = "data/" + cmeshnameActual + "_const" + std::to_string(const_profile) + "_" +std::to_string(const_dir)+"_"+std::to_string(const_tol)+".mtx";
        if(!Eigen::loadMarketVector(indices,constraint_file_name))
        {
            cout<<"File does not exist, creating new file..."<<endl;
            indices = minVertices(test, const_dir,const_tol);
            Eigen::saveMarketVector(indices,constraint_file_name);
        }
        
        P = fixedPointProjectionMatrix(indices, *test,world);
        
        
        
        
    }else if(const_profile == 4)
    {
        //            zero gravity
        cout<<"Setting zero gravity..."<<endl;
        Eigen::Vector3x<double> g;
        g(0) = 0;
        g(1) = 9.8 * motion_multiplier;
        g(2) = 0;
        
        for(unsigned int iel=0; iel<test->getImpl().getF().rows(); ++iel) {
            
            test->getImpl().getElement(iel)->setGravity(g);
            
        }
        cout<<"Building fix constraint projection matrix"<<endl;
        //    default constraint
        fixDisplacementMin(world, test,const_dir,const_tol);
        world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
        
        Eigen::VectorXi indices;
        // construct the projection matrix for stepper
        std::string constraint_file_name = "data/" + cmeshnameActual + "_const" + std::to_string(const_profile) + "_" +std::to_string(const_dir)+"_"+std::to_string(const_tol)+".mtx";
        if(!Eigen::loadMarketVector(indices,constraint_file_name))
        {
            cout<<"File does not exist, creating new file..."<<endl;
            indices = minVertices(test, const_dir,const_tol);
            Eigen::saveMarketVector(indices,constraint_file_name);
        }
        
        P = fixedPointProjectionMatrix(indices, *test,world);
        
        
        
        
    }else if(const_profile == 5)
    {
        //            zero gravity
        cout<<"Setting zero gravity..."<<endl;
        Eigen::Vector3x<double> g;
        g(0) = 9.8 * motion_multiplier;
        g(1) = 0;
        g(2) = 0;
        
        for(unsigned int iel=0; iel<test->getImpl().getF().rows(); ++iel) {
            
            test->getImpl().getElement(iel)->setGravity(g);
            
        }
        cout<<"Building fix constraint projection matrix"<<endl;
        //    default constraint
        fixDisplacementMin(world, test,const_dir,const_tol);
        world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
        
        Eigen::VectorXi indices;
        // construct the projection matrix for stepper
        std::string constraint_file_name = "data/" + cmeshnameActual + "_const" + std::to_string(const_profile) + "_" +std::to_string(const_dir)+"_"+std::to_string(const_tol)+".mtx";
        if(!Eigen::loadMarketVector(indices,constraint_file_name))
        {
            cout<<"File does not exist, creating new file..."<<endl;
            indices = minVertices(test, const_dir,const_tol);
            Eigen::saveMarketVector(indices,constraint_file_name);
        }
        
        P = fixedPointProjectionMatrix(indices, *test,world);
        
    }else if(const_profile == 6)
    {
        //            zero gravity
        cout<<"Setting zero gravity..."<<endl;
        Eigen::Vector3x<double> g;
        g(0) = 0;
        g(1) = 0;
        g(2) = 9.8 * motion_multiplier;
        
        for(unsigned int iel=0; iel<test->getImpl().getF().rows(); ++iel) {
            
            test->getImpl().getElement(iel)->setGravity(g);
            
        }
        cout<<"Building fix constraint projection matrix"<<endl;
        //    default constraint
        fixDisplacementMin(world, test,const_dir,const_tol);
        world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
        
        Eigen::VectorXi indices;
        // construct the projection matrix for stepper
        std::string constraint_file_name = "data/" + cmeshnameActual + "_const" + std::to_string(const_profile) + "_" +std::to_string(const_dir)+"_"+std::to_string(const_tol)+".mtx";
        if(!Eigen::loadMarketVector(indices,constraint_file_name))
        {
            cout<<"File does not exist, creating new file..."<<endl;
            indices = minVertices(test, const_dir,const_tol);
            Eigen::saveMarketVector(indices,constraint_file_name);
        }
        
        P = fixedPointProjectionMatrix(indices, *test,world);
        
        
    }
    else if(const_profile == 7)
    {
        //            zero gravity
        cout<<"Setting zero gravity..."<<endl;
        Eigen::Vector3x<double> g;
        g(0) = 0;
        g(1) = 0;
        g(2) = -9.8 * motion_multiplier;
        
        for(unsigned int iel=0; iel<test->getImpl().getF().rows(); ++iel) {
            
            test->getImpl().getElement(iel)->setGravity(g);
            
        }
        cout<<"Building fix constraint projection matrix"<<endl;
        //    default constraint
        fixDisplacementMin(world, test,const_dir,const_tol);
        world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
        
        Eigen::VectorXi indices;
        // construct the projection matrix for stepper
        std::string constraint_file_name = "data/" + cmeshnameActual + "_const" + std::to_string(const_profile) + "_" +std::to_string(const_dir)+"_"+std::to_string(const_tol)+".mtx";
        if(!Eigen::loadMarketVector(indices,constraint_file_name))
        {
            cout<<"File does not exist, creating new file..."<<endl;
            indices = minVertices(test, const_dir,const_tol);
            Eigen::saveMarketVector(indices,constraint_file_name);
        }
        
        P = fixedPointProjectionMatrix(indices, *test,world);
        
        
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
        std::string constraint_file_name = "data/" + cmeshnameActual + "_const" + std::to_string(const_profile) + "_" +std::to_string(const_dir)+"_"+std::to_string(const_tol)+".mtx";
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
    
    
    if(hete_filename != "0")
    {
        cout<<"replacing "<< hete_filename << " to ";
        size_t index = 0;
        while (true) {
            /* Locate the substring to replace. */
            index = hete_filename.find(fmeshnameActual, index);
            if (index == std::string::npos) break;
            
            /* Make the replacement. */
            hete_filename.replace(index, fmeshnameActual.size(), cmeshnameActual);
            
            /* Advance index forward so the next iteration doesn't pick it up as well. */
            index += fmeshnameActual.size();
        }
        cout<< hete_filename<<endl;
    }
    
    std::vector<double> stiffness_ratio;
    
    if (hete_filename != "0") {
        std::ifstream ifile(hete_filename, std::ios::in);
        //check to see that the file was opened correctly:
        if (!ifile.is_open()) {
            std::cerr << "There was a problem opening the input file!\n";
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
    
    // initialize the state (position and velocity)
    auto q = mapStateEigen(world);
    q.setZero();
    // if static, should calculate the ratios here (before loading the deformation)
    // or if DAC (dynamic_flag == 6), calculate the first ratio
    if( num_modes != 0)
    {
        t = clock();
        auto q = mapStateEigen<0>(world);
        //            cout<<"setting random perturbation to vertices"<<endl;
        q.setZero();

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
        
        t = clock() -t;
        std::ofstream pre_calc_time_file;
        pre_calc_time_file.open ("pre_calc_time.txt");
        pre_calc_time_file<<t;
        pre_calc_time_file.close();
        
    
    }
    
    if(dynamic_flag == 6)
    {
        if(hete_filename!="0")
        {
            cout<<"can't you DAC with heterogenous material."<<endl;
            exit(1);
        }
        
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
            
            
            q_state_to_position(tempv, Vtemp);
            
            igl::writeOBJ("loadedpos.obj",Vtemp,surfF);
        }
        else{
            cout<<"can't load initial deformation\n";
            exit(1);
        }
    }
    
    MyTimeStepper stepper(step_size,P,num_modes,a,b, integrator);
    stepper.getImpl().eigenfit_damping = eigenfit_damping;
    // rayleigh damping. should not be here but this will do for now
    //         the number of steps to take
    
    unsigned int file_ind = 0;
    //        Eigen::MatrixXd coarse_eig_def;
    //        Eigen::VectorXd fine_eig_def;
    
//    struct stat buf;
    unsigned int idxc;
    clock_t dt;
    clock_t total_t = 0;
    double actual_t = 0.0;
    for(istep=0; istep<num_steps ; ++istep)
    {
        // update step number;
        test->step_number++;
        cout<<"simulating frame #" << test->step_number<<endl;
        t = clock();
        stepper.step(world);
        dt = clock() - t;
        total_t += dt;
        actual_t = ((double)total_t)/CLOCKS_PER_SEC;
        if(!stepper.getImpl().step_success)
        {
            cout<<"Error: stepper fail at frame "<< test->step_number <<" with parameters: "<<endl;
            cout<<"Using coarse mesh: "<<cmeshname<<endl;
            cout<<"Using fine mesh: "<<fmeshname<<endl;
            cout<<"Using Youngs: "<<youngs<<endl;
            cout<<"Using constraint tolerance: "<<const_tol<<endl;
            cout<<"Using constriant profile: "<<const_profile<<endl;
            cout<<"Using initial deformation: "<<initial_def<<endl;
            cout<<"Using number of steps: "<< num_steps<<endl;
            cout<<"Using haus: "<<haus<<endl;
            cout<<"Using number of modes: "<<num_modes<<endl;
            cout<<"Using constraint direction: "<<const_dir<<endl;
            cout<<"Using step size: "<<step_size<<endl;
            cout<<"Using dynamic_flag: "<<dynamic_flag<<endl;
            cout<<"Using a: "<<a<<endl;
            cout<<"Using b: "<<b<<endl;
            cout<<"Using output data flag: "<<output_data_flag<<endl;
            cout<<"Using simple mass flag: "<<simple_mass_flag<<endl;
            cout<<"Using mode matching tol: "<<mode_matching_tol<<endl;
            cout<<"Using calculate matching data flag: "<<calculate_matching_data_flag<<endl;
            cout<<"Using init mode matching tol: "<<init_mode_matching_tol<<endl;
            cout<<"Using init eigenvalue criteria: "<<init_eigenvalue_criteria<<endl;
            cout<<"Using init eigenvalue criteria factor: "<<init_eigenvalue_criteria_factor<<endl;
            cout<<"Using integrator: "<< integrator<<endl;
            cout<<"Using eigenfit damping: "<< eigenfit_damping<<endl;
            cout<<"Using hete filename: "<< hete_filename<<endl;
            cout<<"Using hete falloff ratio: "<< hete_falloff_ratio<<endl;

            std::ofstream myfile;
            myfile.open ("error_log.txt", std::ofstream::app);
            
            myfile<<"Error: stepper fail at frame "<< test->step_number <<" with parameters: "<<endl;
            myfile<<"Using coarse mesh: "<<cmeshname<<endl;
            myfile<<"Using fine mesh: "<<fmeshname<<endl;
            myfile<<"Using Youngs: "<<youngs<<endl;
            myfile<<"Using constraint tolerance: "<<const_tol<<endl;
            myfile<<"Using constriant profile: "<<const_profile<<endl;
            myfile<<"Using initial deformation: "<<initial_def<<endl;
            myfile<<"Using number of steps: "<< num_steps<<endl;
            myfile<<"Using haus: "<<haus<<endl;
            myfile<<"Using number of modes: "<<num_modes<<endl;
            myfile<<"Using constraint direction: "<<const_dir<<endl;
            myfile<<"Using step size: "<<step_size<<endl;
            myfile<<"Using dynamic_flag: "<<dynamic_flag<<endl;
            myfile<<"Using a: "<<a<<endl;
            myfile<<"Using b: "<<b<<endl;
            myfile<<"Using output data flag: "<<output_data_flag<<endl;
            myfile<<"Using simple mass flag: "<<simple_mass_flag<<endl;
            myfile<<"Using mode matching tol: "<<mode_matching_tol<<endl;
            myfile<<"Using calculate matching data flag: "<<calculate_matching_data_flag<<endl;
            myfile<<"Using init mode matching tol: "<<init_mode_matching_tol<<endl;
            myfile<<"Using init eigenvalue criteria: "<<init_eigenvalue_criteria<<endl;
            myfile<<"Using init eigenvalue criteria factor: "<<init_eigenvalue_criteria_factor<<endl;
            myfile<<"Using integrator: "<< integrator<<endl;
            myfile<<"Using eigenfit damping: "<< eigenfit_damping<<endl;
            myfile<<"Using hete filename: "<< hete_filename<<endl;
            myfile<<"Using hete falloff ratio: "<< hete_falloff_ratio<<endl;
            myfile.close();
            
            return 1;
        }
        
        if (test->eigenfit_data == 1) {
            cout<<"Initial mode matching missing. Two meshes too different. Need to change resolution."<<endl;
            
            std::ofstream myfile;
            myfile.open ("error_log.txt", std::ofstream::app);
            
            myfile<<"Initial mode matching missing. Two meshes too different. Need to change resolution."<<endl;
            myfile<<"Using coarse mesh: "<<cmeshname<<endl;
            myfile<<"Using fine mesh: "<<fmeshname<<endl;
            myfile<<"Using Youngs: "<<youngs<<endl;
            myfile<<"Using constraint tolerance: "<<const_tol<<endl;
            myfile<<"Using constriant profile: "<<const_profile<<endl;
            myfile<<"Using initial deformation: "<<initial_def<<endl;
            myfile<<"Using number of steps: "<< num_steps<<endl;
            myfile<<"Using haus: "<<haus<<endl;
            myfile<<"Using number of modes: "<<num_modes<<endl;
            myfile<<"Using constraint direction: "<<const_dir<<endl;
            myfile<<"Using step size: "<<step_size<<endl;
            myfile<<"Using dynamic_flag: "<<dynamic_flag<<endl;
            myfile<<"Using a: "<<a<<endl;
            myfile<<"Using b: "<<b<<endl;
            myfile<<"Using output data flag: "<<output_data_flag<<endl;
            myfile<<"Using simple mass flag: "<<simple_mass_flag<<endl;
            myfile<<"Using mode matching tol: "<<mode_matching_tol<<endl;
            myfile<<"Using calculate matching data flag: "<<calculate_matching_data_flag<<endl;
            myfile<<"Using init mode matching tol: "<<init_mode_matching_tol<<endl;
            myfile<<"Using init eigenvalue criteria: "<<init_eigenvalue_criteria<<endl;
            myfile<<"Using init eigenvalue criteria factor: "<<init_eigenvalue_criteria_factor<<endl;
            myfile<<"Using integrator: "<< integrator<<endl;
            myfile<<"Using eigenfit damping: "<< eigenfit_damping<<endl;
            myfile<<"Using hete filename: "<< hete_filename<<endl;
            myfile<<"Using hete falloff ratio: "<< hete_falloff_ratio<<endl;
            
            myfile.close();
#ifndef NOEXIT
            return 1;
#endif
        }
        else if (test->eigenfit_data == 3)
        {
            
            std::ofstream myfile;
            myfile.open ("error_log.txt", std::ofstream::app);
            cout<<"Eigenvalues change too much, Eigenfit won't work."<<endl;
            myfile<<"Eigenvalues change too much, Eigenfit won't work."<<endl;
            myfile<<"Using coarse mesh: "<<cmeshname<<endl;
            myfile<<"Using fine mesh: "<<fmeshname<<endl;
            myfile<<"Using Youngs: "<<youngs<<endl;
            myfile<<"Using constraint tolerance: "<<const_tol<<endl;
            myfile<<"Using constriant profile: "<<const_profile<<endl;
            myfile<<"Using initial deformation: "<<initial_def<<endl;
            myfile<<"Using number of steps: "<< num_steps<<endl;
            myfile<<"Using haus: "<<haus<<endl;
            myfile<<"Using number of modes: "<<num_modes<<endl;
            myfile<<"Using constraint direction: "<<const_dir<<endl;
            myfile<<"Using step size: "<<step_size<<endl;
            myfile<<"Using dynamic_flag: "<<dynamic_flag<<endl;
            myfile<<"Using a: "<<a<<endl;
            myfile<<"Using b: "<<b<<endl;
            myfile<<"Using output data flag: "<<output_data_flag<<endl;
            myfile<<"Using simple mass flag: "<<simple_mass_flag<<endl;
            myfile<<"Using mode matching tol: "<<mode_matching_tol<<endl;
            myfile<<"Using calculate matching data flag: "<<calculate_matching_data_flag<<endl;
            myfile<<"Using init mode matching tol: "<<init_mode_matching_tol<<endl;
            myfile<<"Using init eigenvalue criteria: "<<init_eigenvalue_criteria<<endl;
            myfile<<"Using init eigenvalue criteria factor: "<<init_eigenvalue_criteria_factor<<endl;
            myfile<<"Using integrator: "<< integrator<<endl;
            myfile<<"Using eigenfit damping: "<< eigenfit_damping<<endl;
            myfile<<"Using hete filename: "<< hete_filename<<endl;
            myfile<<"Using hete falloff ratio: "<< hete_falloff_ratio<<endl;
            
            myfile.close();
#ifndef NOEXIT
            return 1;
#endif
        }
        else if (test->eigenfit_data == 4)
        {
            
            std::ofstream myfile;
            myfile.open ("error_log.txt", std::ofstream::app);
            cout<<"Eigenvectors change too much, Eigenfit won't work."<<endl;
            myfile<<"Eigenvectors change too much, Eigenfit won't work."<<endl;
            myfile<<"Using coarse mesh: "<<cmeshname<<endl;
            myfile<<"Using fine mesh: "<<fmeshname<<endl;
            myfile<<"Using Youngs: "<<youngs<<endl;
            myfile<<"Using constraint tolerance: "<<const_tol<<endl;
            myfile<<"Using constriant profile: "<<const_profile<<endl;
            myfile<<"Using initial deformation: "<<initial_def<<endl;
            myfile<<"Using number of steps: "<< num_steps<<endl;
            myfile<<"Using haus: "<<haus<<endl;
            myfile<<"Using number of modes: "<<num_modes<<endl;
            myfile<<"Using constraint direction: "<<const_dir<<endl;
            myfile<<"Using step size: "<<step_size<<endl;
            myfile<<"Using dynamic_flag: "<<dynamic_flag<<endl;
            myfile<<"Using a: "<<a<<endl;
            myfile<<"Using b: "<<b<<endl;
            myfile<<"Using output data flag: "<<output_data_flag<<endl;
            myfile<<"Using simple mass flag: "<<simple_mass_flag<<endl;
            myfile<<"Using mode matching tol: "<<mode_matching_tol<<endl;
            myfile<<"Using calculate matching data flag: "<<calculate_matching_data_flag<<endl;
            myfile<<"Using init mode matching tol: "<<init_mode_matching_tol<<endl;
            myfile<<"Using init eigenvalue criteria: "<<init_eigenvalue_criteria<<endl;
            myfile<<"Using init eigenvalue criteria factor: "<<init_eigenvalue_criteria_factor<<endl;
            myfile<<"Using integrator: "<< integrator<<endl;
            myfile<<"Using eigenfit damping: "<< eigenfit_damping<<endl;
            myfile<<"Using hete filename: "<< hete_filename<<endl;
            myfile<<"Using hete falloff ratio: "<< hete_falloff_ratio<<endl;
            
            myfile.close();
#ifndef NOEXIT
            return 1;
#endif
        }
        
        apply_moving_constraint(const_profile, world.getState(), movingConstraints, istep, motion_multiplier);
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
        
        
        
        
        
        // rest pos for the coarse mesh getGeometry().first is V
        Eigen::VectorXd q = mapStateEigen(world);
        idxc = 0;
        Eigen::MatrixXd V_disp = std::get<0>(world.getSystemList().getStorage())[0]->getGeometry().first;
        // output mesh position with only surface mesh
        
        q_state_to_position(q,V_disp);
        
//        std::string filename = filename_number_padded("surfpos", file_ind,"obj");

        // output mesh position with only surface mesh
        igl::writeOBJ(filename_number_padded("surfpos", file_ind,"obj"),V_disp,surfF);
        igl::writePLY(filename_number_padded("surfpos", file_ind,"ply"),V_disp,surfF);
        
        //
        
        // output eigenvalues
//        Eigen::saveMarketVector(test->coarseEig.second, filename_number_padded("eigenvalues",file_ind,"mtx"));
        // output eigenvalues
        

        // output state
//        Eigen::saveMarketVectorDat(q, filename_number_padded("def",file_ind,"dat"));
//        Eigen::saveMarketVector(q, filename_number_padded("def",file_ind,"mtx"));

        
        if(num_modes != 0)
        {
            
//            Eigen::saveMarketDat(test->coarseEig.first,filename_number_padded("coarse_def_modes", file_ind,"dat"));
            
            Eigen::saveMarketVectorDat(test->coarseEig.second, filename_number_padded("eigenvalues",file_ind,"dat"));
            
            
            Eigen::saveMarketDat(test->dist_map, filename_number_padded("dist_map",file_ind,"dat"));
            Eigen::saveMarketVectorDat(test->matched_modes_list, filename_number_padded("matched_modes_list",file_ind,"dat"));
//            Eigen::saveMarketVectorDat(test->init_matched_modes_list,"init_matched_modes_list.dat");
            Eigen::saveMarketVectorDat(test->m_R_current,filename_number_padded("m_R_current",file_ind,"dat"));
            
        }
        
        file_ind++;
        
    }
//    soutput the total time spent in the stepper
    std::ofstream total_stepper_time;
    total_stepper_time.open ("total_stepper_time.txt");
    total_stepper_time<<total_t<<endl;
    total_stepper_time<<actual_t<<endl;
    total_stepper_time.close();
}




