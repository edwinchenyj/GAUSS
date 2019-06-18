

#include <functional>

#include <Qt3DIncludes.h>
#include <GaussIncludes.h>
#include <FEMIncludes.h>

//Any extra things I need such as constraints
#include <ConstraintFixedPoint.h>
#include <TimeStepperEigenFitSI.h>
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

typedef TimeStepperEigenFitSI<double, AssemblerParallel<double, AssemblerEigenSparseMatrix<double>>, AssemblerParallel<double, AssemblerEigenVector<double>> > MyTimeStepper;

typedef Scene<MyWorld, MyTimeStepper> MyScene;

// used for preStepCallback. should be delete
std::vector<ConstraintFixedPoint<double> *> movingConstraints;
Eigen::VectorXi movingVerts;
Eigen::MatrixXd V, Vtemp;
Eigen::MatrixXi F;
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
    
    std::string cmeshname = "/meshesTetWild/arma/arma_3";
    std::string fmeshname = "/meshesTetWild/arma/arma_0";
    
    //    parameters
    double youngs = 1e6;
    double poisson = 0.45;
    double const_tol = 1e-2;
    int const_profile = 100;
    std::string initial_def = "0";
    int num_steps = 10;
    bool haus = false;
    int num_modes = 10;
    int const_dir = 1; // constraint direction. 0 for x, 1 for y, 2 for z
    double step_size = 1e-2;
    double a = 0;
    double b = -1e-2;
    bool output_data_flag = false;
    bool simple_mass_flag = true;
    double mode_matching_tol = 0.4;
    int calculate_matching_data_flag = 1;
    double init_mode_matching_tol = 0.4;
    bool init_eigenvalue_criteria= false;
    int init_eigenvalue_criteria_factor = 100;
    bool eigenfit_damping = false;
    
    std::string integrator = "SI";
    std::string hete_filename = "0";
    double hete_falloff_ratio = 1.0;
    double motion_multiplier = 1.0;
    int constraint_switch = 30;
    
    parse_input(argc, argv, cmeshname, fmeshname, youngs, const_tol, const_profile, initial_def, num_steps, haus, num_modes, const_dir, step_size, a, b, output_data_flag, simple_mass_flag, mode_matching_tol, calculate_matching_data_flag, init_mode_matching_tol, init_eigenvalue_criteria, init_eigenvalue_criteria_factor, integrator, eigenfit_damping, hete_filename, hete_falloff_ratio, motion_multiplier,constraint_switch);
    
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
    
    
    EigenFit *test = new EigenFit(V,F,Vf,Ff,youngs,poisson,const_dir,const_tol, const_profile,haus,num_modes,cmeshnameActual,fmeshnameActual,simple_mass_flag,mode_matching_tol,hete_filename,hete_falloff_ratio);

    test->a = a;
    test->b = b;
    test->calculate_matching_data_flag = calculate_matching_data_flag;
    test->init_mode_matching_tol = init_mode_matching_tol;
    test->init_eigenvalue_criteria = init_eigenvalue_criteria;
    test->init_eigenvalue_criteria_factor = init_eigenvalue_criteria_factor;
    test->constraint_switch = constraint_switch;
    
    world.addSystem(test);
    
    // projection matrix for constraints
    Eigen::SparseMatrix<double> P;
    // projection matrix for second constraints
    Eigen::SparseMatrix<double> P2;

    
    // constraint switch
    if (const_profile == 100)
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
        
        // second set of constraints
        Eigen::VectorXi indices2;
        Eigen::VectorXi indices_temp;
        int const_dir2 = 0;
        double const_tol2 = 0.3;
        std::string constraint_file_name2 = "data/" + cmeshnameActual + "_const2" + std::to_string(const_profile) + "_" +std::to_string(const_dir2)+"_"+std::to_string(const_tol2)+".mtx";
        cout<<"Setting second moving constraints and constrainting projection matrix"<<endl;
        cout<<"Loading second moving vertices and setting projection matrix..."<<endl;
        if(!Eigen::loadMarketVector(indices2,constraint_file_name2))
        {
            cout<<"File does not exist, creating new file..."<<endl;
            indices_temp = minVertices(test, const_dir2,const_tol2);
            if (indices_temp.size() > indices.size()) {
                indices2.resize(indices_temp.size());
            } else {
                indices2.resize(indices.size());
            }
            auto it = std::set_intersection(indices.data(), indices.data()+indices.size(), indices_temp.data(), indices_temp.data()+indices_temp.size(), indices2.data());
            indices2.conservativeResize(std::distance(indices2.data(), it));
            Eigen::saveMarketVector(indices2,constraint_file_name2);
        }
        
        
        P = fixedPointProjectionMatrix(indices, *test,world);
        P2 = fixedPointProjectionMatrix(indices2, *test,world);
                    movingVerts = minVertices(test, const_dir, const_tol);//indices for moving parts
        
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
        
        test->calculateEigenFitData(q,massMatrix,stiffnessMatrix,m_coarseUs,Y,Z);
        
        
        t = clock() -t;
        std::ofstream pre_calc_time_file;
        pre_calc_time_file.open ("pre_calc_time.txt");
        pre_calc_time_file<<t;
        pre_calc_time_file.close();
        
    
    
    
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
    
    MyTimeStepper stepper(step_size,P,P2,constraint_switch,num_modes,a,b, integrator);
    stepper.getImpl().eigenfit_damping = eigenfit_damping;
    // rayleigh damping. should not be here but this will do for now
    //         the number of steps to take
    
    unsigned int file_ind = 0;
    
    clock_t dt;
    clock_t total_t = 0;
    double actual_t = 0.0;
    for(istep=0; istep<num_steps ; ++istep)
    {
        if (istep == constraint_switch) {
            test->ratio_calculated = false;
            test->switching_constraint = true;
            
            P=P2;
            if( num_modes != 0)
            {
                t = clock();
                auto q = mapStateEigen<0>(world);
                Eigen::VectorXd temp_q = q;
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
                
                test->calculateEigenFitData(q,massMatrix,stiffnessMatrix,m_coarseUs,Y,Z);
                
                
                q = temp_q;
                t = clock() -t;
                std::ofstream pre_calc_time_file;
                pre_calc_time_file.open ("pre_calc_time2.txt");
                pre_calc_time_file<<t;
                pre_calc_time_file.close();
                
                
            }
        }
        
        // update step number;
        test->step_number++;
        cout<<"simulating frame #" << test->step_number<<endl;
        t = clock();
        stepper.step(world);
        dt = clock() - t;
        total_t += dt;
        actual_t = ((double)total_t)/CLOCKS_PER_SEC;
        
        
        apply_moving_constraint(const_profile, world.getState(), movingConstraints, istep, motion_multiplier);
        
        // rest pos for the coarse mesh getGeometry().first is V
        Eigen::VectorXd q = mapStateEigen(world);
        
        Eigen::MatrixXd V_disp = std::get<0>(world.getSystemList().getStorage())[0]->getGeometry().first;
        // output mesh position with only surface mesh
        
        q_state_to_position(q,V_disp);
        
        // output mesh position with only surface mesh
        igl::writeOBJ(filename_number_padded("surfpos", file_ind,"obj"),V_disp,surfF);

        file_ind++;
        
    }
//    output the total time spent in the stepper
    std::ofstream total_stepper_time;
    total_stepper_time.open ("total_stepper_time.txt");
    total_stepper_time<<total_t<<endl;
    total_stepper_time<<actual_t<<endl;
    total_stepper_time.close();
}




