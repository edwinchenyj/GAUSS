//
//  EigenFit.h
//  Gauss
//
//  Created by Edwin Chen on 2018-05-11.
//
//
//#define EDWIN_DEBUG

#ifndef EigenFit_h
#define EigenFit_h


#include <FEMIncludes.h>
#include <GaussIncludes.h>
#include <UtilitiesFEM.h>
#include <State.h>
#include <ParticleSystemIncludes.h>
#include <ConstraintFixedPoint.h>
#include <unsupported/Eigen/SparseExtra>
#include <sys/stat.h>
#include <igl/sortrows.h>
#include <igl/histc.h>
#include <igl/unique_rows.h>
#include <igl/hausdorff.h>
#include <fstream>
#include <SolverPardiso.h>
#include <SparseCholeskyPardiso.h>
#include <SparseGenRealShiftSolvePardiso.h>

#include <igl/boundary_facets.h>
#include <resultsUtilities.h>

using namespace Gauss;
using namespace FEM;
using namespace ParticleSystem; //For Force Spring
using std::cout;
using std::endl;

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



// subclass a hard-coded templated class from PhysicalSystemFEM
// this means that this EigenFit only works for NeohookeanHFixedTets
#ifdef NH
class EigenFit: public PhysicalSystemFEM<double, NeohookeanHFixedTet>
#endif

#ifdef COROT
class EigenFit: public PhysicalSystemFEM<double, FEMPSCoRotTet>
#endif

#ifdef ARAP
class EigenFit: public PhysicalSystemFEM<double, FEMPSARAPTet>
#endif

#ifdef LINEAR
class EigenFit: public PhysicalSystemFEM<double, LinearTet>
#endif

{
    //class EigenFit: public PhysicalSystemFEM<double, NeohookeanHFixedTet>{
    
public:
    // alias the hard-coded template name. Easier to read
    // the following lines read: the Physical System Implementation used here is a neo-hookean tet class
    //    using PhysicalSystemImpl = PhysicalSystemFEM<double, NeohookeanHFixedTet>;
#ifdef NH
    using PhysicalSystemImpl = PhysicalSystemFEM<double, NeohookeanHFixedTet>;
#endif
#ifdef COROT
    using PhysicalSystemImpl = PhysicalSystemFEM<double, FEMPSCoRotTet>;
#endif
#ifdef ARAP
    using PhysicalSystemImpl = PhysicalSystemFEM<double, FEMPSARAPTet>;
#endif
#ifdef LINEAR
    using PhysicalSystemImpl = PhysicalSystemFEM<double, LinearTet>;
#endif
    
    
    // use all the default function for now
    using PhysicalSystemImpl::getEnergy;
    using PhysicalSystemImpl::getMassMatrix;
    using PhysicalSystemImpl::getStiffnessMatrix;
    using PhysicalSystemImpl::getForce;
    using PhysicalSystemImpl::getInternalForce;
    using PhysicalSystemImpl::getQ;
    using PhysicalSystemImpl::getQDot;
    using PhysicalSystemImpl::getPosition;
    using PhysicalSystemImpl::getDPDQ;
    using PhysicalSystemImpl::getVelocity;
    using PhysicalSystemImpl::getDVDQ;
    using PhysicalSystemImpl::getGeometry;
    //    using PhysicalSystemImpl::getElements;
    
    // constructor
    // the constructor will take the two mesh parameters, one coarse one fine.
    // The coarse mesh data will be passed to the parent class constructor to constructor
    // the fine mesh data will be used to initialize the members specific to the EigenFit class
    EigenFit(Eigen::MatrixXx<double> &Vc, Eigen::MatrixXi &Fc,Eigen::MatrixXx<double> &Vf, Eigen::MatrixXi &Ff,  double youngs, double poisson, int constraint_dir, double constraint_tol, unsigned int const_profile, unsigned int hausdorff_dist, unsigned int num_modes, std::string cmeshname, std::string fmeshname, bool simple_mass_flag, double mode_matching_tol = 0, std::string hete_filename = "0", double hete_falloff_ratio = 1) : PhysicalSystemImpl(Vc,Fc)
    {
        
        this->simple_mass_flag = simple_mass_flag;
        coarse_mass_calculated = false;
        fine_mass_calculated = false;
        switching_constraint = false;
        this->mode_matching_tol = mode_matching_tol;
        calculate_matching_data_flag = 1;
        init_eigenvalue_criteria = false;
        init_eigenvalue_criteria_factor = 4;
        eigenfit_data = 0; // code for nothing happens
        
        step_number = 0;
        if(num_modes != 0)
        {
            m_Vf = Vf;
            m_Ff = Ff;
            
            m_Fc = Fc;
            igl::boundary_facets(Fc,m_surfFc);
            m_Vc_current = Vc;
            
            igl::boundary_facets(Ff,m_surfFf);
            m_Vf_current = m_Vf;
            
            m_cmeshname = cmeshname;
            m_fmeshname = fmeshname;
            
            m_constraint_dir = constraint_dir;
            m_constraint_tol = constraint_tol;
            
            
            this->const_profile = const_profile;
            
            //element[i] is a n-vector that stores the index of the element containing the ith vertex in the embedded mesh
            // *N is the upsample operator
            // (*N).transpose is downsample operator
            std::string N_file_name = "data/" + cmeshname + "_to_" + fmeshname+".mtx";
            if(!Eigen::loadMarket(*N,N_file_name))
            {
                cout<<N_file_name<<endl;
                cout<<"File does not exist, creating new file..."<<endl;
                getShapeFunctionMatrix(N,m_elements,Vf, (*this).getImpl());
                Eigen::saveMarket(*N,N_file_name);
            }
            
            // setup the fine mesh
            PhysicalSystemImpl *m_fineMeshSystem = new PhysicalSystemImpl(Vf,Ff);
            
            // set up material parameters
            this->youngs = youngs;
            this->poisson = poisson;
            
            
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
                
                if(stiffness_ratio.size() != Ff.rows())
                {
                    std::cerr << "Hete file need the same number of tets!\n";
                    exit(1);
                }
                
                //        // set material
                cout<<"Setting Youngs and Poisson for heterogeneous object..."<<endl;
                for(unsigned int iel=0; iel<Ff.rows(); ++iel)
                {
#ifdef NH
                    m_fineMeshSystem->getImpl().getElement(iel)->setParameters(low_stiffness + stiffness_ratio[iel] * (youngs - low_stiffness), poisson);
#endif
#ifdef COROT
                    m_fineMeshSystem->getImpl().getElement(iel)->setParameters(low_stiffness + stiffness_ratio[iel] * (youngs - low_stiffness), poisson);
#endif
#ifdef LINEAR
                    m_fineMeshSystem->getImpl().getElement(iel)->setParameters(low_stiffness + stiffness_ratio[iel] * (youngs - low_stiffness), poisson);
#endif
#ifdef ARAP
                    m_fineMeshSystem->getImpl().getElement(iel)->setParameters(low_stiffness + stiffness_ratio[iel] * (youngs - low_stiffness));
#endif
                    
                }
                
                
            }
            else
            {
                
                //        // set material
                cout<<"Setting Youngs and Poisson..."<<endl;
                for(unsigned int iel=0; iel<Ff.rows(); ++iel)
                {
#ifdef NH
                    m_fineMeshSystem->getImpl().getElement(iel)->setParameters(youngs, poisson);
#endif
#ifdef COROT
                    m_fineMeshSystem->getImpl().getElement(iel)->setParameters(youngs, poisson);
#endif
#ifdef LINEAR
                    m_fineMeshSystem->getImpl().getElement(iel)->setParameters(youngs, poisson);
#endif
#ifdef ARAP
                    m_fineMeshSystem->getImpl().getElement(iel)->setParameters(youngs);
#endif
                    
                }
            }
            
            
            m_fineWorld.addSystem(m_fineMeshSystem);
            
            
            
            //       constraints
            Eigen::SparseMatrix<double> fineP;
            Eigen::SparseMatrix<double> coarseP;
            Eigen::SparseMatrix<double> fineP2;
            Eigen::SparseMatrix<double> coarseP2;
            if (const_profile == 100)
            {
                cout<<"Setting constraint on the fine mesh and constructing fine mesh projection matrix"<<endl;
                
                std::string cconstraint_file_name = "data/" +cmeshname + "_const" + std::to_string(const_profile) + "_" +std::to_string(constraint_dir)+"_"+std::to_string(constraint_tol)+".mtx";
                std::string fconstraint_file_name = "data/" +fmeshname + "_const" + std::to_string(const_profile) + "_" +std::to_string(constraint_dir)+"_"+std::to_string(constraint_tol)+".mtx";
                Eigen::VectorXi fineMovingVerts;
                Eigen::VectorXi coarseMovingVerts;
                cout<<"Loading vertices and setting projection matrix..."<<endl;
                if(!Eigen::loadMarketVector(coarseMovingVerts,cconstraint_file_name))
                {
                    cout<<cconstraint_file_name<<endl;
                    cout<<"File does not exist for coarse mesh, creating new file..."<<endl;
                    coarseMovingVerts = minVertices(this, constraint_dir, constraint_tol);
                    Eigen::saveMarketVector(coarseMovingVerts,cconstraint_file_name);
                }
                if(!Eigen::loadMarketVector(fineMovingVerts,fconstraint_file_name))
                {
                    cout<<fconstraint_file_name<<endl;
                    cout<<"File does not exist for fine mesh, creating new file..."<<endl;
                    fineMovingVerts = minVertices(m_fineMeshSystem, constraint_dir, constraint_tol);
                    Eigen::saveMarketVector(fineMovingVerts,fconstraint_file_name);
                }
                
                
                m_fineMovingVerts = fineMovingVerts;
                
                std::vector<ConstraintFixedPoint<double> *> fineMovingConstraints;
                
                for(unsigned int ii=0; ii<fineMovingVerts.rows(); ++ii) {
                    fineMovingConstraints.push_back(new ConstraintFixedPoint<double>(&m_fineMeshSystem->getQ()[fineMovingVerts[ii]], Eigen::Vector3d(0,0,0)));
                    m_fineWorld.addConstraint(fineMovingConstraints[ii]);
                }
                m_fineWorld.finalize(); //After this all we're ready to go (clean up the interface a bit later)
                
                // hard-coded constraint projection
                fineP = fixedPointProjectionMatrix(fineMovingVerts, *m_fineMeshSystem,m_fineWorld);
                m_fineP = fineP;
                // only need to record one because only need to know if it's 0, 3, or 6. either fine or coarse is fine
                m_numConstraints = fineMovingVerts.size();
                
                m_coarseMovingVerts = coarseMovingVerts;
                
                coarseP = fixedPointProjectionMatrixCoarse(coarseMovingVerts);
                m_coarseP = fixedPointProjectionMatrixCoarse(coarseMovingVerts);
                
                Eigen::saveMarketDat(m_fineP, fconstraint_file_name+"_fineP.dat");
                Eigen::saveMarketDat(m_coarseP, cconstraint_file_name+"_cineP.dat");
                
                
                
                cout<<"Setting seoncd constraint on the fine mesh and constructing fine mesh projection matrix"<<endl;
                int constraint_dir2 = 0;
                double constraint_tol2 = 0.3;
                
                std::string cconstraint_file_name2 = "data/" +cmeshname + "_const2" + std::to_string(const_profile) + "_" +std::to_string(constraint_dir2)+"_"+std::to_string(constraint_tol2)+".mtx";
                std::string fconstraint_file_name2 = "data/" +fmeshname + "_const2" + std::to_string(const_profile) + "_" +std::to_string(constraint_dir2)+"_"+std::to_string(constraint_tol2)+".mtx";
                Eigen::VectorXi fineMovingVerts2;
                Eigen::VectorXi coarseMovingVerts2;
                Eigen::VectorXi fineMovingVerts_temp;
                Eigen::VectorXi coarseMovingVerts_temp;
                
                cout<<"Loading second vertices and setting projection matrix..."<<endl;
                if(!Eigen::loadMarketVector(coarseMovingVerts2,cconstraint_file_name2))
                {
                    cout<<cconstraint_file_name2<<endl;
                    cout<<"File does not exist for coarse mesh, creating new file..."<<endl;
                    coarseMovingVerts_temp = minVertices(this, constraint_dir2, constraint_tol2);
                    
                    if (coarseMovingVerts_temp.size() > coarseMovingVerts.size()) {
                        coarseMovingVerts2.resize(coarseMovingVerts_temp.size());
                    } else {
                        coarseMovingVerts_temp.resize(coarseMovingVerts.size());
                    }
                    auto it = std::set_intersection(coarseMovingVerts.data(), coarseMovingVerts.data()+coarseMovingVerts.size(), coarseMovingVerts_temp.data(), coarseMovingVerts_temp.data()+coarseMovingVerts_temp.size(), coarseMovingVerts2.data());
                    coarseMovingVerts2.conservativeResize(std::distance(coarseMovingVerts2.data(), it));
                    Eigen::saveMarketVector(coarseMovingVerts2,cconstraint_file_name2);
                }
                if(!Eigen::loadMarketVector(fineMovingVerts2,fconstraint_file_name2))
                {
                    cout<<fconstraint_file_name2<<endl;
                    cout<<"File does not exist for fine mesh, creating new file..."<<endl;
                    fineMovingVerts_temp = minVertices(m_fineMeshSystem, constraint_dir2, constraint_tol2);
                    
                    if (fineMovingVerts_temp.size() > fineMovingVerts.size()) {
                        fineMovingVerts2.resize(fineMovingVerts_temp.size());
                    } else {
                        fineMovingVerts_temp.resize(fineMovingVerts.size());
                    }
                    auto it = std::set_intersection(fineMovingVerts.data(), fineMovingVerts.data()+fineMovingVerts.size(), fineMovingVerts_temp.data(), fineMovingVerts_temp.data()+fineMovingVerts_temp.size(), fineMovingVerts2.data());
                    fineMovingVerts2.conservativeResize(std::distance(fineMovingVerts2.data(), it));
                    Eigen::saveMarketVector(fineMovingVerts2,fconstraint_file_name2);
                }
                
                
                m_fineMovingVerts2 = fineMovingVerts2;
                
                std::vector<ConstraintFixedPoint<double> *> fineMovingConstraints2;
                
                m_fineWorld.finalize(); //After this all we're ready to go (clean up the interface a bit later)
                
                // hard-coded constraint projection
                fineP2 = fixedPointProjectionMatrix(fineMovingVerts2, *m_fineMeshSystem,m_fineWorld);
                m_fineP2 = fineP2;
                // only need to record one because only need to know if it's 0, 3, or 6. either fine or coarse is fine
                m_numConstraints2 = fineMovingVerts2.size();
                
                m_coarseMovingVerts2 = coarseMovingVerts2;
                
                coarseP2 = fixedPointProjectionMatrixCoarse(coarseMovingVerts2);
                m_coarseP2 = fixedPointProjectionMatrixCoarse(coarseMovingVerts2);
                
                Eigen::saveMarketDat(m_fineP2, fconstraint_file_name2+"_fineP.dat");
                Eigen::saveMarketDat(m_coarseP2, cconstraint_file_name2+"_cineP.dat");
                
                
            }
            
            World<double, std::tuple<PhysicalSystemImpl *>,
            std::tuple<ForceSpringFEMParticle<double> *, ForceParticlesGravity<double> *>,
            std::tuple<ConstraintFixedPoint<double> *> > &world = m_fineWorld;
            
            auto q = mapStateEigen<0>(m_fineWorld);
            q.setZero();
            
            auto v = mapStateEigen<1>(m_fineWorld);
            v.setZero();
            
            // the first few ratios are 1 if less than 6 constraints, because eigenvalues ratio 0/0 is not defined
            if (m_numConstraints > 6) {
                // if constraint is more than a point constaint
                m_num_modes = num_modes;
                
                // put random value to m_R for now
                m_R.setConstant(m_num_modes, 1.0);
                ratio_calculated = false;
                m_I.setConstant(m_num_modes, 1.0);
            }
            else if (m_numConstraints == 3)
            {
                // if constraint is  a point constaint
                m_num_modes = num_modes;
                m_num_modes = m_num_modes + 3;
                
                // put random value to m_R for now
                m_R.setConstant(m_num_modes, 1.0);
                m_R(0) = 1.0;
                m_R(1) = 1.0;
                m_R(2) = 1.0;
                ratio_calculated = false;
                m_I.setConstant(m_num_modes, 1.0);
            }
            else
            {
                cout<<"No constraints so ignore the first 6 eigenvalues."<<endl;
                // otherwise, free boundary
                m_num_modes = num_modes;
                m_num_modes = m_num_modes + 6;
                
                // put random value to m_R for now
                m_R.setConstant(m_num_modes, 1.0);
                m_R(0) = 1.0;
                m_R(1) = 1.0;
                m_R(2) = 1.0;
                m_R(3) = 1.0;
                m_R(4) = 1.0;
                m_R(5) = 1.0;
                
                ratio_calculated = false;
                m_I.setConstant(m_num_modes, 1.0);
                
            }
            // assemble mass matrix in the constructor because it won't change
            AssemblerEigenSparseMatrix<double> &massMatrix = m_fineMassMatrix;
            
            //get mass matrix
            ASSEMBLEMATINIT(massMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
            ASSEMBLELIST(massMatrix, world.getSystemList(), getMassMatrix);
            ASSEMBLEEND(massMatrix);
            
            //constraint Projection
            (*massMatrix) = m_fineP*(*massMatrix)*m_fineP.transpose();
            
            if(simple_mass_flag)
            {
                Eigen::VectorXx<double> ones(m_fineP.rows());
                ones.setOnes();
                fine_mass_lumped.resize(m_fineP.rows());
                fine_mass_lumped_inv.resize(m_fineP.rows());
                fine_mass_lumped = ((*massMatrix)*ones);
                fine_mass_lumped_inv = fine_mass_lumped.cwiseInverse();
                m_fineM.resize(fine_mass_lumped.rows(),fine_mass_lumped.rows());
                m_fineM.setZero();
                typedef Eigen::Triplet<double> T;
                std::vector<T> tripletList;
                tripletList.reserve(fine_mass_lumped.rows());
                for(int i = 0; i < fine_mass_lumped.rows(); i++)
                {
                    tripletList.push_back(T(i,i,fine_mass_lumped(i)));
                }
                m_fineM.resize(fine_mass_lumped.rows(),fine_mass_lumped.rows());
                m_fineM.setFromTriplets(tripletList.begin(),tripletList.end());
                fine_mass_calculated = true;
            }
            
            fineMinvK.resize(fine_mass_lumped.rows(),fine_mass_lumped.rows());
            coarseMinvK.resize(m_coarseP.rows(),m_coarseP.rows());
            
            coarse_mass_lumped.resize(m_coarseP.rows());
            coarse_mass_lumped_inv.resize(m_coarseP.rows());
            m_coarseM.resize(coarse_mass_lumped.rows(),coarse_mass_lumped.rows());
            m_coarseM.setZero();
            // initialize mode matching list to match modes in order
            matched_modes_list.resize(m_num_modes);
            matched_modes_list.setZero();
            for (int i = 0; i < m_num_modes; i++) {
                matched_modes_list(i) = i;
            }
            
            init_matched_modes_list.resize(m_num_modes);
            init_matched_modes_list.setZero();
            for (int i = 0; i < m_num_modes; i++) {
                init_matched_modes_list(i) = i;
            }
            
            fine_pos0 = new double[world.getNumQDOFs()];
            Eigen::Map<Eigen::VectorXd> eigen_fine_pos0(fine_pos0,world.getNumQDOFs());
            
            Eigen::Vector3x<double> pos0;
            
            unsigned int idx;
            idx = 0;
            
            for(unsigned int vertexId=0;  vertexId < std::get<0>(m_fineWorld.getSystemList().getStorage())[0]->getGeometry().first.rows(); ++vertexId) {
                
                pos0 = m_Vf.row(vertexId);
                
                eigen_fine_pos0(idx) = pos0[0];
                idx++;
                eigen_fine_pos0(idx) = pos0[1];
                idx++;
                eigen_fine_pos0(idx) = pos0[2];
                idx++;
            }
            
        }
    }
    
    
    ~EigenFit() {delete fine_pos0;
    }
    
    void calculateFineMass(){
        World<double, std::tuple<PhysicalSystemImpl *>,
        std::tuple<ForceSpringFEMParticle<double> *, ForceParticlesGravity<double> *>,
        std::tuple<ConstraintFixedPoint<double> *> > &world = m_fineWorld;
        
        AssemblerEigenSparseMatrix<double> &massMatrix = m_fineMassMatrix;
        
        //get mass matrix
        ASSEMBLEMATINIT(massMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
        ASSEMBLELIST(massMatrix, world.getSystemList(), getMassMatrix);
        ASSEMBLEEND(massMatrix);
        
        //constraint Projection
        (*massMatrix) = m_fineP*(*massMatrix)*m_fineP.transpose();
        
        if(simple_mass_flag)
        {
            Eigen::VectorXx<double> ones(m_fineP.rows());
            ones.setOnes();
            fine_mass_lumped.resize(m_fineP.rows());
            fine_mass_lumped_inv.resize(m_fineP.rows());
            fine_mass_lumped = ((*massMatrix)*ones);
            fine_mass_lumped_inv = fine_mass_lumped.cwiseInverse();
            m_fineM.resize(fine_mass_lumped.rows(),fine_mass_lumped.rows());
            m_fineM.setZero();
            typedef Eigen::Triplet<double> T;
            std::vector<T> tripletList;
            tripletList.reserve(fine_mass_lumped.rows());
            for(int i = 0; i < fine_mass_lumped.rows(); i++)
            {
                tripletList.push_back(T(i,i,fine_mass_lumped(i)));
            }
            m_fineM.resize(fine_mass_lumped.rows(),fine_mass_lumped.rows());
            m_fineM.setFromTriplets(tripletList.begin(),tripletList.end());
            fine_mass_calculated = true;
        }
        
        
    }
    
    // calculate data, TODO: the first two parameter should be const
    template<typename MatrixAssembler>
    int calculateEigenFitData(const Eigen::VectorXx<double> &q, MatrixAssembler &coarseMassMatrix, MatrixAssembler &coarseStiffnessMatrix,  std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > &m_coarseUs, Eigen::MatrixXd &Y, Eigen::MatrixXd &Z)
    {
        if(simple_mass_flag)
        {
            Eigen::VectorXx<double> ones((*coarseMassMatrix).rows());
            coarse_mass_lumped.resize((*coarseMassMatrix).rows());
            coarse_mass_lumped.setZero();
            coarse_mass_lumped_inv.resize((*coarseMassMatrix).rows());
            coarse_mass_lumped_inv.setZero();
            ones.setOnes();
            coarse_mass_lumped = ((*coarseMassMatrix)*ones);
            coarse_mass_lumped_inv = coarse_mass_lumped.cwiseInverse();
            
            typedef Eigen::Triplet<double> T;
            std::vector<T> tripletList;
            tripletList.reserve(coarse_mass_lumped.rows());
            for(int i = 0; i < coarse_mass_lumped.rows(); i++)
            {
                tripletList.push_back(T(i,i,coarse_mass_lumped(i)));
            }
            m_coarseM.resize(coarse_mass_lumped.rows(),coarse_mass_lumped.rows());
            m_coarseM.setFromTriplets(tripletList.begin(),tripletList.end());
            coarse_mass_calculated = false;
        }
        
        if(simple_mass_flag)
        {
            
            coarseMinvK = (-1)*coarse_mass_lumped_inv.asDiagonal()*(*coarseStiffnessMatrix);
            
            Spectra::SparseGenRealShiftSolvePardiso<double> op(coarseMinvK);
            
            // Construct eigen solver object, requesting the smallest three eigenvalues
            Spectra::GenEigsRealShiftSolver<double, Spectra::LARGEST_MAGN, Spectra::SparseGenRealShiftSolvePardiso<double>> eigs(&op, m_num_modes, 5*m_num_modes,0.0);
            
            // Initialize and compute
            eigs.init();
            eigs.compute(1000,1e-10,Spectra::SMALLEST_MAGN);
            
            if(eigs.info() == Spectra::SUCCESSFUL)
            {
                m_coarseUs = std::make_pair(eigs.eigenvectors().real(), eigs.eigenvalues().real());
            }
            else{
                cout<<"eigen solve failed"<<endl;
                exit(1);
            }
            Eigen::VectorXd normalizing_const;
            normalizing_const.noalias() = (m_coarseUs.first.transpose() * coarse_mass_lumped.asDiagonal() * m_coarseUs.first).diagonal();
            normalizing_const = normalizing_const.cwiseSqrt().cwiseInverse();
            
            m_coarseUs.first = m_coarseUs.first * (normalizing_const.asDiagonal());
        }
        else
        {
            m_coarseUs = generalizedEigenvalueProblemNotNormalized((*coarseStiffnessMatrix), (*coarseMassMatrix), m_num_modes,0.0);
            Eigen::VectorXd normalizing_const;
            normalizing_const = (m_coarseUs.first.transpose() * (*coarseMassMatrix) * m_coarseUs.first).diagonal();
            normalizing_const = normalizing_const.cwiseSqrt().cwiseInverse();
            
            m_coarseUs.first = m_coarseUs.first * (normalizing_const.asDiagonal());
        }
        
        coarseEig = m_coarseUs;
        
        if(calculate_matching_data_flag != 0)
        {
            matched_modes_list.resize(m_num_modes);
            matched_modes_list.setZero();
            for (int i = 0; i < m_num_modes; i++) {
                matched_modes_list(i) = i;
            }
            if(step_number == 0 || step_number == constraint_switch+1)
            {
                init_coarse_eigenvectors = coarseEig.first;
                init_coarse_eigenvalues = coarseEig.second;
                
            }
            else
            {
                dist_map.resize(m_num_modes,m_num_modes);
                if(simple_mass_flag)
                {
                    dist_map.noalias() = init_coarse_eigenvectors.transpose() * (coarse_mass_lumped.asDiagonal() * coarseEig.first);
                }
                else
                {
                    dist_map.noalias() = init_coarse_eigenvectors.transpose() * (*coarseMassMatrix) * coarseEig.first;
                }
                dist_map = dist_map.cwiseAbs(); // flip any -1
                Eigen::MatrixXd ones(m_num_modes,m_num_modes);
                ones.setOnes();
                dist_map = ones - dist_map;
                dist_map = dist_map.cwiseAbs();
                
                
                for (int i = 0; i < m_num_modes; i++)
                {
                    Eigen::VectorXd::Index min_ind;
                    double min_val = dist_map.col(i).minCoeff(&min_ind);
                    if(abs(min_val) < mode_matching_tol) // set the matching tolerance
                    {
                        matched_modes_list(i) = min_ind;
                        
                        if(init_eigenvalue_criteria && coarseEig.second(i)/init_coarse_eigenvalues(min_ind) > init_eigenvalue_criteria_factor)
                        {
                            cout<<"warning: eigenvalue "<<i<<" changed too much from init eigenvalue "<<min_ind<<". To "<< coarseEig.second(i) <<" from "<< init_coarse_eigenvalues(min_ind)<<". Eigenfit will have diffculty at large nonlinearity."<<endl;
                            Eigen::saveMarketDat(dist_map, ("err_dist_map.dat"));
                            Eigen::saveMarketVectorDat(matched_modes_list, ("err_matched_modes_list.dat"));
                            
                            eigenfit_data = 3; // code when init eigenvalue criteria failed
                        }
                        
                    }
                    else
                    {
                        matched_modes_list(i) = -1; // -1 if can't find any matching mode
                        eigenfit_data = 2; // code for unmatching eigenmodes
                        int n1count = 0;
                        for (int list_i = 0; list_i < matched_modes_list.rows(); list_i++) {
                            if (matched_modes_list(list_i) == -1) {
                                n1count++;
                                
                            }
                        }
                        if(n1count > m_num_modes/2.0)
                        {
                            if( const_profile != 100)
                            {   // only check this if the constraint doesn't change
                                cout<<"Eigenmodes changed too much"<<endl;
                                eigenfit_data =4;
                            }
                        }
                        
                    }
                }
            }
        }
        
        if((!ratio_calculated))
        {
            World<double, std::tuple<PhysicalSystemImpl *>,
            std::tuple<ForceSpringFEMParticle<double> *, ForceParticlesGravity<double> *>,
            std::tuple<ConstraintFixedPoint<double> *> > &world = m_fineWorld;
            // should make sure this is performed at fine_q = 0;
            Eigen::Map<Eigen::VectorXd> fine_q = mapStateEigen<0>(m_fineWorld);
            fine_q.setZero();
            AssemblerEigenSparseMatrix<double> &fineStiffnessMatrix = m_fineStiffnessMatrix;
            
            //get stiffness matrix
            ASSEMBLEMATINIT(fineStiffnessMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
            ASSEMBLELIST(fineStiffnessMatrix, world.getSystemList(), getStiffnessMatrix);
            ASSEMBLELIST(fineStiffnessMatrix, world.getForceList(), getStiffnessMatrix);
            ASSEMBLEEND(fineStiffnessMatrix);
            
            //constraint Projection
            if(switching_constraint)
            {
                m_fineP = m_fineP2;
                m_coarseP = m_coarseP2;
                calculateFineMass();
            }
            
            (*fineStiffnessMatrix) = m_fineP*(*fineStiffnessMatrix)*m_fineP.transpose();
            
            
            if(simple_mass_flag)
            {
                fineMinvK = (-1)*fine_mass_lumped_inv.asDiagonal()*(*fineStiffnessMatrix);
                
                Spectra::SparseGenRealShiftSolvePardiso<double> op(fineMinvK);
                
                // Construct eigen solver object, requesting the smallest three eigenvalues
                Spectra::GenEigsRealShiftSolver<double, Spectra::LARGEST_MAGN, Spectra::SparseGenRealShiftSolvePardiso<double>> eigs(&op, m_num_modes, 5*m_num_modes,0.0);
                
                // Initialize and compute
                eigs.init();
                eigs.compute(1000,1e-10,Spectra::SMALLEST_MAGN);
                
                if(eigs.info() == Spectra::SUCCESSFUL)
                {
                    m_Us = std::make_pair(eigs.eigenvectors().real(), eigs.eigenvalues().real());
                }
                else{
                    cout<<"eigen solve failed"<<endl;
                    exit(1);
                }
                Eigen::VectorXd normalizing_const;
                normalizing_const.noalias() = (m_Us.first.transpose() * m_fineM * m_Us.first).diagonal();
                normalizing_const = normalizing_const.cwiseSqrt().cwiseInverse();
                
                m_Us.first = m_Us.first * (normalizing_const.asDiagonal());
            }
            else
            {
                m_Us = generalizedEigenvalueProblemNotNormalized(((*fineStiffnessMatrix)), (*m_fineMassMatrix), m_num_modes, 0.00);
                Eigen::VectorXd normalizing_const;
                normalizing_const = (m_Us.first.transpose() * (*m_fineMassMatrix) * m_Us.first).diagonal();
                normalizing_const = normalizing_const.cwiseSqrt().cwiseInverse();
                
                m_Us.first = m_Us.first * (normalizing_const.asDiagonal());
            }
            
            fineEig = m_Us;
            Eigen::saveMarketVectorDat(fineEig.second, "finemesh_rest_eigenvalues.dat");
            Eigen::saveMarketVectorDat(coarseEig.second, "coarsemesh_rest_eigenvalues.dat");
            if(calculate_matching_data_flag != 0)
            {
                init_matched_modes_list.resize(m_num_modes);
                init_matched_modes_list.setZero();
                for (int i = 0; i < m_num_modes; i++) {
                    init_matched_modes_list(i) = i;
                }
                init_coarse_eigenvectors = coarseEig.first;
                Eigen::MatrixXd  mapped_init_fine_eigenvectors;
                mapped_init_fine_eigenvectors = m_fineP*(*N) * m_coarseP.transpose()*init_coarse_eigenvectors;
                
                dist_map.resize(m_num_modes,m_num_modes);
                dist_map.setZero();
                if(simple_mass_flag)
                {
                    dist_map = (fineEig.first.transpose() * (m_fineM)) * mapped_init_fine_eigenvectors;
                }
                else
                {
                    dist_map = fineEig.first.transpose() * (*m_fineMassMatrix) * mapped_init_fine_eigenvectors;
                }
                dist_map = dist_map.cwiseAbs(); // flip any -1
                Eigen::MatrixXd ones(m_num_modes,m_num_modes);
                ones.setOnes();
                dist_map = ones - dist_map;
                dist_map = dist_map.cwiseAbs();
                for (int i = 0; i < m_num_modes; i++) {
                    Eigen::VectorXd::Index min_ind;
                    double min_val = dist_map.col(i).minCoeff(&min_ind);
                    if(abs(min_val) < init_mode_matching_tol) // set the matching tolerance criteria
                    {
                        init_matched_modes_list(i) = min_ind;
                        
                    }
                    else
                    {
                        init_matched_modes_list(i) = -1; // -1 if can't find any matching mode
                        eigenfit_data = 1;// code for missing initial mode matching (2 meshes too different)
                    }
                    
                }
                
            }
            
            Eigen::saveMarketVectorDat(init_matched_modes_list,"init_matched_modes_list.dat");
            Eigen::saveMarketVectorDat(dist_map,"init_dist_map.dat");
        Eigen::saveMarketVectorDat(fineEig.second,"init_fineeigenvalues.dat");
            
            
            cout<<"Calculating init ratio\n";
            m_R.setOnes();
            
            for(int i = 0; i < m_num_modes; ++i)
            {
                if(calculate_matching_data_flag == 2)
                {
                    if(init_matched_modes_list(i) != -1)
                    {
                        m_R(i) = m_Us.second(init_matched_modes_list(i))/m_coarseUs.second(i);
                    }
                    else{
                        m_R(i) = 1;
                        
                    }
                }
                else
                {
                    m_R(i) = m_Us.second(i)/m_coarseUs.second(i);
                }
                
                if (m_numConstraints == 3)
                {
                    // if constraint is  a point constaint
                    m_R(0) = 1.0;
                    m_R(1) = 1.0;
                    m_R(2) = 1.0;
                }
                else if(m_numConstraints == 0)
                {
                    //  free boundary
                    cout<<"Free boundary, setting first 6 ratios to 1."<<endl;
                    m_R(0) = 1.0;
                    m_R(1) = 1.0;
                    m_R(2) = 1.0;
                    m_R(3) = 1.0;
                    m_R(4) = 1.0;
                    m_R(5) = 1.0;
                    
                }
                
            }
            ratio_calculated = true;
        }
        
        m_R_current = m_R;
        if(calculate_matching_data_flag == 2  )
        {
            m_R_current.setOnes();
            for (int i = 0; i < m_num_modes; i++) {
                if(matched_modes_list(i) != -1)
                {
                    m_R_current(i) = m_R(matched_modes_list(i));
                }
                else
                {
                    m_R_current(i) = 1; // if no matched mode, use ratio = 1;
                }
            }
        }
        if(simple_mass_flag)
        {
            Y = (m_coarseM)*m_coarseUs.first*((m_R_current-m_I).asDiagonal());
            Z =  (m_coarseUs.second.asDiagonal()*m_coarseUs.first.transpose())*(m_coarseM);
        }
        else
        {
            Y = (*coarseMassMatrix)*m_coarseUs.first*(m_R_current-m_I).asDiagonal();
            Z =  (m_coarseUs.second.asDiagonal()*m_coarseUs.first.transpose()*(*coarseMassMatrix));
        }
        // no error
        return 0;
    }
    
    
    Eigen::SparseMatrix<double> fixedPointProjectionMatrixCoarse(Eigen::VectorXi &indices) {
        
        std::vector<Eigen::Triplet<double> > triplets;
        Eigen::SparseMatrix<double> P;
        Eigen::VectorXi sortedIndices = indices;
        std::sort(sortedIndices.data(), sortedIndices.data()+indices.rows());
        
        int fIndex = 0;
        
        //total number of DOFS in system
        
        unsigned int n = 3*this->getImpl().getV().rows();
        unsigned int m = n - 3*indices.rows();
        
        P.resize(m,n);
        
        //number of unconstrained DOFs
        unsigned int rowIndex =0;
        for(unsigned int vIndex = 0; vIndex < this->getImpl().getV().rows(); vIndex++) {
            
            while((vIndex < this->getImpl().getV().rows()) && (fIndex < sortedIndices.rows()) &&(vIndex == sortedIndices[fIndex])) {
                fIndex++;
                vIndex++;
            }
            
            if(vIndex == this->getImpl().getV().rows())
                break;
            
            //add triplet into matrix
            triplets.push_back(Eigen::Triplet<double>(this->getQ().getGlobalId() +rowIndex, this->getQ().getGlobalId() + 3*vIndex,1));
            triplets.push_back(Eigen::Triplet<double>(this->getQ().getGlobalId() +rowIndex+1, this->getQ().getGlobalId() + 3*vIndex+1, 1));
            triplets.push_back(Eigen::Triplet<double>(this->getQ().getGlobalId() +rowIndex+2, this->getQ().getGlobalId() + 3*vIndex+2, 1));
            
            rowIndex+=3;
        }
        
        P.setFromTriplets(triplets.begin(), triplets.end());
        
        //build the matrix and  return
        return P;
    }
    
    
    inline World<double, std::tuple<PhysicalSystemImpl *>,
    std::tuple<ForceSpringFEMParticle<double> *, ForceParticlesGravity<double> *>,
    std::tuple<ConstraintFixedPoint<double> *> > & getFineWorld(){ return m_fineWorld;}
    
    
    std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > coarseEig;
    std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > fineEig;
    std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > m_Us;
    
    Eigen::VectorXx<double> coarseEigenvalues;
    Eigen::VectorXx<double> fineEigenvalues;
    Eigen::MatrixXx<double> init_coarse_eigenvectors;
    Eigen::VectorXx<double> init_coarse_eigenvalues;
    
    
    
    Eigen::SparseMatrix<double> m_fineP;
    Eigen::SparseMatrix<double> m_coarseP;
    Eigen::SparseMatrix<double> m_fineP2;
    Eigen::SparseMatrix<double> m_coarseP2;

    //        Eigen::MatrixXd coarse_V_disp_p;
    
    AssemblerEigenSparseMatrix<double> m_coarseMassMatrix;
    AssemblerEigenSparseMatrix<double> m_fineMassMatrix;
    
    AssemblerEigenVector<double> m_fineforceVector;
    AssemblerEigenVector<double> m_finefExt;
    
    AssemblerEigenVector<double> m_forceVector;
    
    AssemblerEigenSparseMatrix<double> N;
    
    // rest state of fine q
    double* fine_pos0  = NULL;
    
    
    int flag = 0;
    
    double a;
    double b;
    
    int step_number;
    bool ratio_calculated;
    
    std::string m_fmeshname, m_cmeshname;
    
    Eigen::MatrixXd Vf_reset;
    Eigen::MatrixXd V_reset;
    Eigen::MatrixXd m_Vf_current;
    Eigen::MatrixXd m_Vc_current;
    Eigen::MatrixXd m_Vc;
    Eigen::MatrixXi m_surfFf;
    Eigen::MatrixXi m_surfFc;
    Eigen::VectorXd m_R;
    Eigen::VectorXd m_R_current;
    Eigen::MatrixXx<double> m_Vf;
    
    bool simple_mass_flag;
    bool coarse_mass_calculated;
    bool fine_mass_calculated;
    Eigen::VectorXx<double> coarse_mass_lumped;
    Eigen::VectorXx<double> coarse_mass_lumped_inv;
    Eigen::VectorXx<double> fine_mass_lumped;
    
    Eigen::VectorXx<double> fine_mass_lumped_inv;
    Eigen::SparseMatrix<double,Eigen::RowMajor> m_coarseM;
    Eigen::SparseMatrix<double,Eigen::RowMajor> m_fineM;
    
    Eigen::SparseMatrix<double,Eigen::RowMajor> coarseMinvK;
    Eigen::SparseMatrix<double,Eigen::RowMajor> fineMinvK;
    
    Eigen::VectorXi matched_modes_list;
    Eigen::VectorXi init_matched_modes_list;
    
    double mode_matching_tol;
    Eigen::MatrixXd dist_map;
    
    int calculate_matching_data_flag;
    
    bool init_eigenvalue_criteria;
    int init_eigenvalue_criteria_factor;
    double init_mode_matching_tol;
    
    int eigenfit_data;
    
    std::string hete_filename;
    double hete_falloff_ratio;
    
    bool switching_constraint;
    int constraint_switch;
protected:
    
    World<double, std::tuple<PhysicalSystemImpl *>,
    std::tuple<ForceSpringFEMParticle<double> *, ForceParticlesGravity<double> *>,
    std::tuple<ConstraintFixedPoint<double> *> > m_fineWorld;
    
    World<double, std::tuple<PhysicalSystemImpl *>,
    std::tuple<ForceSpringFEMParticle<double> *, ForceParticlesGravity<double> *>,
    std::tuple<ConstraintFixedPoint<double> *> > m_coarseWorld;
    
    
    AssemblerEigenSparseMatrix<double> m_fineStiffnessMatrix;
    
    AssemblerEigenSparseMatrix<double> m_coarseStiffnessMatrix;
    
    Eigen::VectorXi m_fineMovingVerts;
    Eigen::VectorXi m_fineFixedVerts;
    
    Eigen::VectorXi m_coarseMovingVerts;
    
    Eigen::VectorXi m_fineMovingVerts2;
    
    Eigen::VectorXi m_coarseMovingVerts2;
    

    
    double youngs;
    double poisson;
    
    int m_constraint_dir;
    double m_constraint_tol;
    
    //num modes to correct
    unsigned int m_num_modes;
    
    Eigen::VectorXd m_I;
    
    Eigen::MatrixXi m_Ff;
    Eigen::MatrixXi m_Fc;
    Eigen::MatrixXd m_N;
    //m_elements[i] is a n-vector that stores the index of the element containing the ith vertex in the embedded mesh
    Eigen::VectorXi m_elements;
    
    
    unsigned int const_profile;
    unsigned int m_numConstraints;
    unsigned int m_numConstraints2;
private:
    
};

#endif /* EigenFit_h */

