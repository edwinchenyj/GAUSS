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
    //    using PhysicalSystemImpl::getStrainEnergy;
    //    using PhysicalSystemImpl::getStrainEnergyPerElement;
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
    EigenFit(Eigen::MatrixXx<double> &Vc, Eigen::MatrixXi &Fc,Eigen::MatrixXx<double> &Vf, Eigen::MatrixXi &Ff, int dynamic_flag, double youngs, double poisson, int constraint_dir, double constraint_tol, unsigned int const_profile, unsigned int hausdorff_dist, unsigned int num_modes, std::string cmeshname, std::string fmeshname, bool simple_mass_flag, double mode_matching_tol = 0) : PhysicalSystemImpl(Vc,Fc)
    {
        this->simple_mass_flag = simple_mass_flag;
        coarse_mass_calculated = false;
        fine_mass_calculated = false;
        
        this->mode_matching_tol = mode_matching_tol;
        calculate_matching_data_flag = 1;
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
            
            m_feval_manual.resize(num_modes);
            m_feval_manual.setZero();
            
            cout<<"Fine mesh size"<<endl;
            std::cout<<m_Vf.rows()<<std::endl;
            std::cout<<m_Vf.cols()<<std::endl;
            
            ratio_recalculation_switch = dynamic_flag;
            this->const_profile = const_profile;
            
            //element[i] is a n-vector that stores the index of the element containing the ith vertex in the embedded mesh
            // *N is the upsample operator
            // (*N).transpose is downsample operator
            getShapeFunctionMatrix(N,m_elements,Vf, (*this).getImpl());
            
            // set the flag
            haus = hausdorff_dist;
            
            // setup the fine mesh
            PhysicalSystemImpl *m_fineMeshSystem = new PhysicalSystemImpl(Vf,Ff);
            
            // set up material parameters
            this->youngs = youngs;
            this->poisson = poisson;
            for(unsigned int iel=0; iel<m_fineMeshSystem->getImpl().getF().rows(); ++iel) {
                
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
            m_fineWorld.addSystem(m_fineMeshSystem);
            
            //       constraints
            Eigen::SparseMatrix<double> fineP;
            Eigen::SparseMatrix<double> coarseP;
            
            cout<<"Setting fine mesh constraints..."<<endl;
            if (const_profile == 0) {
                // hard-coded constraint projection
                cout<<"No constraints"<<endl;
                m_fineWorld.finalize();
                
                fineP.resize(Vf.rows()*3,Vf.rows()*3);
                fineP.setIdentity();
                m_fineP = fineP;
                coarseP.resize(Vc.rows()*3,Vc.rows()*3);
                coarseP.setIdentity();
                m_coarseP = coarseP;
                //                Eigen::saveMarketDat(m_fineP, fconstraint_file_name+"_fineP.dat");
                //                Eigen::saveMarketDat(m_coarseP, cconstraint_file_name+"_cineP.dat");
                
                
                m_numConstraints = 0;
            }
            else if (const_profile == 1)
            {
                cout<<"Setting constraint on the fine mesh and constructing fine mesh projection matrix"<<endl;
                // default constraint
                //            fix displacement
                fixDisplacementMin(m_fineWorld, m_fineMeshSystem, constraint_dir, constraint_tol);
                
                m_fineWorld.finalize();
                // hard-coded constraint projection
                
                std::string cconstraint_file_name = "data/" + cmeshname + "_const" + std::to_string(const_profile) + "_" +std::to_string(constraint_dir)+"_"+std::to_string(constraint_tol)+".mtx";
                std::string fconstraint_file_name = "data/" + fmeshname + "_const" + std::to_string(const_profile) + "_" +std::to_string(constraint_dir)+"_"+std::to_string(constraint_tol)+".mtx";
                Eigen::VectorXi fineFixedVerts;
                Eigen::VectorXi coarseFixedVerts;
                cout<<"Loading vertices and setting projection matrix..."<<endl;
                if(!Eigen::loadMarketVector(coarseFixedVerts,cconstraint_file_name))
                {
                    cout<<cconstraint_file_name<<endl;
                    cout<<"File does not exist for coarse mesh, creating new file..."<<endl;
                    coarseFixedVerts = minVertices(this, constraint_dir, constraint_tol);
                    Eigen::saveMarketVector(coarseFixedVerts,cconstraint_file_name);
                }
                if(!Eigen::loadMarketVector(fineFixedVerts,fconstraint_file_name))
                {
                    cout<<fconstraint_file_name<<endl;
                    cout<<"File does not exist for fine mesh, creating new file..."<<endl;
                    fineFixedVerts = minVertices(m_fineMeshSystem, constraint_dir, constraint_tol);
                    Eigen::saveMarketVector(fineFixedVerts,fconstraint_file_name);
                }
                
                
                m_fineFixedVerts = fineFixedVerts;
                fineP = fixedPointProjectionMatrix(fineFixedVerts, *m_fineMeshSystem,m_fineWorld);
                m_fineP = fineP;
                
                coarseP = fixedPointProjectionMatrixCoarse(coarseFixedVerts);
                m_coarseP = fixedPointProjectionMatrixCoarse(coarseFixedVerts);
                Eigen::saveMarketDat(m_fineP, fconstraint_file_name+"_fineP.dat");
                Eigen::saveMarketDat(m_coarseP, cconstraint_file_name+"_cineP.dat");
                
                // only need to record one because only need to know if it's 0, 3, or 6. either fine or coarse would work
                m_numConstraints = fineFixedVerts.size();
                
            }
            else if (const_profile == 2)
            {
                
                
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
                fineP = fixedPointProjectionMatrix(fineMovingVerts, *m_fineMeshSystem,m_fineWorld);
                m_fineP = fineP;
                
                
                coarseP = fixedPointProjectionMatrixCoarse(coarseMovingVerts);
                m_coarseP = fixedPointProjectionMatrixCoarse(coarseMovingVerts);
                
                Eigen::saveMarketDat(m_fineP, fconstraint_file_name+"_fineP.dat");
                Eigen::saveMarketDat(m_coarseP, cconstraint_file_name+"_cineP.dat");
                
                std::vector<ConstraintFixedPoint<double> *> fineMovingConstraints;
                
                for(unsigned int ii=0; ii<fineMovingVerts.rows(); ++ii) {
                    fineMovingConstraints.push_back(new ConstraintFixedPoint<double>(&m_fineMeshSystem->getQ()[fineMovingVerts[ii]], Eigen::Vector3d(0,0,0)));
                    m_fineWorld.addConstraint(fineMovingConstraints[ii]);
                }
                m_fineWorld.finalize(); //After this all we're ready to go (clean up the interface a bit later)
                
                // hard-coded constraint projection
                //
                // only need to record one because only need to know if it's 0, 3, or 6. either fine or coarse is fine
                m_numConstraints = fineMovingVerts.size();
                
            }
            else if (const_profile == 3)
            {
                std::string cconstraint_file_name = "data/" +cmeshname + "_const" + std::to_string(const_profile) + "_" +std::to_string(constraint_dir)+"_"+std::to_string(constraint_tol)+".mtx";
                std::string fconstraint_file_name = "data/" +fmeshname + "_const" + std::to_string(const_profile) + "_" +std::to_string(constraint_dir)+"_"+std::to_string(constraint_tol)+".mtx";
                Eigen::VectorXi fineFixedVerts;
                Eigen::VectorXi coarseFixedVerts;
                cout<<"Loading vertices and setting projection matrix..."<<endl;
                if(!Eigen::loadMarketVector(coarseFixedVerts,cconstraint_file_name))
                {
                    cout<<cconstraint_file_name<<endl;
                    cout<<"File does not exist for coarse mesh, creating new file..."<<endl;
                    coarseFixedVerts = minVertices(this, constraint_dir, constraint_tol);
                    Eigen::saveMarketVector(coarseFixedVerts,cconstraint_file_name);
                }
                if(!Eigen::loadMarketVector(fineFixedVerts,fconstraint_file_name))
                {
                    cout<<fconstraint_file_name<<endl;
                    cout<<"File does not exist for fine mesh, creating new file..."<<endl;
                    fineFixedVerts = minVertices(m_fineMeshSystem, constraint_dir, constraint_tol);
                    Eigen::saveMarketVector(fineFixedVerts,fconstraint_file_name);
                }
                
                std::vector<ConstraintFixedPoint<double> *> fixedConstraints;
                //
                for(unsigned int ii=0; ii<fineFixedVerts.rows(); ++ii) {
                    fixedConstraints.push_back(new ConstraintFixedPoint<double>(&m_fineMeshSystem->getQ()[fineFixedVerts[ii]], Eigen::Vector3d(0,0,0)));
                    m_fineWorld.addConstraint(fixedConstraints[ii]);
                }
                
                m_fineWorld.finalize(); //After this all we're ready to go (clean up the interface a bit later)
                
                fineP = fixedPointProjectionMatrix(fineFixedVerts, *m_fineMeshSystem,m_fineWorld);
                m_fineP = fineP;
                
                
                coarseP = fixedPointProjectionMatrixCoarse(coarseFixedVerts);
                m_coarseP = fixedPointProjectionMatrixCoarse(coarseFixedVerts);
                
                Eigen::saveMarketDat(m_fineP, fconstraint_file_name+"_fineP.dat");
                Eigen::saveMarketDat(m_coarseP, cconstraint_file_name+"_cineP.dat");
                
                
                m_numConstraints = fineFixedVerts.rows();
                
                
            }
            else if (const_profile == 4 || const_profile == 5 || const_profile == 6 || const_profile == 7 || const_profile == 8)
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
                
                
            }
            
            
            //lambda can't capture member variable, so create a local one for lambda in ASSEMBLELIST
            // world name must match "world"?!
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
            
            //lambda can't capture member variable, so create a local one for lambda in ASSEMBLELIST
            cout<<"Pre-calculate mass matrix for the fine mesh."<<endl;
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
            // fill in the rest state position
            restFineState = m_fineWorld.getState();
            
            
            // create a deep copy for the rest state position
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
    
    
    ~EigenFit() {delete fine_pos0; delete coarse_pos0; delete fine_q_transfered;
    }
    
    // calculate data, TODO: the first two parameter should be const
    template<typename MatrixAssembler>
    bool calculateEigenFitData(const Eigen::VectorXx<double> &q, MatrixAssembler &coarseMassMatrix, MatrixAssembler &coarseStiffnessMatrix,  std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > &m_coarseUs, Eigen::MatrixXd &Y, Eigen::MatrixXd &Z)
    {
        
        //        Eigen::saveMarketDat((*coarseStiffnessMatrix), "coarseStiffness.dat");
        //        Eigen::saveMarketDat((*coarseMassMatrix), "coarseMass.dat");
        if(simple_mass_flag)
        {
            //            cout<<"using simple mass"<<endl;
            
            
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
            coarse_mass_calculated = true;
        }
        
        //        cout<<"calculate eigenfit data"<<endl;
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
                //                cout<<"spectra successful"<<endl;
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
            //            cout<<"use consistent mass"<<endl;
            m_coarseUs = generalizedEigenvalueProblemNotNormalized((*coarseStiffnessMatrix), (*coarseMassMatrix), m_num_modes,0.0);
            Eigen::VectorXd normalizing_const;
            normalizing_const = (m_coarseUs.first.transpose() * (*coarseMassMatrix) * m_coarseUs.first).diagonal();
            normalizing_const = normalizing_const.cwiseSqrt().cwiseInverse();
            
            m_coarseUs.first = m_coarseUs.first * (normalizing_const.asDiagonal());
        }
        
        //        coarseEigMassProj = m_coarseUs;
        //        prev_coarse_eigenvectors = coarseEig.first;
        coarseEig = m_coarseUs;

        if(calculate_matching_data_flag != 0)
        {
            //            cout<<"matching modes. ";
            matched_modes_list.resize(m_num_modes);
            matched_modes_list.setZero();
            for (int i = 0; i < m_num_modes; i++) {
                matched_modes_list(i) = i;
            }
            if(step_number == 0)
            {
                cout<<"storing initial eigenvectors"<<endl;
                init_coarse_eigenvectors = coarseEig.first;
                init_coarse_eigenvalues = coarseEig.second;
                
            }
            else
            {
                //            calculate normed distance to initial eigenvectors
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
                dist_map = dist_map.cwiseAbs(); // should have no effect
                
                cout<<"dist map: "<<endl<<dist_map<<endl;
                
                for (int i = 0; i < m_num_modes; i++)
                {
                    Eigen::VectorXd::Index min_ind;
                    double min_val = dist_map.col(i).minCoeff(&min_ind);
                    if(abs(min_val) < mode_matching_tol) // set the matching tolerance
                    {
                        matched_modes_list(i) = min_ind;
                        
                        cout<<"init matched: "<<init_matched_modes_list<<endl;
                        if(i == min_ind)
                        {
                            cout<<"no mode crossing. matching % = "<< 1-min_val<<endl;
                            if(init_matched_modes_list(min_ind) != -1)
                            {
                            cout<<"will apply ratio: "<<m_R(min_ind)<<endl;
                            cout<<"or: "<<fineEig.second(init_matched_modes_list(min_ind))<<"/"<<init_coarse_eigenvalues(min_ind)<<endl;
                            }
                            else
                            {
                                cout<<"initial mode "<<min_ind<<" matches to nothing."<<endl;
                            }
                            
                        }
                        else
                        {                            //                            cout<<"mode crossing occurred"<<endl;
                            cout<<"mode "<<i<<" matched to initial mode "<<min_ind<<", matching % = "<<1-min_val<<endl;
                            if(init_matched_modes_list(min_ind) != -1)
                            {
                            cout<<"need to apply ratio: "<<m_R(min_ind)<<endl;
                            cout<<"or: "<<fineEig.second(init_matched_modes_list(min_ind))<<"/"<<init_coarse_eigenvalues(min_ind)<<endl;
                            }
                            else
                            {
                                cout<<"initial mode "<<min_ind<<" matches to nothing."<<endl;
                            }
                        }
                    }
                    else
                    {
                        matched_modes_list(i) = -1; // -1 if can't find any matching mode
                        cout<<"no matching mode"<<endl;
                    }
                }
            }
        }
        
        //        std::cout<<"Dynamic switch: "<<ratio_recalculation_switch<<std::endl;
        if((!ratio_calculated))
        {
            World<double, std::tuple<PhysicalSystemImpl *>,
            std::tuple<ForceSpringFEMParticle<double> *, ForceParticlesGravity<double> *>,
            std::tuple<ConstraintFixedPoint<double> *> > &world = m_fineWorld;
            // should make sure this is performed at fine_q = 0;
            Eigen::Map<Eigen::VectorXd> fine_q = mapStateEigen<0>(m_fineWorld);
            fine_q.setZero();
            fine_q = (*N) * q;
            AssemblerEigenSparseMatrix<double> &fineStiffnessMatrix = m_fineStiffnessMatrix;
            
            //get stiffness matrix
            ASSEMBLEMATINIT(fineStiffnessMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
            ASSEMBLELIST(fineStiffnessMatrix, world.getSystemList(), getStiffnessMatrix);
            ASSEMBLELIST(fineStiffnessMatrix, world.getForceList(), getStiffnessMatrix);
            ASSEMBLEEND(fineStiffnessMatrix);
            
            //constraint Projection
            (*fineStiffnessMatrix) = m_fineP*(*fineStiffnessMatrix)*m_fineP.transpose();
            
            cout<<"Performing eigendecomposition on the embedded fine mesh"<<endl;
            
            if(simple_mass_flag)
            {
                cout<<"using simple mass for fine mesh"<<endl;
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
                cout<<"matching fine modes at rest state"<<endl;
                init_matched_modes_list.resize(m_num_modes);
                init_matched_modes_list.setZero();
                for (int i = 0; i < m_num_modes; i++) {
                    init_matched_modes_list(i) = i;
                }
                cout<<"mapping initial eigenvectors"<<endl;
                init_coarse_eigenvectors = coarseEig.first;
                Eigen::MatrixXd  mapped_init_fine_eigenvectors;
                mapped_init_fine_eigenvectors = m_fineP*(*N) * m_coarseP.transpose()*init_coarse_eigenvectors;
                cout<<"mapped"<<endl;
                
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
                dist_map = dist_map.cwiseAbs(); // should have no effect
                cout<<"dist map: "<<endl;
                cout<<dist_map<<endl;
                
                for (int i = 0; i < m_num_modes; i++) {
                    Eigen::VectorXd::Index min_ind;
                    double min_val = dist_map.col(i).minCoeff(&min_ind);
                    if(abs(min_val) < 0.2) // set the matching tolerance criteria
                    {
                        init_matched_modes_list(i) = min_ind;
                        
                        cout<<"coarse mode "<<i<<" mapped to fine mode "<<min_ind<<". matching % = "<< 1-min_val<<endl;
                        cout<<"coarse eigenvalue "<<coarseEig.second(i)<<" matched to fine eigenvalue "<<fineEig.second(min_ind)<<endl;
                        
                    }
                    else
                    {
                        init_matched_modes_list(i) = -1; // -1 if can't find any matching mode
                        cout<<"coarse mode "<<i<<" has no matched fine mode "<<".  best matching % = "<<1-min_val<<endl;
                    }
                    
                }
                
                
            }
            
            cout<<"Writing fine eigen deformation to file (for Hausdorff distance check and reloading in static)."<<endl;
            //
            int mode = 0;
            Eigen::VectorXd fine_eig_def;
            for (mode = 0; mode < m_num_modes; ++mode) {
                fine_eig_def = (m_fineP.transpose()*m_Us.first.col(mode)).transpose();
                int idx = 0;
                // getGeometry().first is V
                Eigen::MatrixXd fine_V_disp = std::get<0>(m_fineWorld.getSystemList().getStorage())[0]->getGeometry().first;
                for(unsigned int vertexId=0;  vertexId < std::get<0>(m_fineWorld.getSystemList().getStorage())[0]->getGeometry().first.rows(); ++vertexId) {
                    
                    fine_V_disp(vertexId,0) += (1*fine_eig_def(idx));
                    idx++;
                    fine_V_disp(vertexId,1) += (1*fine_eig_def(idx));
                    idx++;
                    fine_V_disp(vertexId,2) += (1*fine_eig_def(idx));
                    idx++;
                }
                
                Eigen::MatrixXi fine_F;
                igl::boundary_facets(std::get<0>(m_fineWorld.getSystemList().getStorage())[0]->getGeometry().second,fine_F);
                
                igl::writeOBJ("finemesh_eigenmode" + std::to_string(mode) + ".obj",fine_V_disp,fine_F);
                
                
                std::string ffilename = "data/feigendef"+ std::to_string(mode) + "_"+ std::to_string(youngs) + "_" + std::to_string(poisson) + "_" + std::to_string(const_profile) + "_" + std::to_string(m_constraint_dir) + "_" + std::to_string(m_constraint_tol) + ".mtx";
                Eigen::saveMarket(fine_V_disp, ffilename );
            }
            
            
            // reset deformation if it is not zero. need zero (rest state configuration) to calculate static ratio.
            cout<<"writing coarse eigen deformation into files (for Hausdorff distance check)."<<endl;
            
            Eigen::VectorXd coarse_eig_def;
            for (mode = 0; mode < m_num_modes; ++mode)
            {
                coarse_eig_def = (m_coarseP.transpose()*m_coarseUs.first.col(mode)).transpose();
                //        //
                int idx = 0;
                //                    // getGeometry().first is V
                Eigen::MatrixXd coarse_V_disp_p = this->getImpl().getV();
                Eigen::MatrixXd coarse_V_disp_n = this->getImpl().getV();
                for(unsigned int vertexId=0;  vertexId < this->getImpl().getV().rows(); ++vertexId)
                {
                    coarse_V_disp_p(vertexId,0) += (1*coarse_eig_def(idx));
                    coarse_V_disp_n(vertexId,0) -= (1*coarse_eig_def(idx));
                    idx++;
                    coarse_V_disp_p(vertexId,1) += (1*coarse_eig_def(idx));
                    coarse_V_disp_n(vertexId,1) -= (1*coarse_eig_def(idx));
                    idx++;
                    coarse_V_disp_p(vertexId,2) += (1*coarse_eig_def(idx));
                    coarse_V_disp_n(vertexId,2) -= (1*coarse_eig_def(idx));
                    idx++;
                }
                Eigen::MatrixXi coarse_F;
                igl::boundary_facets(this->getImpl().getF(),coarse_F);
                
                //                    Eigen::MatrixXi coarse_F = surftri(this->getImpl().getV(), this->getImpl().getF());
                igl::writeOBJ("cmesh_eigenmode_p" + std::to_string(mode) + ".obj" ,coarse_V_disp_p, coarse_F);
                igl::writeOBJ("cmesh_eigenmode_n" + std::to_string(mode) + ".obj",coarse_V_disp_n, coarse_F);
            }
            
            
            
            cout<<"Calculating init ratio\n";
            //                std::cout<<"m_numConstraints "<< m_numConstraints << std::endl;
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
                if(!m_ratio_manual.isZero())
                {
                    cout<<"setting ratio manually"<<endl;
                    m_R(i) = m_ratio_manual(i);
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
                //#ifdef EDWIN_DEBUG
                cout<<"Ratios: "<<endl;
                std::cout<<m_R(i)<<std::endl;
                //#endif
                
            }
            ratio_calculated = true;
        }
        
        
        
        
        cout<<"using ratios: "<<endl;
        //                cout<<m_R<<endl;
        Eigen::VectorXd m_R_current(m_R.rows());
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
            cout<<"after mode matching: "<<endl;
            //            cout<<m_R_current<<endl;
        }
        //        mode composition not implemented
        //        else if(mode_matching_tol == 2)
        //        {
        //            m_R_current = dist_map.transpose()*m_R;
        //            cout<<"after mode composition: "<<endl;
        //
        //        }
        cout<<m_R_current<<endl;
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
        
        //build a projection matrix P which projects fixed points out of a physical syste
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
    
    
    //per vertex accessors. takes the state of the coarse mesh
    inline Eigen::Vector3x<double> getFinePosition(const State<double> &state, unsigned int vertexId) const {
        return m_Vf.row(vertexId).transpose() + m_N.block(3*vertexId, 0, 3, m_N.cols())*(*this).getImpl().getElement(m_elements[vertexId])->q(state);
    }
    
    
    inline Eigen::VectorXx<double> getFinePositionFull(const Eigen::VectorXd V) const {
        
        Eigen::Map<Eigen::VectorXd> eigen_fine_pos0(fine_pos0,m_fineWorld.getNumQDOFs());
        
        return eigen_fine_pos0 + (*N)*V;
    }
    
    
    inline World<double, std::tuple<PhysicalSystemImpl *>,
    std::tuple<ForceSpringFEMParticle<double> *, ForceParticlesGravity<double> *>,
    std::tuple<ConstraintFixedPoint<double> *> > & getFineWorld(){ return m_fineWorld;}
    
    
    std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > coarseEig;
    std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > fineEig;
    std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > m_Us;
    Eigen::MatrixXx<double> prev_coarseEigenvectors;
    
    Eigen::VectorXx<double> coarseEigenvalues;
    Eigen::VectorXx<double> fineEigenvalues;
    Eigen::MatrixXx<double> prev_coarse_eigenvectors;
    Eigen::MatrixXx<double> init_coarse_eigenvectors;
    Eigen::VectorXx<double> init_coarse_eigenvalues;
    Eigen::MatrixXx<double> diff_coarse_eigenvectors;
    Eigen::MatrixXx<double> coarseEigenvectors;
    Eigen::MatrixXx<double> fineEigenvectors;
    
    Eigen::VectorXx<double> m_ratio_manual;
    int m_compute_frequency;
    Eigen::VectorXx<double> m_feval_manual;
    std::string m_finepos_manual;
    
    Eigen::SparseMatrix<double> m_fineP;
    Eigen::SparseMatrix<double> m_coarseP;
    
    //        Eigen::MatrixXd coarse_V_disp_p;
    
    AssemblerEigenSparseMatrix<double> m_coarseMassMatrix;
    AssemblerEigenSparseMatrix<double> m_fineMassMatrix;
    
    AssemblerEigenVector<double> m_fineforceVector;
    AssemblerEigenVector<double> m_finefExt;
    
    AssemblerEigenVector<double> m_forceVector;
    
    AssemblerEigenSparseMatrix<double> N;
    
    SolverPardiso<Eigen::SparseMatrix<double, Eigen::RowMajor> > m_pardiso_test;
    Eigen::VectorXd minvf;
    Eigen::VectorXd minvfCP;
    
    // rest state of fine q
    double* fine_pos0  = NULL;
    // rest state of coarse q
    double* coarse_pos0 = NULL;
    
    double* fine_q_transfered = NULL;
    
    bool haus = false;
    
    int flag = 0;
    
    // TODO: parameters for rayleigh damping. should not be here...
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
    int ratio_recalculation_switch;
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
    
    bool init_mode_matching_flag;
    double mode_matching_tol;
    Eigen::MatrixXd dist_map;
    
    int calculate_matching_data_flag;
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
    Eigen::VectorXi m_coarseFixedVerts;
    
    
    double youngs;
    double poisson;
    
    int m_constraint_dir;
    double m_constraint_tol;
    
    //num modes to correct
    unsigned int m_num_modes;
    //    unsigned int m_numToCorrect;
    
    //Ratios diagonal matrix, stored as vector
    
    Eigen::VectorXd m_I;
    
    
    
    Eigen::MatrixXi m_Ff;
    Eigen::MatrixXi m_Fc;
    Eigen::MatrixXd m_N;
    //m_elements[i] is a n-vector that stores the index of the element containing the ith vertex in the embedded mesh
    Eigen::VectorXi m_elements;
    
    State<double> restFineState;
    
    
    bool ratio_recalculation_flag;
    
    
    unsigned int const_profile;
    unsigned int m_numConstraints;
private:
    
};

#endif /* EigenFit_h */

