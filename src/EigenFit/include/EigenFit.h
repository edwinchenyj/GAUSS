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
#include <SparseRegularInversePardiso.h>
#include <igl/boundary_facets.h>

using namespace Gauss;
using namespace FEM;
using namespace ParticleSystem; //For Force Spring
using std::cout;
using std::endl;

// subclass a hard-coded templated class from PhysicalSystemFEM
// this means that this EigenFit only works for NeohookeanHFixedTets
class EigenFit: public PhysicalSystemFEM<double, NeohookeanHFixedTet>{
    //class EigenFit: public PhysicalSystemFEM<double, NeohookeanHFixedTet>{
    
public:
    // alias the hard-coded template name. Easier to read
    // the following lines read: the Physical System Implementation used here is a neo-hookean tet class
    //    using PhysicalSystemImpl = PhysicalSystemFEM<double, NeohookeanHFixedTet>;
    using PhysicalSystemImpl = PhysicalSystemFEM<double, NeohookeanHFixedTet>;
    
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
    EigenFit(Eigen::MatrixXx<double> &Vc, Eigen::MatrixXi &Fc,Eigen::MatrixXx<double> &Vf, Eigen::MatrixXi &Ff, int dynamic_switch, double youngs, double poisson, int constraintDir, double constraintTol, unsigned int cswitch, unsigned int hausdorff_dist, unsigned int numModes, std::string cmeshname, std::string fmeshname, Eigen::VectorXx<double> ratio_manual, int compute_frequency, bool simple_mass_flag ) : PhysicalSystemImpl(Vc,Fc)
    {
        this->simple_mass_flag = simple_mass_flag;
        coarse_mass_calculated = false;
        fine_mass_calculated = false;
        
        step_number = 0;
        cout<<"Hausdorff distance flag: "<<hausdorff_dist<<endl;
        if(numModes != 0)
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
            
            m_constraintDir = constraintDir;
            m_constraintTol = constraintTol;
            
            m_ratio_manual = ratio_manual;
            m_compute_frequency = compute_frequency;
            m_feval_manual.resize(numModes);
            m_feval_manual.setZero();
            
            
            std::cout<<m_Vf.rows()<<std::endl;
            std::cout<<m_Vf.cols()<<std::endl;
            
            ratio_recalculation_switch = dynamic_switch;
            constraint_switch = cswitch;
            
            //element[i] is a n-vector that stores the index of the element containing the ith vertex in the embedded mesh
            // *N is the upsample operator
            // (*N).transpose is downsample operator
            getShapeFunctionMatrix(N,m_elements,Vf, (*this).getImpl());
            
            Eigen::Vector3x<double> vertex = m_Vf.row(0);
            
            // col of the shape func, I think. 12 for Tet, 24 for hex
            unsigned int numCols = (*this).getImpl().getElements()[0]->N(vertex.data()).cols();
            unsigned int el;
            
            // set the flag
            haus = hausdorff_dist;
            
            // setup the fine mesh
            PhysicalSystemImpl *m_fineMeshSystem = new PhysicalSystemImpl(Vf,Ff);
            
            cout<<"Setting fine mesh..."<<endl;
            // set up material parameters
            this->youngs = youngs;
            this->poisson = poisson;
            for(unsigned int iel=0; iel<m_fineMeshSystem->getImpl().getF().rows(); ++iel) {
                
                m_fineMeshSystem->getImpl().getElement(iel)->setParameters(youngs, poisson);
                
            }
            m_fineWorld.addSystem(m_fineMeshSystem);
            
            //            Eigen::Map<Eigen::VectorXd> fine_q = mapStateEigen<0>(m_fineWorld);
            //            fine_q_transfered = new double[fine_q.rows()];
            
            
            //            Eigen::Map<Eigen::VectorXd> eigen_fineq_q_transfered(fine_q_transfered,fine_q.rows());
            //            eigen_fineq_q_transfered.setZero();
            
            //            fine_q_transfered = mapStateEigen<0>(m_fineWorld);
            
            //       constraints
            Eigen::SparseMatrix<double> fineP;
            Eigen::SparseMatrix<double> coarseP;
            
            cout<<"Setting fine mesh constraints..."<<endl;
            if (constraint_switch == 0) {
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
            else if (constraint_switch == 1)
            {
                cout<<"Setting constraint on the fine mesh and constructing fine mesh projection matrix"<<endl;
                // default constraint
                //            fix displacement
                fixDisplacementMin(m_fineWorld, m_fineMeshSystem, constraintDir, constraintTol);
                
                m_fineWorld.finalize();
                // hard-coded constraint projection
                
                std::string cconstraint_file_name = "data/" + cmeshname + "_const" + std::to_string(constraint_switch) + "_" +std::to_string(constraintDir)+"_"+std::to_string(constraintTol)+".mtx";
                std::string fconstraint_file_name = "data/" + fmeshname + "_const" + std::to_string(constraint_switch) + "_" +std::to_string(constraintDir)+"_"+std::to_string(constraintTol)+".mtx";
                Eigen::VectorXi fineFixedVerts;
                Eigen::VectorXi coarseFixedVerts;
                cout<<"Loading vertices and setting projection matrix..."<<endl;
                if(!Eigen::loadMarketVector(coarseFixedVerts,cconstraint_file_name))
                {
                    cout<<cconstraint_file_name<<endl;
                    cout<<"File does not exist for coarse mesh, creating new file..."<<endl;
                    coarseFixedVerts = minVertices(this, constraintDir, constraintTol);
                    Eigen::saveMarketVector(coarseFixedVerts,cconstraint_file_name);
                }
                if(!Eigen::loadMarketVector(fineFixedVerts,fconstraint_file_name))
                {
                    cout<<fconstraint_file_name<<endl;
                    cout<<"File does not exist for fine mesh, creating new file..."<<endl;
                    fineFixedVerts = minVertices(m_fineMeshSystem, constraintDir, constraintTol);
                    Eigen::saveMarketVector(fineFixedVerts,fconstraint_file_name);
                }
                
                
                m_fineFixedVerts = fineFixedVerts;
                fineP = fixedPointProjectionMatrix(fineFixedVerts, *m_fineMeshSystem,m_fineWorld);
                m_fineP = fineP;
                
                coarseP = fixedPointProjectionMatrixCoarse(coarseFixedVerts);
                m_coarseP = coarseP;
                Eigen::saveMarketDat(m_fineP, fconstraint_file_name+"_fineP.dat");
                Eigen::saveMarketDat(m_coarseP, cconstraint_file_name+"_cineP.dat");
                
                // only need to record one because only need to know if it's 0, 3, or 6. either fine or coarse would work
                m_numConstraints = fineFixedVerts.size();
                
            }
            else if (constraint_switch == 2)
            {
                
                
                std::string cconstraint_file_name = "data/" +cmeshname + "_const" + std::to_string(constraint_switch) + "_" +std::to_string(constraintDir)+"_"+std::to_string(constraintTol)+".mtx";
                std::string fconstraint_file_name = "data/" +fmeshname + "_const" + std::to_string(constraint_switch) + "_" +std::to_string(constraintDir)+"_"+std::to_string(constraintTol)+".mtx";
                Eigen::VectorXi fineMovingVerts;
                Eigen::VectorXi coarseMovingVerts;
                cout<<"Loading vertices and setting projection matrix..."<<endl;
                if(!Eigen::loadMarketVector(coarseMovingVerts,cconstraint_file_name))
                {
                    cout<<cconstraint_file_name<<endl;
                    cout<<"File does not exist for coarse mesh, creating new file..."<<endl;
                    coarseMovingVerts = minVertices(this, constraintDir, constraintTol);
                    Eigen::saveMarketVector(coarseMovingVerts,cconstraint_file_name);
                }
                if(!Eigen::loadMarketVector(fineMovingVerts,fconstraint_file_name))
                {
                    cout<<fconstraint_file_name<<endl;
                    cout<<"File does not exist for fine mesh, creating new file..."<<endl;
                    fineMovingVerts = minVertices(m_fineMeshSystem, constraintDir, constraintTol);
                    Eigen::saveMarketVector(fineMovingVerts,fconstraint_file_name);
                }
                
                
                m_fineMovingVerts = fineMovingVerts;
                fineP = fixedPointProjectionMatrix(fineMovingVerts, *m_fineMeshSystem,m_fineWorld);
                m_fineP = fineP;
                
                
                coarseP = fixedPointProjectionMatrixCoarse(coarseMovingVerts);
                m_coarseP = coarseP;
                
                Eigen::saveMarketDat(m_fineP, fconstraint_file_name+"_fineP.dat");
                Eigen::saveMarketDat(m_coarseP, cconstraint_file_name+"_cineP.dat");
                
                //                Eigen::VectorXi fineMovingVerts = minVertices(m_fineMeshSystem, constraintDir, constraintTol);//indices for moving parts
                //                Eigen::VectorXi fineMovingVerts;
                //
                //                // read constraints
                //                cout<<"Loading fine mesh constraint vertices list"<<endl;
                //                Eigen::loadMarketVector(fineMovingVerts,  "def_init/" + fmeshname + "_fixed_min_verts.mtx");
                //                m_fineMovingVerts = fineMovingVerts;
                
                std::vector<ConstraintFixedPoint<double> *> fineMovingConstraints;
                
                for(unsigned int ii=0; ii<fineMovingVerts.rows(); ++ii) {
                    fineMovingConstraints.push_back(new ConstraintFixedPoint<double>(&m_fineMeshSystem->getQ()[fineMovingVerts[ii]], Eigen::Vector3d(0,0,0)));
                    m_fineWorld.addConstraint(fineMovingConstraints[ii]);
                }
                m_fineWorld.finalize(); //After this all we're ready to go (clean up the interface a bit later)
                
                // hard-coded constraint projection
                //                fineP = fixedPointProjectionMatrix(fineMovingVerts, *m_fineMeshSystem,m_fineWorld);
                //                m_fineP = fineP;
                // only need to record one because only need to know if it's 0, 3, or 6. either fine or coarse is fine
                m_numConstraints = fineMovingVerts.size();
                
                //                Eigen::VectorXi coarseMovingVerts = minVertices(this, constraintDir, constraintTol);
                //                Eigen::VectorXi coarseMovingVerts;
                //                cout<<"Loading coarse mesh constraint vertices list"<<endl;
                //                Eigen::loadMarketVector(coarseMovingVerts,  "def_init/" + cmeshname + "_fixed_min_verts.mtx");
                //
                //                coarseP = fixedPointProjectionMatrixCoarse(coarseMovingVerts);
                //                m_coarseP = coarseP;
                
            }
            else if (constraint_switch == 3)
            {
                std::string cconstraint_file_name = "data/" +cmeshname + "_const" + std::to_string(constraint_switch) + "_" +std::to_string(constraintDir)+"_"+std::to_string(constraintTol)+".mtx";
                std::string fconstraint_file_name = "data/" +fmeshname + "_const" + std::to_string(constraint_switch) + "_" +std::to_string(constraintDir)+"_"+std::to_string(constraintTol)+".mtx";
                Eigen::VectorXi fineFixedVerts;
                Eigen::VectorXi coarseFixedVerts;
                cout<<"Loading vertices and setting projection matrix..."<<endl;
                if(!Eigen::loadMarketVector(coarseFixedVerts,cconstraint_file_name))
                {
                    cout<<cconstraint_file_name<<endl;
                    cout<<"File does not exist for coarse mesh, creating new file..."<<endl;
                    coarseFixedVerts = minVertices(this, constraintDir, constraintTol);
                    Eigen::saveMarketVector(coarseFixedVerts,cconstraint_file_name);
                }
                if(!Eigen::loadMarketVector(fineFixedVerts,fconstraint_file_name))
                {
                    cout<<fconstraint_file_name<<endl;
                    cout<<"File does not exist for fine mesh, creating new file..."<<endl;
                    fineFixedVerts = minVertices(m_fineMeshSystem, constraintDir, constraintTol);
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
                m_coarseP = coarseP;
                
                Eigen::saveMarketDat(m_fineP, fconstraint_file_name+"_fineP.dat");
                Eigen::saveMarketDat(m_coarseP, cconstraint_file_name+"_cineP.dat");
                
                
                m_numConstraints = fineFixedVerts.rows();
                
                
            }
            else if (constraint_switch == 4 || constraint_switch == 5 || constraint_switch == 6 || constraint_switch == 7 || constraint_switch == 8)
            {
                
                cout<<"Setting constraint on the fine mesh and constructing fine mesh projection matrix"<<endl;
                
                std::string cconstraint_file_name = "data/" +cmeshname + "_const" + std::to_string(constraint_switch) + "_" +std::to_string(constraintDir)+"_"+std::to_string(constraintTol)+".mtx";
                std::string fconstraint_file_name = "data/" +fmeshname + "_const" + std::to_string(constraint_switch) + "_" +std::to_string(constraintDir)+"_"+std::to_string(constraintTol)+".mtx";
                Eigen::VectorXi fineMovingVerts;
                Eigen::VectorXi coarseMovingVerts;
                cout<<"Loading vertices and setting projection matrix..."<<endl;
                if(!Eigen::loadMarketVector(coarseMovingVerts,cconstraint_file_name))
                {
                    cout<<cconstraint_file_name<<endl;
                    cout<<"File does not exist for coarse mesh, creating new file..."<<endl;
                    coarseMovingVerts = minVertices(this, constraintDir, constraintTol);
                    Eigen::saveMarketVector(coarseMovingVerts,cconstraint_file_name);
                }
                if(!Eigen::loadMarketVector(fineMovingVerts,fconstraint_file_name))
                {
                    cout<<fconstraint_file_name<<endl;
                    cout<<"File does not exist for fine mesh, creating new file..."<<endl;
                    fineMovingVerts = minVertices(m_fineMeshSystem, constraintDir, constraintTol);
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
                m_coarseP = coarseP;
                
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
                m_numModes = numModes;
                
                // put random value to m_R for now
                m_R.setConstant(m_numModes, 1.0);
                ratio_calculated = false;
                m_I.setConstant(m_numModes, 1.0);
            }
            else if (m_numConstraints == 3)
            {
                // if constraint is  a point constaint
                m_numModes = numModes;
                m_numModes = m_numModes + 3;
                
                // put random value to m_R for now
                m_R.setConstant(m_numModes, 1.0);
                m_R(0) = 1.0;
                m_R(1) = 1.0;
                m_R(2) = 1.0;
                ratio_calculated = false;
                m_I.setConstant(m_numModes, 1.0);
            }
            else
            {
                cout<<"No constraints so ignore the first 6 eigenvalues."<<endl;
                // otherwise, free boundary
                m_numModes = numModes;
                m_numModes = m_numModes + 6;
                
                // put random value to m_R for now
                m_R.setConstant(m_numModes, 1.0);
                m_R(0) = 1.0;
                m_R(1) = 1.0;
                m_R(2) = 1.0;
                m_R(3) = 1.0;
                m_R(4) = 1.0;
                m_R(5) = 1.0;
                
                ratio_calculated = false;
                m_I.setConstant(m_numModes, 1.0);
                
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
            
            if(simple_mass_flag && !fine_mass_calculated)
            {
                Eigen::VectorXx<double> ones(m_fineP.rows());
                ones.setOnes();
                fine_mass_lumped = ((*massMatrix)*ones);
                fine_mass_lumped_inv = fine_mass_lumped.cwiseInverse();
                
                fine_mass_calculated = true;
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
    
    
    ~EigenFit() {delete fine_pos0;
    }
    
    void calculateFineMesh(){
        World<double, std::tuple<PhysicalSystemImpl *>,
        std::tuple<ForceSpringFEMParticle<double> *, ForceParticlesGravity<double> *>,
        std::tuple<ConstraintFixedPoint<double> *> > &world = m_fineWorld;
        
        AssemblerEigenSparseMatrix<double> &fineStiffnessMatrix = m_fineStiffnessMatrix;
        
        //            std::cout<<
        
        //get stiffness matrix
        ASSEMBLEMATINIT(fineStiffnessMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
        ASSEMBLELIST(fineStiffnessMatrix, world.getSystemList(), getStiffnessMatrix);
        ASSEMBLELIST(fineStiffnessMatrix, world.getForceList(), getStiffnessMatrix);
        ASSEMBLEEND(fineStiffnessMatrix);
        //        Eigen::saveMarket(fine_q, "fineq0.dat");
        
        // for edwin debug
        AssemblerEigenVector<double> &fineforceVector = m_fineforceVector;
        AssemblerEigenVector<double> &finefExt = m_finefExt;
        
        //Need to filter internal forces seperately for this applicat
        ASSEMBLEVECINIT(fineforceVector, world.getNumQDotDOFs());
        ASSEMBLELIST(fineforceVector, world.getSystemList(), getImpl().getInternalForce);
        ASSEMBLEEND(fineforceVector);
        
        ASSEMBLEVECINIT(finefExt, world.getNumQDotDOFs());
        ASSEMBLELIST(finefExt, world.getSystemList(), getImpl().getBodyForce);
        ASSEMBLEEND(finefExt);
        
        // add external force
        (*fineforceVector) = m_fineP * (*fineforceVector);
        
        (*fineforceVector) = (*fineforceVector) + m_fineP *(*finefExt);
        
        //constraint Projection
        (*fineStiffnessMatrix) = m_fineP*(*fineStiffnessMatrix)*m_fineP.transpose();
        
        //Eigendecomposition for the embedded fine mesh
        //        std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > m_Us;
        m_Us = generalizedEigenvalueProblemNotNormalized((*fineStiffnessMatrix), (*m_fineMassMatrix), m_numModes, 0.00);
        //        m_Us = generalizedEigenvalueProblemNotNormalized((*stiffnessMatrix), m_M, 10, 0.00);
        //    Eigen::saveMarketDat(m_M,"mass.dat");
        //    Eigen::saveMarketDat(m_Us.first,"not_normalized.dat");
        Eigen::VectorXd normalizing_const;
        normalizing_const = (m_Us.first.transpose() * (*m_fineMassMatrix) * m_Us.first).diagonal();
        normalizing_const = normalizing_const.cwiseSqrt().cwiseInverse();
        
        m_Us.first = m_Us.first * (normalizing_const.asDiagonal());
        //
        fineEigMassProj = m_Us;
        fineEig = m_Us;
        fineEigMassProj.first = (*m_fineMassMatrix)*fineEigMassProj.first;
    }
    
    // calculate data, TODO: the first two parameter should be const
    template<typename MatrixAssembler>
    //    void calculateEigenFitData(State<double> &state, MatrixAssembler &coarseMassMatrix, MatrixAssembler &coarseStiffnessMatrix,  std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > &m_coarseUs, Eigen::MatrixXd &Y, Eigen::MatrixXd &Z){
    bool calculateEigenFitData(const Eigen::VectorXx<double> &q, MatrixAssembler &coarseMassMatrix, MatrixAssembler &coarseStiffnessMatrix,  std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > &m_coarseUs, Eigen::MatrixXd &Y, Eigen::MatrixXd &Z){
        
        //        Eigen::saveMarketDat((*coarseStiffnessMatrix), "coarseStiffness.dat");
        //        Eigen::saveMarketDat((*coarseMassMatrix), "coarseMass.dat");
        
        
        if(simple_mass_flag && !coarse_mass_calculated)
        {
            Eigen::VectorXx<double> ones(m_coarseP.rows());
            ones.setOnes();
            coarse_mass_lumped = ((*coarseMassMatrix)*ones);
            coarse_mass_lumped_inv = coarse_mass_lumped.cwiseInverse();
            
            coarse_mass_calculated = true;
        }
        
        m_coarseUs = generalizedEigenvalueProblemNotNormalized((*coarseStiffnessMatrix), (*coarseMassMatrix), m_numModes,0.0);
        Eigen::VectorXd normalizing_const;
        normalizing_const = (m_coarseUs.first.transpose() * (*coarseMassMatrix) * m_coarseUs.first).diagonal();
        normalizing_const = normalizing_const.cwiseSqrt().cwiseInverse();
        
        m_coarseUs.first = m_coarseUs.first * (normalizing_const.asDiagonal());
        
//        cout<<"coarse eval:"<<m_coarseUs.second<<endl;
        //        Eigen::saveMarketDat(m_coarseUs.first, "coarseEigenvectors.dat");
        //        Eigen::saveMarketVectorDat(m_coarseUs.second, "coarseEigenvalues.dat");
        coarseEigMassProj = m_coarseUs;
        coarseEig = m_coarseUs;
        coarseEigMassProj.first = (*coarseMassMatrix)*coarseEigMassProj.first;
        
        
        std::cout<<"Dynamic switch: "<<ratio_recalculation_switch<<std::endl;
        if((!ratio_calculated))
        {
            if( ratio_recalculation_switch == 1 || ratio_recalculation_switch == 0 || ratio_recalculation_switch == 6)
            {
                
                
                if(ratio_recalculation_switch == 0 || ratio_recalculation_switch == 6)
                {
                    
                        World<double, std::tuple<PhysicalSystemImpl *>,
                        std::tuple<ForceSpringFEMParticle<double> *, ForceParticlesGravity<double> *>,
                        std::tuple<ConstraintFixedPoint<double> *> > &world = m_fineWorld;
                        
                        Eigen::Map<Eigen::VectorXd> fine_q = mapStateEigen<0>(m_fineWorld);
                        
                        //            double pd_fine_pos[world.getNumQDOFs()]; // doesn't work for MSVS
                        Eigen::Map<Eigen::VectorXd> eigen_fine_pos0(fine_pos0,world.getNumQDOFs());
                        
                        Eigen::VectorXx<double> posFull;
                        posFull = this->getFinePositionFull(q);
                        //
                        fine_q = posFull - eigen_fine_pos0;
                        //        lambda can't capture member variable, so create a local one for lambda in ASSEMBLELIST
                        AssemblerEigenSparseMatrix<double> &fineStiffnessMatrix = m_fineStiffnessMatrix;
                    
                        //get stiffness matrix
                        ASSEMBLEMATINIT(fineStiffnessMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
                        ASSEMBLELIST(fineStiffnessMatrix, world.getSystemList(), getStiffnessMatrix);
                        ASSEMBLELIST(fineStiffnessMatrix, world.getForceList(), getStiffnessMatrix);
                        ASSEMBLEEND(fineStiffnessMatrix);
                        
                        
                        //constraint Projection
                        (*fineStiffnessMatrix) = m_fineP*(*fineStiffnessMatrix)*m_fineP.transpose();
                        
                        cout<<"Performing eigendecomposition on the embedded fine mesh"<<endl;
                        m_Us = generalizedEigenvalueProblemNotNormalized(((*fineStiffnessMatrix)), (*m_fineMassMatrix), m_numModes, 0.00);
                        Eigen::VectorXd normalizing_const;
                        normalizing_const = (m_Us.first.transpose() * (*m_fineMassMatrix) * m_Us.first).diagonal();
                        normalizing_const = normalizing_const.cwiseSqrt().cwiseInverse();
                        
                        m_Us.first = m_Us.first * (normalizing_const.asDiagonal());
                        
                        
                        fineEigMassProj = m_Us;
                        fineEig = m_Us;
                        fineEigMassProj.first = (*m_fineMassMatrix)*fineEigMassProj.first;
                    
                }
                else
                {
                    
                    // dynamic eigenfit dynamic flag = 1
                    World<double, std::tuple<PhysicalSystemImpl *>,
                    std::tuple<ForceSpringFEMParticle<double> *, ForceParticlesGravity<double> *>,
                    std::tuple<ConstraintFixedPoint<double> *> > &world = m_fineWorld;
                    
                    Eigen::Map<Eigen::VectorXd> fine_q = mapStateEigen<0>(m_fineWorld);
                    fine_q = (*(this->N)) * q;
                    
                    //        lambda can't capture member variable, so create a local one for lambda in ASSEMBLELIST
                    AssemblerEigenSparseMatrix<double> &fineStiffnessMatrix = m_fineStiffnessMatrix;
                    
                    //get stiffness matrix
                    ASSEMBLEMATINIT(fineStiffnessMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
                    ASSEMBLELIST(fineStiffnessMatrix, world.getSystemList(), getStiffnessMatrix);
                    ASSEMBLELIST(fineStiffnessMatrix, world.getForceList(), getStiffnessMatrix);
                    ASSEMBLEEND(fineStiffnessMatrix);
                    
                    
                    //constraint Projection
                    (*fineStiffnessMatrix) = m_fineP*(*fineStiffnessMatrix)*m_fineP.transpose();
                    
                    cout<<"Performing eigendecomposition on the embedded fine mesh"<<endl;
                    m_Us = generalizedEigenvalueProblemNotNormalized(((*fineStiffnessMatrix)), (*m_fineMassMatrix), m_numModes, 0.00);
                    Eigen::VectorXd normalizing_const;
                    normalizing_const = (m_Us.first.transpose() * (*m_fineMassMatrix) * m_Us.first).diagonal();
                    normalizing_const = normalizing_const.cwiseSqrt().cwiseInverse();
                    
                    m_Us.first = m_Us.first * (normalizing_const.asDiagonal());
                    
                    fineEigMassProj = m_Us;
                    fineEig = m_Us;
                    fineEigMassProj.first = (*m_fineMassMatrix)*fineEigMassProj.first;
                    
                }
                
                if(haus || ratio_recalculation_switch == 0 || ratio_recalculation_switch == 6)
                {
                    cout<<"Writing fine eigen deformation to file (for Hausdorff distance check and reloading in static)."<<endl;
                    //
                    int mode = 0;
                    Eigen::VectorXd fine_eig_def;
                    for (mode = 0; mode < m_numModes; ++mode) {
                        fine_eig_def = (m_fineP.transpose()*m_Us.first.col(mode)).transpose();
                        int idx = 0;
                        // getGeometry().first is V
                        Eigen::MatrixXd fine_V_disp = std::get<0>(m_fineWorld.getSystemList().getStorage())[0]->getGeometry().first;
                        for(unsigned int vertexId=0;  vertexId < std::get<0>(m_fineWorld.getSystemList().getStorage())[0]->getGeometry().first.rows(); ++vertexId) {
                            
                            // because getFinePosition is in EigenFit, not another physical system Impl, so don't need getImpl()
                            fine_V_disp(vertexId,0) += (1*fine_eig_def(idx));
                            idx++;
                            fine_V_disp(vertexId,1) += (1*fine_eig_def(idx));
                            idx++;
                            fine_V_disp(vertexId,2) += (1*fine_eig_def(idx));
                            idx++;
                        }
                        
                        
                        //                    Eigen::MatrixXi fine_F = surftri(std::get<0>(m_fineWorld.getSystemList().getStorage())[0]->getGeometry().first, std::get<0>(m_fineWorld.getSystemList().getStorage())[0]->getGeometry().second);
                        Eigen::MatrixXi fine_F;
                        igl::boundary_facets(std::get<0>(m_fineWorld.getSystemList().getStorage())[0]->getGeometry().second,fine_F);
                        
                        //                    igl::writeOBJ("fine_mesh_eigen_mode" + std::to_string(mode) + ".obj",std::get<0>(m_fineWorld.getSystemList().getStorage())[0]->getGeometry().first,fine_F);
                        igl::writeOBJ("finemesh_neohookeaneigenmode" + std::to_string(mode) + ".obj",fine_V_disp,fine_F);
                        
                        std::string ffilename = "data/feigendefneohookean"+ std::to_string(mode) + "_"+ std::to_string(youngs) + "_" + std::to_string(poisson) + "_" + std::to_string(constraint_switch) + "_" + std::to_string(m_constraintDir) + "_" + std::to_string(m_constraintTol) + ".mtx";
                        Eigen::saveMarket(fine_V_disp, ffilename );
                    }
                }
                
                if(haus)
                {
                    // reset deformation if it is not zero. need zero (rest state configuration) to calculate static ratio.
                    cout<<"writing coarse eigen deformation into files (for Hausdorff distance check)."<<endl;
                    unsigned int mode = 0;
                    unsigned int idx = 0;
                    Eigen::VectorXd coarse_eig_def;
                    for (mode = 0; mode < m_numModes; ++mode) {
                        coarse_eig_def = (m_coarseP.transpose()*m_coarseUs.first.col(mode)).transpose();
                        //        //
                        idx = 0;
                        //                    // getGeometry().first is V
                        Eigen::MatrixXd coarse_V_disp_p = this->getImpl().getV();
                        Eigen::MatrixXd coarse_V_disp_n = this->getImpl().getV();
                        for(unsigned int vertexId=0;  vertexId < this->getImpl().getV().rows(); ++vertexId) {
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
                        igl::writeOBJ("cmesh_neohookean_eigenmode_p" + std::to_string(mode) + ".obj" ,coarse_V_disp_p, coarse_F);
                        igl::writeOBJ("cmesh_neohookean_eigenmode_n" + std::to_string(mode) + ".obj",coarse_V_disp_n, coarse_F);
                        std::string cfilename = "ceigendefneohookean"+ std::to_string(mode) + "_"+ std::to_string(youngs) + "_" + std::to_string(poisson) + "_" + std::to_string(constraint_switch) + "_" + std::to_string(m_constraintDir) + "_" + std::to_string(m_constraintTol) + ".mtx";
                        Eigen::saveMarket(coarse_V_disp_n, cfilename);
                        
                        
                        cout<<"Loading coarse eigen deformation."<<endl;
                        igl::readOBJ("cmesh_neohookean_eigenmode_p" + std::to_string(mode) + ".obj",coarse_V_disp_p, coarse_F);
                        igl::readOBJ("cmesh_neohookean_eigenmode_n" + std::to_string(mode) + ".obj",coarse_V_disp_n, coarse_F);
                        
                        double dist_p, dist_n, dist_scaled;
                        Eigen::MatrixXd coarse_V_disp = this->getImpl().getV();
                        Eigen::MatrixXd fine_V_disp = std::get<0>(m_fineWorld.getSystemList().getStorage())[0]->getGeometry().first;
                        Eigen::Array3d xyz_scales(coarse_V_disp.col(0).maxCoeff() - coarse_V_disp.col(0).minCoeff(), coarse_V_disp.col(1).maxCoeff() - coarse_V_disp.col(1).minCoeff(),coarse_V_disp.col(2).maxCoeff() - coarse_V_disp.col(2).minCoeff());
                        double max_scale = xyz_scales.abs().maxCoeff();
                        Eigen::MatrixXi fine_F;
                        igl::boundary_facets(std::get<0>(m_fineWorld.getSystemList().getStorage())[0]->getGeometry().second,fine_F);
                        
                        igl::hausdorff(fine_V_disp, fine_F, coarse_V_disp_p, coarse_F, dist_p);
                        igl::hausdorff(fine_V_disp, fine_F, coarse_V_disp_n, coarse_F, dist_n);
                        if(dist_p < dist_n) dist_scaled = dist_p/max_scale;
                        else dist_scaled = dist_n/max_scale;
                        
                        
                        cout<<"Eigen mode "<<mode<<endl;
                        std::cout<<"Dist scaled: "<< dist_scaled<<std::endl;
                        
                        // fail on hausdorff distance check
                        if( dist_scaled > 0.4)
                        {
                            //                        if(haus)
                            //                        {
                            // if required to check hausdorff and failed, return 1
                            return true;
                            //                        }
                        }
                    }
                }
                
                
                cout<<"Calculating ratio\n";
                //                std::cout<<"m_numConstraints "<< m_numConstraints << std::endl;
                for(int i = 0; i < m_numModes; ++i)
                {
                    m_R(i) = m_Us.second(i)/m_coarseUs.second(i);
                    
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
            else if (ratio_recalculation_switch == 4)
            {
                
                if(step_number%m_compute_frequency == 0) // update once every few frames
                {
                    
                    Eigen::MatrixXd fine_V;
                    Eigen::MatrixXd fine_F;
                    if(igl::readOBJ(m_finepos_manual + std::to_string(step_number) + ".obj",fine_V, fine_F))
                    {
                        cout<<"fine mesh loaded at frame "<<step_number<<endl;
                        
                        Eigen::VectorXd fine_displacement(3*fine_V.rows());
                        fine_displacement.setZero();
                        int idx = 0;
                        // getGeometry().first is V
                        Eigen::MatrixXd fine_V_disp0 = std::get<0>(m_fineWorld.getSystemList().getStorage())[0]->getGeometry().first;
                        for(unsigned int vertexId=0;  vertexId < std::get<0>(m_fineWorld.getSystemList().getStorage())[0]->getGeometry().first.rows(); ++vertexId) {
                            
                            fine_displacement(idx) = fine_V(vertexId,0);
                            idx++;
                            
                            fine_displacement(idx) = fine_V(vertexId,1);
                            idx++;
                            
                            fine_displacement(idx) = fine_V(vertexId,2);
                            idx++;
                        }
                        
                        World<double, std::tuple<PhysicalSystemImpl *>,
                        std::tuple<ForceSpringFEMParticle<double> *, ForceParticlesGravity<double> *>,
                        std::tuple<ConstraintFixedPoint<double> *> > &world = m_fineWorld;
                        
                        Eigen::Map<Eigen::VectorXd> fine_q = mapStateEigen<0>(m_fineWorld);
                        
                        fine_q = fine_displacement;
                        //            double pd_fine_pos[world.getNumQDOFs()]; // doesn't work for MSVS
                        Eigen::Map<Eigen::VectorXd> eigen_fine_pos0(fine_pos0,world.getNumQDOFs());
                        
                        //                        Eigen::VectorXx<double> posFull;
                        //                        posFull = this->getFinePositionFull(q);
                        //
                        fine_q = fine_displacement - eigen_fine_pos0;
                        
                        AssemblerEigenSparseMatrix<double> &fineStiffnessMatrix = m_fineStiffnessMatrix;
                        
                        //            std::cout<<
                        
                        //get stiffness matrix
                        ASSEMBLEMATINIT(fineStiffnessMatrix, world.getNumQDotDOFs(), world.getNumQDotDOFs());
                        ASSEMBLELIST(fineStiffnessMatrix, world.getSystemList(), getStiffnessMatrix);
                        ASSEMBLELIST(fineStiffnessMatrix, world.getForceList(), getStiffnessMatrix);
                        ASSEMBLEEND(fineStiffnessMatrix);
                        
                        
                        //constraint Projection
                        (*fineStiffnessMatrix) = m_fineP*(*fineStiffnessMatrix)*m_fineP.transpose();
                        
                        
                        m_Us = generalizedEigenvalueProblemNotNormalized(((*fineStiffnessMatrix)), (*m_fineMassMatrix), m_numModes, 0.00);
                        Eigen::VectorXd normalizing_const;
                        normalizing_const = (m_Us.first.transpose() * (*m_fineMassMatrix) * m_Us.first).diagonal();
                        normalizing_const = normalizing_const.cwiseSqrt().cwiseInverse();
                        
                        m_Us.first = m_Us.first * (normalizing_const.asDiagonal());
                        
                        Eigen::saveMarketVector(m_Us.second, "finemesheigenvalues" + std::to_string(step_number) + ".mtx");
                        
                        fineEigMassProj = m_Us;
                        fineEig = m_Us;
                        fineEigMassProj.first = (*m_fineMassMatrix)*fineEigMassProj.first;
                        
                        for(int i = 0; i < m_numModes; ++i)
                        {
                            m_R(i) = m_Us.second(i)/m_coarseUs.second(i);
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
                                m_R(0) = 1.0;
                                m_R(1) = 1.0;
                                m_R(2) = 1.0;
                                m_R(3) = 1.0;
                                m_R(4) = 1.0;
                                m_R(5) = 1.0;
                                
                            }
                            //#ifdef EDWIN_DEBUG
                            std::cout<<m_R(i)<<std::endl;
                            //#endif
                            
                        }
                        
                        ratio_calculated = true;
                    }
                    else{
                        cout<<"can't load eigenvalues from fine mesh"<<endl;
                        exit(1);
                    }
                }
                
                
            }
            
            //            std::cout<<m_coarseUs.second<<std::endl;
            
        }
        
        cout<<"using ratio: "<<endl;
        cout<<m_R<<endl;
        Y = (*coarseMassMatrix)*m_coarseUs.first*(m_R-m_I).asDiagonal();
        Z =  (m_coarseUs.second.asDiagonal()*m_coarseUs.first.transpose()*(*coarseMassMatrix));
        //        Eigen::saveMarket(Y, "Y.dat");
        //        Eigen::saveMarket(Z, "Z.dat");
        
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
        //            std::cout<<m_Vf.rows()<<std::endl;
        //            std::cout<<m_Vf.cols()<<std::endl;
        //            std::cout<<V.rows()<<std::endl;
        //            std::cout<<V.cols()<<std::endl;
        //            std::cout<<((*N)*V).rows()<<std::endl;
        //            std::cout<<((*N)*V).cols()<<std::endl;
        Eigen::Map<Eigen::VectorXd> eigen_fine_pos0(fine_pos0,m_fineWorld.getNumQDOFs());
        
        return eigen_fine_pos0 + (*N)*V;
    }
    
    
    inline World<double, std::tuple<PhysicalSystemImpl *>,
    std::tuple<ForceSpringFEMParticle<double> *, ForceParticlesGravity<double> *>,
    std::tuple<ConstraintFixedPoint<double> *> > & getFineWorld(){ return m_fineWorld;}
    
    
    std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > coarseEigMassProj;
    std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > fineEigMassProj;
    std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > coarseEig;
    std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > fineEig;
    std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > m_Us;
    std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > m_fineUs_restart;
    std::pair<Eigen::MatrixXx<double>, Eigen::VectorXx<double> > m_coarseUs_restart;
    Eigen::VectorXx<double> coarseEigenvalues;
    Eigen::VectorXx<double> fineEigenvalues;
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

protected:
    
    //
    //    World<double, std::tuple<PhysicalSystemImpl *>,
    //    std::tuple<ForceSpringFEMParticle<double> *, ForceParticlesGravity<double> *>,
    //    std::tuple<ConstraintFixedPoint<double> *> > m_fineWorld_reset;
    //
    //    World<double, std::tuple<PhysicalSystemImpl *>,
    //    std::tuple<ForceSpringFEMParticle<double> *, ForceParticlesGravity<double> *>,
    //    std::tuple<ConstraintFixedPoint<double> *> > m_coarseWorld_reset;
    //
    
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
    
    int m_constraintDir;
    double m_constraintTol;
    
    //num modes to correct
    unsigned int m_numModes;
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
    
    
    unsigned int constraint_switch;
    unsigned int m_numConstraints;
private:
    
};

#endif /* EigenFit_h */

