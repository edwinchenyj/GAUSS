//
//  ConstraintFixedPoint.h
//  Gauss
//
//  Created by David Levin on 4/20/17.
//
//

#ifndef ConstraintFixedPoint_h
#define ConstraintFixedPoint_h

#include <array>
#include <Constraint.h>
#include <DOFParticle.h>
#include <UtilitiesEigen.h>
#include <iterator>

namespace Gauss {
    template<typename DataType>
    class ConstraintFixedPointImpl
    {
    public:
        
        ConstraintFixedPointImpl(DOFParticle<DataType> *q0, Eigen::Vector3x<DataType> x, Eigen::Vector3x<DataType> v = Eigen::Vector3x<DataType>::Zero()) {
            m_dofFixed = q0;
            m_p0 = x;
            m_dpdt = v;
        }
        
        // position constraint
        void setFixedPoint(Eigen::VectorXx<DataType> pos) {
            m_p0 = pos;
        }
        
        // overload velocity constraint
        void setFixedPoint(Eigen::VectorXx<DataType> pos, Eigen::VectorXx<DataType> vel) {
            m_p0 = pos;
            m_dpdt = vel;
        }
        
        auto & getFixedPoint() {
            return m_p0;
        }
        ~ConstraintFixedPointImpl() { }
        
        //get
        constexpr unsigned int getNumRows() { return 3; }
        //value of constraint (supports vector valued constraint functions for points and what not)
        
        template<typename Vector>
        inline void getError(Vector &f,  const State<DataType> &state, const ConstraintIndex &index) {
            
            Eigen::Vector3d func = m_p0 - mapDOFEigen(*m_dofFixed, state);
            assign(f, func, std::array<ConstraintIndex,1>{{index}});
        }
        
        template<typename Vector>
        inline void getB(Vector &f,  const State<DataType> &state, const ConstraintIndex &index) {
            
            Eigen::Vector3d func = m_p0;
            assign(f, func, std::array<ConstraintIndex,1>{{index}});
        }
        
        template<typename Vector>
        inline void getDbDt(Vector &f,  const State<DataType> &state, const ConstraintIndex &index) {
            
            Eigen::Vector3d func = m_dpdt;
            assign(f, func, std::array<ConstraintIndex,1>{{index}});
        }
        
        //get DOFs that this constraint is acting on
        auto & getDOF(unsigned int index) {
            return *m_dofFixed;
        }
        
        //how many DOFs are involved in this constraint
        constexpr unsigned int getNumDOF() const { return 1; }
        
        template <typename World, typename Matrix, unsigned int Operation>
        inline void getGradient(Matrix &g,  const World &world, const State<DataType> &state, const ConstraintIndex &index) {
            Eigen::Matrix3d I;
            I.setIdentity();
            
            assign<Matrix, Eigen::Matrix<double,3,3>, std::array<ConstraintIndex,1>, std::array<DOFBase<DataType,0>, 1>, Operation>(g, I, std::array<ConstraintIndex,1>{{index}}, std::array<DOFBase<DataType,0>, 1>{{*m_dofFixed}});
        }
        
        
    protected:
        
        Eigen::VectorXx<DataType> m_dpdt;
        Eigen::VectorXx<DataType> m_p0; //position to fix point at
        DOFParticle<DataType> *m_dofFixed; //pointer to the thing I'm fixing in space
        
    private:
    };
    
    template<typename DataType>
    using ConstraintFixedPoint = Constraint<DataType, ConstraintFixedPointImpl<DataType> >;
    
    template<typename World, typename FEMSystem>
    Eigen::Vector3d getMinXYZ(World &world, FEMSystem *system)
    {
        return system->getImpl().getV().colwise().minCoeff();
    }

    template<typename World, typename FEMSystem>
    Eigen::Vector3d getMaxXYZ(World &world, FEMSystem *system)
    {
        return system->getImpl().getV().colwise().maxCoeff();
    }

    //Utility functions to fix a bunch of points
    template<typename World, typename FEMSystem>
    void fixDisplacementMin(World &world, FEMSystem *system, unsigned int dim = 0, double tolerance=1e-5) {
        //find all vertices with minimum x coordinate and fix DOF associated with them
        auto minX = system->getImpl().getV().col(dim).minCoeff();
        std::vector<unsigned int> minV;

	   for(unsigned int ii=0; ii<system->getImpl().getV().rows(); ++ii) {
            if(fabs(system->getImpl().getV()(ii,dim) - minX) < tolerance) {
                minV.push_back(ii);
            }
        }
        
        //add a bunch of constraints
        for(auto iV : minV) {
            // set constraint with 0 velocity
            world.addConstraint(new ConstraintFixedPoint<decltype(minX)>(&system->getQ()[iV], Eigen::Vector3x<decltype(minX)>(0,0,0)));
        }
    }

    //Utility functions to fix a bunch of points
    template<typename World, typename FEMSystem>
    void fixDisplacementMax(World &world, FEMSystem *system, unsigned int dim = 0, double tolerance=1e-5) {
        //find all vertices with minimum x coordinate and fix DOF associated with them
        auto maxX = system->getImpl().getV().col(dim).maxCoeff();
        std::vector<unsigned int> maxV;
        for(unsigned int ii=0; ii<system->getImpl().getV().rows(); ++ii) {
            
            if(fabs(system->getImpl().getV()(ii,dim) - maxX) < tolerance) {
                maxV.push_back(ii);
            }
        }
        
        //add a bunch of constraints
        for(auto iV : maxV) {
            world.addConstraint(new ConstraintFixedPoint<decltype(maxX)>(&system->getQ()[iV], Eigen::Vector3d(0,0,0)));
        }
    }

//Utility functions to fix a bunch of points
    template<typename World, typename FEMSystem>
    void fixDisplacementBetween(World &world, FEMSystem *system, unsigned int dim = 0, double x1=0, double x2=0) {
        //find all vertices with minimum x coordinate and fix DOF associated with them
        auto minX = system->getImpl().getV()(0,dim);
	std::vector<unsigned int> minV;
        
        for(unsigned int ii=0; ii<system->getImpl().getV().rows(); ++ii) {
            
            if(system->getImpl().getV()(ii,dim) > x1 && system->getImpl().getV()(ii,dim) < x2) {
                minV.push_back(ii);
            }
        }

        
        //add a bunch of constraints
        for(auto iV : minV) {
            world.addConstraint(new ConstraintFixedPoint<decltype(minX)>(&system->getQ()[iV], Eigen::Vector3d(0,0,0)));
        }
    }
    //Utility functions to fix a bunch of points
    template<typename FEMSystem>
    Eigen::VectorXi maxVertices(FEMSystem *system, unsigned int dim = 0, double tolerance = 1e-5) {
        
        
        //find all vertices with minimum x coordinate and fix DOF associated with them
        auto maxX = system->getImpl().getV()(0,dim);
        std::vector<unsigned int> maxV;
        
        for(unsigned int ii=0; ii<system->getImpl().getV().rows(); ++ii) {
            
            if(system->getImpl().getV()(ii,dim) > maxX) {
                maxX = system->getImpl().getV()(ii,dim);
                maxV.clear();
                maxV.push_back(ii);
            } else if(fabs(system->getImpl().getV()(ii,dim) - maxX) < tolerance) {
                maxV.push_back(ii);
            }
        }
        
        Eigen::VectorXi indices(maxV.size());
        
        //add a bunch of constraints
        for(unsigned int iV = 0; iV < maxV.size(); ++iV) {
            indices(iV) = maxV[iV];
        }
        
        return indices;
    }
    
    //Utility functions to fix a bunch of points
    template<typename FEMSystem>
    Eigen::VectorXi minVertices(FEMSystem *system, unsigned int dim = 0, double tolerance = 1e-5) {
        
        
        //find all vertices with minimum x coordinate and fix DOF associated with them
        auto minX = system->getImpl().getV()(0,dim);
        std::vector<unsigned int> minV;
        
        for(unsigned int ii=0; ii<system->getImpl().getV().rows(); ++ii) {
            
            if(system->getImpl().getV()(ii,dim) < minX) {
                minX = system->getImpl().getV()(ii,dim);
                minV.clear();
                minV.push_back(ii);
            } else if(fabs(system->getImpl().getV()(ii,dim) - minX) < tolerance) {
                minV.push_back(ii);
            }
        }
        
        Eigen::VectorXi indices(minV.size());
        
        //add a bunch of constraints
        for(unsigned int iV = 0; iV < minV.size(); ++iV) {
            indices(iV) = minV[iV];
        }
        
        return indices;
    }


    template<typename World, typename FEMSystem, typename DataType = double>
    Eigen::SparseMatrix<DataType> fixedPointProjectionMatrix(Eigen::VectorXi &indices, FEMSystem &system,World &world) {
        
        std::vector<Eigen::Triplet<DataType> > triplets;
        Eigen::SparseMatrix<DataType> P;
        Eigen::VectorXi sortedIndices = indices;
        std::sort(sortedIndices.data(), sortedIndices.data()+indices.rows());
        
        //build a projection matrix P which projects fixed points out of a physical syste
        int fIndex = 0;
        
        //total number of DOFS in system
        
        unsigned int n = mapStateEigen<0>(world).rows();
        unsigned int m = mapStateEigen<0>(world).rows() - 3*indices.rows();
        
        P.resize(m,n);
        
        //add ones up to global index of system
        for(unsigned int ii=0; ii<system.getQ().getGlobalId(); ++ii) {
            triplets.push_back(Eigen::Triplet<DataType>(ii, ii, 1));
        }
        
        //number of unconstrained DOFs
        unsigned int rowIndex =0;
        for(unsigned int vIndex = 0; vIndex < system.getImpl().getV().rows(); vIndex++) {
            
            while((vIndex < system.getImpl().getV().rows()) && (fIndex < sortedIndices.rows()) &&(vIndex == sortedIndices[fIndex])) {
                fIndex++;
                vIndex++;
            }
            
            if(vIndex == system.getImpl().getV().rows())
                break;
            
            //add triplet into matrix
            triplets.push_back(Eigen::Triplet<DataType>(system.getQ().getGlobalId() +rowIndex, system.getQ().getGlobalId() + 3*vIndex,1));
            triplets.push_back(Eigen::Triplet<DataType>(system.getQ().getGlobalId() +rowIndex+1, system.getQ().getGlobalId() + 3*vIndex+1, 1));
            triplets.push_back(Eigen::Triplet<DataType>(system.getQ().getGlobalId() +rowIndex+2, system.getQ().getGlobalId() + 3*vIndex+2, 1));
            
            rowIndex+=3;
        }
        
        //add ones on the diagonal after this system
        unsigned int ii, jj;
        for(ii=system.getQ().getGlobalId()+rowIndex, jj=system.getQ().getGlobalId()+system.getQ().getNumScalarDOF(); jj<world.getNumQDOFs(); ++ii, ++jj) {
            triplets.push_back(Eigen::Triplet<DataType>(ii, jj, 1));
        }
        
        P.setFromTriplets(triplets.begin(), triplets.end());
        
        //build the matrix and  return
        return P;
    }

    
}

#endif /* ConstraintFixedPoint_h */
