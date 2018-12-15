//
//  resultsUtilities.h
//  Gauss
//
//  Created by Yu Ju Edwin Chen on 2018-12-14.
//

#ifndef resultsUtilities_h
#define resultsUtilities_h

#include <string>
#include <GaussIncludes.h>
#include <FEMIncludes.h>
#include <Eigen/Core>

void filename_number_padded(std::string& filename, int file_ind, std::string& extension, int num_length = 5)
{
    filename = filename + std::string(num_length - std::to_string(file_ind).length(),'0') + std::to_string(file_ind) + "." + extension;
}
//
//std::string pos_filename_padded(int file_ind)
//{
//    std::string filename = "pos";
//    filename_number_padded(filename, file_ind, "obj");
//    return filename;
//}
//
//
//std::string surfpos_filename_padded(int file_ind)
//{
//    std::string filename = "surfpos";
//    filename_number_padded(filename, file_ind, "obj");
//    return filename;
//}


//template <typename Vector>
void q_state_to_position(Eigen::VectorXd& q, Eigen::MatrixXd& V_pos)
{
    unsigned int idxc = 0;
    
    for(unsigned int vertexId=0;  vertexId < V_pos.rows(); ++vertexId) {
        
        V_pos(vertexId,0) += q(idxc);
        idxc++;
        V_pos(vertexId,1) += q(idxc);
        idxc++;
        V_pos(vertexId,2) += q(idxc);
        idxc++;
    }
}

#endif /* resultsUtilities_h */
