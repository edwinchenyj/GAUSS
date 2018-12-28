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

using namespace std;

std::string filename_number_padded(std::string filename, int file_ind, std::string extension, int num_length = 5)
{
    filename = filename + std::string(num_length - std::to_string(file_ind).length(),'0') + std::to_string(file_ind) + "." + extension;
    return filename;
}

//template <typename Vector>
void parse_input(int argc, char **argv, std::string &cmeshname,
                 std::string &fmeshname, double &youngs, double &const_tol,
                 int &const_profile, std::string &initial_def, int &num_steps, bool &haus,
                 int &num_modes, int &const_dir, double &step_size, int &dynamic_flag,
                 double &a, double &b, bool &output_data_flag, bool &simple_mass_flag, double &mode_matching_tol, int & calculate_matching_data_flag, double & init_mode_matching_tol, bool & init_eigenvalue_criteria, int & init_eigenvalue_criteria_factor, std::string & integrator, bool & eigenfit_damping)
{
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        arg.erase(remove_if(arg.begin(), arg.end(), ::isspace), arg.end());
        std::size_t eq_found = arg.find_first_of("=");
        std::string field(arg.substr(1,eq_found-1));
        
        if (field.compare("cmeshname") == 0) {
            cmeshname =arg.substr(eq_found+1,arg.length()-eq_found-1);
            cout<<"Using coarse mesh: "<<cmeshname<<endl;
            
        }
        else if(field.compare("fmeshname") == 0) {
            fmeshname =arg.substr(eq_found+1,arg.length()-eq_found-1);
            cout<<"Using fine mesh: "<<fmeshname<<endl;
        }
        else if(field.compare("youngs") == 0)
        {
            youngs = stod(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using Youngs: "<<youngs<<endl;
        }
        else if(field.compare("const_tol") == 0)
        {
            const_tol =stod(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using constraint tolerance: "<<const_tol<<endl;
        }
        else if(field.compare("const_profile") == 0)
        {
            const_profile =stoi(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using constriant profile: "<<const_profile<<endl;
        }
        else if(field.compare("initial_def") == 0)
        {
            initial_def = arg.substr(eq_found+1,arg.length()-eq_found-1);
            cout<<"Using initial deformation: "<<initial_def<<endl;
        }
        else if(field.compare("num_steps") == 0)
        {
            num_steps = stoi(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using number of steps: "<< num_steps<<endl;
        }
        else if(field.compare("haus") == 0)
        {
            haus = stoi(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using haus: "<<haus<<endl;
        }
        else if(field.compare("num_modes") == 0)
        {
            num_modes = stoi(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using number of modes: "<<num_modes<<endl;
        }
        else if(field.compare("const_dir") == 0)
        {
            const_dir = stoi(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using constraint direction: "<<const_dir<<endl;
        }
        else if(field.compare("step_size") == 0)
        {
            step_size = stod(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using step size: "<<step_size<<endl;
        }
        else if(field.compare("dynamic_flag") == 0)
        {
            dynamic_flag = stoi(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using dynamic_flag: "<<dynamic_flag<<endl;
        }
        else if(field.compare("a") == 0)
        {
            a = stod(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using a: "<<a<<endl;
        }
        else if(field.compare("b") == 0)
        {
            b = stod(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using b: "<<b<<endl;
        }
        else if(field.compare("output_data_flag") == 0)
        {
            output_data_flag = stoi(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using output data flag: "<<output_data_flag<<endl;
        }
        else if(field.compare("simple_mass_flag") == 0)
        {
            simple_mass_flag = stoi(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using simple mass flag: "<<simple_mass_flag<<endl;
        }
        else if(field.compare("mode_matching_tol") == 0)
        {
            mode_matching_tol = stod(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using mode matching tol: "<<mode_matching_tol<<endl;
        }
        else if(field.compare("calculate_matching_data_flag") == 0)
        {
            calculate_matching_data_flag = stoi(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using calculate matching data flag: "<<calculate_matching_data_flag<<endl;
        }
        else if(field.compare("init_mode_matching_tol") == 0)
        {
            init_mode_matching_tol = stod(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using init mode matching tol: "<<init_mode_matching_tol<<endl;
        }
        else if(field.compare("init_eigenvalue_criteria") == 0)
        {
            init_eigenvalue_criteria = stoi(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using init eigenvalue criteria: "<<init_eigenvalue_criteria<<endl;
        }
        else if(field.compare("init_eigenvalue_criteria_factor") == 0)
        {
            init_eigenvalue_criteria_factor = stoi(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using init eigenvalue criteria factor: "<<init_eigenvalue_criteria_factor<<endl;
        }
        else if(field.compare("integrator") == 0) {
            integrator =arg.substr(eq_found+1,arg.length()-eq_found-1);
            cout<<"Using integrator: "<<integrator<<endl;
        }
        else if(field.compare("eigenfit_damping") == 0)
        {
            eigenfit_damping = stoi(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using eigenfit damping: "<<eigenfit_damping<<endl;
        }
        else
        {
            cout<<"Warning: Unknown field "<< field<<" with unused value " << arg.substr(eq_found+1,arg.length()-eq_found-1)<<endl;
        }
        
    }
    
    
}

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
//
//template<typename DataType>
void apply_moving_constraint(int const_profile, State<double> & state, std::vector<ConstraintFixedPoint<double> *> & movingConstraints, int frame_number)
{
    
    // acts like the "callback" block for moving constraint
    if (const_profile == 2)
    {
        // constraint profile 2 will move some vertices
        //script some motion
        cout<<"Moving constrained vertices in y..."<<endl;
        for(unsigned int jj=0; jj<movingConstraints.size(); ++jj) {
            
            auto v_q = mapDOFEigen(movingConstraints[jj]->getDOF(0), state);
            Eigen::Vector3d new_q = (frame_number)*Eigen::Vector3d(0.0,-1.0/100,0.0);
            v_q = new_q;
            
        }
    }
    else if (const_profile == 4 )
    {
        cout<<"Moving constrained vertices in x..."<<endl;
        for(unsigned int jj=0; jj<movingConstraints.size(); ++jj) {
            
            auto v_q = mapDOFEigen(movingConstraints[jj]->getDOF(0), state);
            //
            if ((frame_number) < 50) {
                Eigen::Vector3d new_q = (frame_number)*Eigen::Vector3d(-1.0/100,0.0,0.0);
                v_q = new_q;
            }
            
        }
    }
    else if (const_profile == 5)
    {
        cout<<"Moving constraint vertices in y..."<<endl;
        for(unsigned int jj=0; jj<movingConstraints.size(); ++jj) {
            
            auto v_q = mapDOFEigen(movingConstraints[jj]->getDOF(0), state);
            //
            if ((frame_number) < 50) {
                Eigen::Vector3d new_q = (frame_number)*Eigen::Vector3d(0.0,-1.0/100,0.0);
                v_q = new_q;
            }
            
        }
    }else if (const_profile == 6)
    {
        cout<<"Moving constrained vertices in z..."<<endl;
        for(unsigned int jj=0; jj<movingConstraints.size(); ++jj) {
            
            auto v_q = mapDOFEigen(movingConstraints[jj]->getDOF(0), state);
            //
            if ((frame_number) < 50) {
                Eigen::Vector3d new_q = (frame_number)*Eigen::Vector3d(0.0,0.0,-1.0/100);
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
            
            auto v_q = mapDOFEigen(movingConstraints[jj]->getDOF(0), state);
            //
            if ((frame_number) < 250) {
                //                        Eigen::Vector3d new_q = (frame_number)*Eigen::Vector3d(0.0,0.0,-1.0/100);
                v_q(0) += 0.1*Xvel(frame_number);
                v_q(1) += 0.1*Yvel(frame_number);
                v_q(2) += 0.1*Zvel(frame_number);
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
            auto v_q = mapDOFEigen(movingConstraints[jj]->getDOF(0), state);
            if ((frame_number) < 250) {
                if(Xvel(frame_number) <= 0){   v_q(0) += 0.5*std::max(Xvel(frame_number),-0.005);}
                else{ v_q(0) += 0.5*std::min(Xvel(frame_number),0.005);}
                if(Yvel(frame_number) <= 0){   v_q(1) += 0.5*std::max(Yvel(frame_number),-0.005);}
                else{ v_q(1) += 0.5*std::min(Yvel(frame_number),0.005);}
                if(Zvel(frame_number) <= 0){   v_q(2) += 0.5*std::max(Zvel(frame_number),-0.005);}
                else{ v_q(2) += 0.5*std::min(Zvel(frame_number),0.005);}
            }
        }
    }
    else if (const_profile == 9)
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
            auto v_q = mapDOFEigen(movingConstraints[jj]->getDOF(0), state);
            if ((frame_number) < 100) {
                if(Xvel(frame_number) <= 0){   v_q(0) += 2.5*std::max(Xvel(frame_number),-0.005);}
                else{ v_q(0) += 2.5*std::min(Xvel(frame_number),0.005);}
                if(Yvel(frame_number) <= 0){   v_q(1) += 2.5*std::max(Yvel(frame_number),-0.005);}
                else{ v_q(1) += 2.5*std::min(Yvel(frame_number),0.005);}
                if(Zvel(frame_number) <= 0){   v_q(2) += 2.5*std::max(Zvel(frame_number),-0.005);}
                else{ v_q(2) += 2.5*std::min(Zvel(frame_number),0.005);}
            }
        }
    }
    else if (const_profile == 10)
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
            auto v_q = mapDOFEigen(movingConstraints[jj]->getDOF(0), state);
            if ((frame_number) < 100) {
                if(Xvel(frame_number) <= 0){   v_q(0) += 4.5*std::max(Xvel(frame_number),-0.005);}
                else{ v_q(0) += 4.5*std::min(Xvel(frame_number),0.005);}
                if(Yvel(frame_number) <= 0){   v_q(1) += 4.5*std::max(Yvel(frame_number),-0.005);}
                else{ v_q(1) += 4.5*std::min(Yvel(frame_number),0.005);}
                if(Zvel(frame_number) <= 0){   v_q(2) += 4.5*std::max(Zvel(frame_number),-0.005);}
                else{ v_q(2) += 4.5*std::min(Zvel(frame_number),0.005);}
            }
        }
    }
    else if (const_profile == 11)
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
            auto v_q = mapDOFEigen(movingConstraints[jj]->getDOF(0), state);
            if ((frame_number) < 100) {
                if(Xvel(frame_number) <= 0){   v_q(0) += 1.5*std::max(Xvel(frame_number),-0.005);}
                else{ v_q(0) += 1.5*std::min(Xvel(frame_number),0.005);}
                if(Yvel(frame_number) <= 0){   v_q(1) += 1.5*std::max(Yvel(frame_number),-0.005);}
                else{ v_q(1) += 1.5*std::min(Yvel(frame_number),0.005);}
                if(Zvel(frame_number) <= 0){   v_q(2) += 1.5*std::max(Zvel(frame_number),-0.005);}
                else{ v_q(2) += 1.5*std::min(Zvel(frame_number),0.005);}
            }
        }
    }
    else if (const_profile == 12)
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
            auto v_q = mapDOFEigen(movingConstraints[jj]->getDOF(0), state);
            if ((frame_number) < 30) {
                if(Xvel(frame_number) <= 0){   v_q(0) += 2.5*std::max(Xvel(frame_number),-0.005);}
                else{ v_q(0) += 2.5*std::min(Xvel(frame_number),0.005);}
                if(Yvel(frame_number) <= 0){   v_q(1) += 2.5*std::max(Yvel(frame_number),-0.005);}
                else{ v_q(1) += 2.5*std::min(Yvel(frame_number),0.005);}
                if(Zvel(frame_number) <= 0){   v_q(2) += 2.5*std::max(Zvel(frame_number),-0.005);}
                else{ v_q(2) += 2.5*std::min(Zvel(frame_number),0.005);}
            }
        }
    }
    else if (const_profile == 13)
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
            auto v_q = mapDOFEigen(movingConstraints[jj]->getDOF(0), state);
            if ((frame_number) < 30) {
                if(Xvel(frame_number) <= 0){   v_q(0) += 4.5*std::max(Xvel(frame_number),-0.005);}
                else{ v_q(0) += 4.5*std::min(Xvel(frame_number),0.005);}
                if(Yvel(frame_number) <= 0){   v_q(1) += 4.5*std::max(Yvel(frame_number),-0.005);}
                else{ v_q(1) += 4.5*std::min(Yvel(frame_number),0.005);}
                if(Zvel(frame_number) <= 0){   v_q(2) += 4.5*std::max(Zvel(frame_number),-0.005);}
                else{ v_q(2) += 4.5*std::min(Zvel(frame_number),0.005);}
            }
        }
    }
    else if (const_profile == 14)
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
            auto v_q = mapDOFEigen(movingConstraints[jj]->getDOF(0), state);
            if ((frame_number) < 30) {
                if(Xvel(frame_number) <= 0){   v_q(0) += 1.5*std::max(Xvel(frame_number),-0.005);}
                else{ v_q(0) += 1.5*std::min(Xvel(frame_number),0.005);}
                if(Yvel(frame_number) <= 0){   v_q(1) += 1.5*std::max(Yvel(frame_number),-0.005);}
                else{ v_q(1) += 1.5*std::min(Yvel(frame_number),0.005);}
                if(Zvel(frame_number) <= 0){   v_q(2) += 1.5*std::max(Zvel(frame_number),-0.005);}
                else{ v_q(2) += 1.5*std::min(Zvel(frame_number),0.005);}
            }
        }
    }
}

void parse_input(int argc, char **argv, std::string &meshname, double &youngs, double &const_tol,
                 int &const_profile, std::string &initial_def, int &num_steps,
                 int &num_modes, int &const_dir, double &step_size,
                 double &a, double &b)
{
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        arg.erase(remove_if(arg.begin(), arg.end(), ::isspace), arg.end());
        std::size_t eq_found = arg.find_first_of("=");
        std::string field(arg.substr(1,eq_found-1));
        
        if (field.compare("meshname") == 0) {
            meshname =arg.substr(eq_found+1,arg.length()-eq_found-1);
            cout<<"Using mesh: "<<meshname<<endl;
            
        }
        else if(field.compare("youngs") == 0)
        {
            youngs = stod(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using Youngs: "<<youngs<<endl;
        }
        else if(field.compare("const_tol") == 0)
        {
            const_tol =stod(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using constraint tolerance: "<<const_tol<<endl;
        }
        else if(field.compare("const_profile") == 0)
        {
            const_profile =stoi(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using constriant profile: "<<const_profile<<endl;
        }
        else if(field.compare("initial_def") == 0)
        {
            initial_def = arg.substr(eq_found+1,arg.length()-eq_found-1);
            cout<<"Using initial deformation: "<<initial_def<<endl;
        }
        else if(field.compare("num_steps") == 0)
        {
            num_steps = stoi(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using number of steps: "<< num_steps<<endl;
        }
        else if(field.compare("num_modes") == 0)
        {
            num_modes = stoi(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using number of modes: "<<num_modes<<endl;
        }
        else if(field.compare("const_dir") == 0)
        {
            const_dir = stoi(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using constraint direction: "<<const_dir<<endl;
        }
        else if(field.compare("step_size") == 0)
        {
            step_size = stod(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using step size: "<<step_size<<endl;
        }
        else if(field.compare("a") == 0)
        {
            a = stod(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using a: "<<a<<endl;
        }
        else if(field.compare("b") == 0)
        {
            b = stod(arg.substr(eq_found+1,arg.length()-eq_found-1));
            cout<<"Using b: "<<b<<endl;
        }
        else
        {
            cout<<"Warning: Unknown field "<< field<<" with unused value " << arg.substr(eq_found+1,arg.length()-eq_found-1)<<endl;
        }
        
    }
    
    
}

#endif /* resultsUtilities_h */
