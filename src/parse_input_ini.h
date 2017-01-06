/*
 * parse_input_ini.h
 *
 *  Created on: Nov 13, 2016
 *      Author: cg
 */

#ifndef PARSE_INPUT_INI_H_
#define PARSE_INPUT_INI_H_

#include "constant.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

bool parse_input_ini(std::string ini_filename,
		int & nx, int & ny, int & nz,
		double & dx,double & dy,double & dz,

		FLOAT_ &  Bextx,FLOAT_ &  Bexty,FLOAT_ &  Bextz,

		int & pbc_x, int & pbc_y, int & pbc_z,

		FLOAT_ &  ms, FLOAT_ &  Aex,FLOAT_ &  Dind,

		FLOAT_ &  anisUx, FLOAT_ &  anisUy,FLOAT_ &  anisUz,FLOAT_ &  Ku1,

		bool &  use_random_init,
		long long &  randomseed,
		std::string &  ovf_filename,

		FLOAT_ &  Temperature_start, FLOAT_  & Temperature_end,
		FLOAT_  & Temperature_step,

		long  & circle_per_stage ,
		long  & terminal_output_period,
		long  & energy_output_period,
		long  & ms_output_period,

		int  & rand_block_size	,
		bool & cal_demag_flag,
		bool & Temperature_use_exp)
{



	return true;
}

void cout_input_ini(
		int  nx, int  ny, int  nz,
		double  dx,double  dy,double  dz,

		FLOAT_   Bextx,FLOAT_   Bexty,FLOAT_   Bextz,

		int  pbc_x, int  pbc_y, int  pbc_z,

		FLOAT_   ms, FLOAT_   Aex,FLOAT_   Dind,

		FLOAT_   anisUx, FLOAT_   anisUy,FLOAT_   anisUz,FLOAT_   Ku1,

		bool   use_random_init,
		long long   randomseed,
		std::string   ovf_filename,

		FLOAT_   Temperature_start, FLOAT_   Temperature_end,
		FLOAT_   Temperature_step,

		long   circle_per_stage ,
		long   terminal_output_period,
		long   energy_output_period,
		long   ms_output_period,

		int   rand_block_size,
		bool cal_demag_flag,
		bool Temperature_use_exp)
{
	using namespace std;
	cout<<"====================display ini file============\n";
	cout<<"nx"<<": "<< nx  <<endl;
	cout<<"ny"<<": "<<  ny <<endl;
	cout<<"nz"<<": "<< nz  <<endl;
	cout<<"dx"<<": "<< dx  <<endl;
	cout<<"dy"<<": "<< dy  <<endl;
	cout<<"dz"<<": "<< dz  <<endl;
	cout<<"Bextx"<<": "<< Bextx  <<endl;
	cout<<"Bexty"<<": "<< Bexty  <<endl;
	cout<<"Bextz"<<": "<<Bextz   <<endl;
	cout<<"pbc_x"<<": "<<  pbc_x <<endl;
	cout<<"pbc_y"<<": "<< pbc_y  <<endl;
	cout<<"pbc_z"<<": "<<  pbc_z <<endl;
	cout<<"ms"<<": "<< ms  <<endl;
	cout<<"Aex"<<": "<<  Aex <<endl;
	cout<<"Dind"<<": "<< Dind  <<endl;
	cout<<"anisUx"<<": "<< anisUx  <<endl;
	cout<<"anisUy"<<": "<< anisUy  <<endl;
	cout<<"anisUz"<<": "<< anisUz  <<endl;
	cout<<"Ku1"<<": "<< Ku1  <<endl;
	cout<<"use_random_init"<<": "<< use_random_init  <<endl;
	cout<<"randomseed"<<": "<< randomseed  <<endl;
	cout<<"ovf_filename"<<": "<<  ovf_filename <<endl;
	cout<<"Temperature_start"<<": "<< Temperature_start  <<endl;
	cout<<"Temperature_end"<<": "<< Temperature_end  <<endl;
	cout<<"Temperature_step"<<": "<<  Temperature_step <<endl;
	cout<<"Temperature_use_exp"<<": "<<  Temperature_use_exp <<endl;

	cout<<"circle_per_stage"<<": "<< circle_per_stage  <<endl;
	cout<<"terminal_output_period"<<": "<< terminal_output_period  <<endl;
	cout<<"energy_output_period"<<": "<<  energy_output_period <<endl;
	cout<<"ms_output_period"<<": "<< ms_output_period  <<endl;
	cout<<"rand_block_size"<<": "<< rand_block_size  <<endl;
	cout<<"cal_demag_flag"<<": "<< cal_demag_flag  <<endl;
	cout<<"---------------------END display ini file------------\n";

}
#endif /* PARSE_INPUT_INI_H_ */
