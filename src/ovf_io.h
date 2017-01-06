/*
 * ovf_io.h
template<typename T> bool parseOvfFile(std::string filename, OvfInfo ovfinfo_in,
		T * mx, T * my, T * mz);
template<typename T> bool writetoOvfFile(std::string filename, OvfInfo info,
		T * mx, T * my, T * mz);
template<typename T> void print_mat(string s, T *mx, int nx, int ny, int nz);
 *  Created on: Nov 11, 2016
 *      Author: cg
 */

#ifndef OVF_IO_H_
#define OVF_IO_H_

#include "constant.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <cctype>

#define  DX_ERROR_RATION 1e-5
#define IS_DIFFERENCE_SMALL_ENOUGH(A,B)   (abs((A)-(B))<= DX_ERROR_RATION *   ( abs(A) + abs(B)   )   )
// https://github.com/mumax/3/blob/master/oommf/oommf.go




namespace ovf_io {
using namespace std;
struct OvfInfo {
	int xnodes = -1;
	int ynodes = -1;
	int znodes = -1;
	double xstepsize = -1;
	double ystepsize = -1;
	double zstepsize = -1;
	bool operator ==(OvfInfo b) {
		if (this->xnodes == b.xnodes&&
		this->ynodes == b.ynodes &&
		this->znodes == b.znodes &&
		IS_DIFFERENCE_SMALL_ENOUGH(this->xstepsize,b.xstepsize) &&
		IS_DIFFERENCE_SMALL_ENOUGH(this->ystepsize,b.ystepsize) &&
		IS_DIFFERENCE_SMALL_ENOUGH(this->zstepsize,b.zstepsize)
		)
			return true;
		else
			return false;
	}

	void disp(string s = "") {
		cout << s << endl;
		cout << "xnodes: " << this->xnodes << endl;
		cout << "ynodes: " << this->ynodes << endl;
		cout << "znodes: " << this->znodes << endl;
		cout << "xstepsize: " << this->xstepsize << endl;
		cout << "ystepsize: " << this->ystepsize << endl;
		cout << "zstepsize: " << this->zstepsize << endl;
	}
};

std::pair<string, string> parseHeaderLine(string str)
{
    using namespace boost;

    vector< string > SplitVec;
    split( SplitVec, str, is_any_of(":") );
    if (SplitVec.size()<=1)
        return make_pair("","");
    string key   = SplitVec[0];

    split( SplitVec, SplitVec[1], is_any_of("#") );
    string value = SplitVec[0];

    trim_left_if(key , is_any_of("#"));
    trim(key);
    to_lower(key);
    trim(value);
    to_lower(value);
    return make_pair( key,value );
}
    ;

template<typename T>  bool parseOvfFile(std::string filename,OvfInfo ovfinfo_in,
                  T * mx, T * my, T * mz)
{

}

template<typename T>  bool writetoOvfFile(std::string filename,OvfInfo info,
                  T * mx, T * my, T * mz)
{


}
template<typename T>  bool writetoOvfFile_device(std::string filename,OvfInfo info,
                  T * Mx_padded, T * My_padded, T * Mz_padded)
{


}
template<typename T>   void print_mat(string s,T *mx , int nx , int ny, int nz)
{
    cout << "--------" << s << "----" << "----" << endl;
            for (int kk = 0; kk < nz; kk++) {
                cout << "nz = " << kk << endl;
                for (int ii = 0; ii < nx; ii++) {
                    for (int jj = 0; jj < ny; jj++) {
                        printf("%g ",
                                mx[kk + jj * nz
                                        + ii * nz * ny]);
                    }
                    cout << endl;
                }
            }
}

int test()
{
    int nx,ny,nz;
    nx=2;ny=2;nz=1;
    double dx,dy,dz;
    dx =2,dy=2;dz=1;

    OvfInfo ovfinfo;
    ovfinfo.xnodes = nx;
    ovfinfo.ynodes = ny;
    ovfinfo.znodes = nz;
    ovfinfo.xstepsize = dx;
    ovfinfo.ystepsize = dy;
    ovfinfo.zstepsize = dz;

//    FLOAT_ mx[nx*ny*nz],my[nx*ny*nz],mz[nx*ny*nz];
    FLOAT_ * mx = new FLOAT_[nx*ny*nz];
    FLOAT_ * my = new FLOAT_[nx*ny*nz];
    FLOAT_ * mz = new FLOAT_[nx*ny*nz];

    parseOvfFile<FLOAT_>("a.ovf",ovfinfo, mx, my, mz);

    print_mat<FLOAT_>("mx",mx , nx ,ny, nz);
    print_mat<FLOAT_>("my",my , nx ,ny, nz);
    print_mat<FLOAT_>("mz",mz , nx ,ny, nz);

    writetoOvfFile<FLOAT_>("out2.txt", ovfinfo,
                      mx, my, mz);

    delete mx,my,mz;
    return 0;
}

}
/*
 *
 *
 * For rectangular meshes, data input is field values only,
 * in x , y , z component triples.
 * These are ordered with the x index incremented first,
 * then the y index, and the z index last.
 *
 */

#endif /* OVF_IO_H_ */
