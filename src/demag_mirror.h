/*
 * demag_mirror.h
 *
 *  Created on: 10 Nov, 2016
 *      Author: cg
 */

#ifndef DEMAG_MIRROR_H_
#define DEMAG_MIRROR_H_

#include "constant.h"
#include <iostream>
#include <stdio.h>
#include <math.h>

namespace demag_mirror {

void calDemagMirrorTensor(int nx_padded, int ny_padded, int nz_padded,
		int pbc_x, int pbc_y, int pbc_z,
		double dx, double dy, double dz,
		double & Kxx, double  & Kxy, double & Kxz,
		double & Kyy, double  & Kyz, double & Kzz

);

}
#endif /* DEMAG_MIRROR_H_ */
