/*
 * terminal_print.h
 *
 *  Created on: 12 Nov, 2016
 *      Author: cg
 */

#ifndef TERMINAL_PRINT_H_
#define TERMINAL_PRINT_H_




template<typename T> void print_real(std::string s, T * d_array[], int l,
		int nx_padded, int ny_padded, int nz_padded, bool print_float = 1) {
	using namespace std;
	std::string p_format;
	if (print_float)
		p_format = "%.4f ";
	else
		p_format = "%.4g ";
	T * h_tmp = new T[nx_padded * ny_padded * nz_padded];
	FOR(i,0,l)
	{
		cudaMemcpy(h_tmp, d_array[i],
				sizeof(T) * nx_padded * ny_padded * nz_padded,
				cudaMemcpyDeviceToHost);

		cout << "--------" << s << "----" << i << "----" << endl;
		for (int kk = 0; kk < nz_padded; kk++) {
			cout << "nz = " << kk << endl;
			for (int ii = 0; ii < nx_padded; ii++) {
				for (int jj = 0; jj < ny_padded; jj++) {
					printf(p_format.c_str(),
							h_tmp[kk + jj * nz_padded
									+ ii * nz_padded * ny_padded]);
				}
				cout << endl;
			}
		}
		cout << "----------------" << endl;
	}

	delete h_tmp;
}
template<typename T> void print_real_along_col(std::string s, T * Hx, T * Hy, T * Hz, int l,
		int nx_padded, int ny_padded, int nz_padded, bool print_float = 1) {
	using namespace std;
	std::string p_format;
	if (print_float)
		p_format = "%.4f %.4f %.4f\n";
	else
		p_format = "%.4g %.4g %.4g\n";
	T * h_Hx = new T[nx_padded * ny_padded * nz_padded];
	T * h_Hy = new T[nx_padded * ny_padded * nz_padded];
	T * h_Hz = new T[nx_padded * ny_padded * nz_padded];
	cudaMemcpy(h_Hx, Hx,
					sizeof(T) * nx_padded * ny_padded * nz_padded,
					cudaMemcpyDeviceToHost);
	cudaMemcpy(h_Hy, Hy,
					sizeof(T) * nx_padded * ny_padded * nz_padded,
					cudaMemcpyDeviceToHost);
	cudaMemcpy(h_Hz, Hz,
					sizeof(T) * nx_padded * ny_padded * nz_padded,
					cudaMemcpyDeviceToHost);
	cout << "--------" << s << "----"  << "----" << endl;
	if(nx_padded*ny_padded*nz_padded<100)
	{
		for (int kk = 0; kk < nz_padded; kk++) {
			//cout << "nz = " << kk << endl;
			for (int jj = 0; jj < ny_padded; jj++){
				for (int ii = 0; ii < nx_padded; ii++) {
					printf(p_format.c_str(),
							h_Hx[kk + jj * nz_padded
									+ ii * nz_padded * ny_padded],
							h_Hy[kk + jj * nz_padded
									+ ii * nz_padded * ny_padded],
							h_Hz[kk + jj * nz_padded
									+ ii * nz_padded * ny_padded]);
				}
				//cout << endl;
			}
		}
	}
	cout << "----------------" << endl;

	delete h_Hx,h_Hy,h_Hz;
}
template<typename T> void print_complex(std::string s, T * d_array[], int l,
		int nx_padded, int ny_padded, int nz_padded, bool print_float = 1) {
	using namespace std;
	std::string p_format;
	if (print_float)
		p_format = "%.4f+i%.4f ";
	else
		p_format = "%.4g+i%.4g ";
	T * h_tmp = new T[nx_padded * ny_padded * nz_padded];
	FOR(i,0,l)
	{
		cudaMemcpy(h_tmp, d_array[i],
				sizeof(T) * nx_padded * ny_padded * nz_padded,
				cudaMemcpyDeviceToHost);

		cout << "--------" << s << "----" << i << "----" << endl;
		for (int kk = 0; kk < nz_padded; kk++) {
			cout << "nz = " << kk << endl;
			for (int ii = 0; ii < nx_padded; ii++) {
				for (int jj = 0; jj < ny_padded; jj++) {
					printf(p_format.c_str(),
							h_tmp[kk + jj * nz_padded
									+ ii * nz_padded * ny_padded].x,
							h_tmp[kk + jj * nz_padded
									+ ii * nz_padded * ny_padded].y);
				}
				cout << endl;
			}
		}
		cout << "----------------" << endl;
	}

	delete h_tmp;
}
;
template<typename T> void print_complex_real(std::string s, T * d_array[], int l,
		int nx_padded, int ny_padded, int nz_padded, bool print_float = 1) {
	using namespace std;
	std::string p_format;
	if (print_float)
		p_format = "%.4f ";
	else
		p_format = "%.4g ";
	T * h_tmp = new T[nx_padded * ny_padded * nz_padded];
	FOR(i,0,l)
	{
		cudaMemcpy(h_tmp, d_array[i],
				sizeof(T) * nx_padded * ny_padded * nz_padded,
				cudaMemcpyDeviceToHost);

		cout << "--------" << s << "----" << i << "----" << endl;
		for (int kk = 0; kk < nz_padded; kk++) {
			cout << "nz = " << kk << endl;
			for (int ii = 0; ii < nx_padded; ii++) {
				for (int jj = 0; jj < ny_padded; jj++) {
					printf(p_format.c_str(),
							h_tmp[kk + jj * nz_padded
									+ ii * nz_padded * ny_padded].x);
				}
				cout << endl;
			}
		}
		cout << "----------------" << endl;
	}

	delete h_tmp;
}
;
template<typename T> void print_complex_imag(std::string s, T * d_array[], int l,
		int nx_padded, int ny_padded, int nz_padded, bool print_float = 1) {
	using namespace std;
	std::string p_format;
	if (print_float)
		p_format = "%.4f ";
	else
		p_format = "%.4g ";
	T * h_tmp = new T[nx_padded * ny_padded * nz_padded];
	FOR(i,0,l)
	{
		cudaMemcpy(h_tmp, d_array[i],
				sizeof(T) * nx_padded * ny_padded * nz_padded,
				cudaMemcpyDeviceToHost);

		cout << "--------" << s << "----" << i << "----" << endl;
		for (int kk = 0; kk < nz_padded; kk++) {
			cout << "nz = " << kk << endl;
			for (int ii = 0; ii < nx_padded; ii++) {
				for (int jj = 0; jj < ny_padded; jj++) {
					printf(p_format.c_str(),
							h_tmp[kk + jj * nz_padded
									+ ii * nz_padded * ny_padded].y);
				}
				cout << endl;
			}
		}
		cout << "----------------" << endl;
	}

	delete h_tmp;
}
template<typename T> void print_complex_real_nopadding(std::string s, T * d_array[], int l,
		int nx_padded, int ny_padded, int nz_padded,
		int nx, int ny, int nz, bool print_float = 1) {
	using namespace std;
	std::string p_format;
	if (print_float)
		p_format = "%.4f ";
	else
		p_format = "%.4g ";
	T * h_tmp = new T[nx_padded * ny_padded * nz_padded];
	FOR(i,0,l)
	{
		cudaMemcpy(h_tmp, d_array[i],
				sizeof(T) * nx_padded * ny_padded * nz_padded,
				cudaMemcpyDeviceToHost);

		cout << "--------" << s << "----" << i << "----" << endl;
		for (int kk = 0; kk < nz; kk++) {
			cout << "nz = " << kk << endl;
			for (int ii = 0; ii < nx; ii++) {
				for (int jj = 0; jj < ny; jj++) {
					printf(p_format.c_str(),
							h_tmp[kk + jj * nz_padded
									+ ii * nz_padded * ny_padded].x);
				}
				cout << endl;
			}
		}
		cout << "----------------" << endl;
	}

	delete h_tmp;
}
;
template<typename T> void print_complex_imag_nopadding(std::string s, T * d_array[], int l,
		int nx_padded, int ny_padded, int nz_padded,
		int nx, int ny, int nz,bool print_float = 1) {
	using namespace std;
	std::string p_format;
	if (print_float)
		p_format = "%.4f ";
	else
		p_format = "%.4g ";
	T * h_tmp = new T[nx_padded * ny_padded * nz_padded];
	FOR(i,0,l)
	{
		cudaMemcpy(h_tmp, d_array[i],
				sizeof(T) * nx_padded * ny_padded * nz_padded,
				cudaMemcpyDeviceToHost);

		cout << "--------" << s << "----" << i << "----" << endl;
		for (int kk = 0; kk < nz; kk++) {
			cout << "nz = " << kk << endl;
			for (int ii = 0; ii < nx; ii++) {
				for (int jj = 0; jj < ny; jj++) {
					printf(p_format.c_str(),
							h_tmp[kk + jj * nz_padded
									+ ii * nz_padded * ny_padded].y);
				}
				cout << endl;
			}
		}
		cout << "----------------" << endl;
	}

	delete h_tmp;
};
template<typename T> void print_array (std::string s, T * d_array, int length, bool print_float = 1) {
	using namespace std;
	std::string p_format;
	if (print_float)
		p_format = "%.4f ";
	else
		p_format = "%.4g ";
	T * h_tmp = new T[length];

		cudaMemcpy(h_tmp, d_array,
				sizeof(T) * length,
				cudaMemcpyDeviceToHost);

		cout << "--------" << s << "----"  << "----" << endl;
		for (int kk = 0; kk < length; kk++) {

					printf(p_format.c_str(),
							h_tmp[kk]);
				if (kk%10 == 9)
					cout << endl;

		}
		cout << "----------------" << endl;


	delete h_tmp;
};

#endif /* TERMINAL_PRINT_H_ */
