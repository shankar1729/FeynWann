#include <core/scalar.h>

extern "C"
{
	int tdep_initialize_();
	int tdep_compute_(double* qcart, double* omega, complex* U);
}
