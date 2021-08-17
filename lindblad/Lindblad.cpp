#include <lindblad/Lindblad.h>


Lindblad::Lindblad(const LindbladParams& lp)
: lp(lp), stepID(0), Emin(+DBL_MAX), Emax(-DBL_MAX), tPrev(0.),
	K(1./3, 1./3, 0), Kp(-1./3, -1./3, 0)
{
}


Lindblad::~Lindblad()
{
}


DM1 Lindblad::compute(double t, const DM1& v)
{
	return DM1(); //TODO
}


void Lindblad::report(double t, const DM1& v) const
{
	//TODO
}
