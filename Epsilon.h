#ifndef WANNIERMC_EPSILON_H
#define WANNIERMC_EPSILON_H

#include <core/string.h>
#include <vector>

class Epsilon
{	double omegaStart, domega, omegaStop; //!< frequency grid
	complex epsilonStart; //!< epsilon at omegaStart; the splines store the difference from this value
	std::vector<double> coeffRe, coeffIm; //!< quintic spline coefficients for epsilon real and imaginary parts
	double Gamma0, OmegaPsq; //!< Drude model parameters for omega < omegaStart
public:
	double omega; //!< current frequency
	complex epsilon; //!< complex dielectric constant
	double modGammaPlus; //!< decay length of plasmon in vacuum
	double modGammaMinus; //!< decay length of plasmon in metal
	double Lquant; //!< quantization length of plasmon
	double k; //!< propagation wave-vector
	
	Epsilon(string filename); //!< initialize fom a numerical tabulation of the dielectric constant
	void setFrequency(double omegaIn, bool print=true);
	double exptLinewidth() const; //!< calculate experimental linewidth
};
#endif //WANNIERMC_EPSILON_H
