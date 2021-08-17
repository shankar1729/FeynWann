#include <lindblad/Lindblad.h>


LindbladNonlinear::LindbladNonlinear(const LindbladParams& lp) : Lindblad(lp)
{	assert((not lp.spectrumMode) and (not lp.linearized));
	
	//Initialize A+ and A- for e-ph matrix elements if required:
	if(lp.ePhEnabled)
	{	for(State& s: state)
		{	for(LindbladFile::GePhEntry& g: s.GePh)
			{	g.G.init(s.nInner, nInnerAll[g.jk]);
				g.initA(lp.T, lp.defectFraction);
			}
		}
	}
}


LindbladNonlinear::~LindbladNonlinear()
{
}
