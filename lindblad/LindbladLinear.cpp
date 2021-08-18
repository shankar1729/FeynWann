#include <lindblad/Lindblad.h>


LindbladLinear::LindbladLinear(const LindbladParams& lp)
: LindbladMatrix(lp), nnzD(rhoSizeTot), nnzO(rhoSizeTot)
{
	assert((not lp.spectrumMode) and lp.linearized);

	if(lp.ePhEnabled)
	{
		#ifdef PETSC_ENABLED
		initialize();
		#endif

		//Initialize sparse time evolution matrix in evolveMat:
		initializeMatrix();
		
		//Finalize matrix assembly:
		CHECKERR(MatAssemblyBegin(evolveMat, MAT_FINAL_ASSEMBLY));
		CHECKERR(MatAssemblyEnd(evolveMat, MAT_FINAL_ASSEMBLY));
		MatInfo info; CHECKERR(MatGetInfo(evolveMat, MAT_GLOBAL_SUM, &info));
		logPrintf("done. Net sparsity: %.0lf non-zero in %lu x %lu matrix (%.1lf%% fill)\n",
			info.nz_used, rhoSizeTot, rhoSizeTot, info.nz_used*100./(rhoSizeTot*rhoSizeTot));
		logFlush();
		
		//Create corresponding vectors:
		CHECKERR(MatCreateVecs(evolveMat, &vRho, &vRhoDot));
	}
}


LindbladLinear::~LindbladLinear()
{
	#ifdef PETSC_ENABLED
	cleanup();
	#endif
}


void LindbladLinear::rhoDotScatter()
{	//TODO
}


//------ PETSC library initalize and cleanup ------

#ifdef PETSC_ENABLED

void LindbladLinear::initialize()
{	//Create a fake command line for PetscInitialize (condlicts with FW command line):
	int argc = 1;
	char argvBuf[256]; strcpy(argvBuf, "lindblad/run");
	char* argv0 = &argvBuf[0];
	char** argv = &argv0;
	CHECKERR(PetscInitialize(&argc, &argv, (char*)0, ""));
}

void LindbladLinear::cleanup()
{	CHECKERR(MatDestroy(&evolveMat));
	CHECKERR(VecDestroy(&vRho));
	CHECKERR(VecDestroy(&vRhoDot));
	CHECKERR(PetscFinalize());
}

#endif
