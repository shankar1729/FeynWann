#include <core/Util.h>
#include <core/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <commands/command.h>
#include "FeynWann.h"
#include "Histogram.h"
#include "InputMap.h"
#include "LindbladFile.h"
#include "Integrator.h"
#include <core/Units.h>

int main(int argc, char** argv)
{            
    LindbladFile::Kpoint k0;
    
    int nBands        = 2;
    double larmorFreq = .001; // time unit is 1/140 fs
    k0.nInner     = k0.nOuter = nBands;
    k0.innerStart = 0;
    k0.E.resize(nBands);
    k0.E[0] = larmorFreq/2;
    k0.E[1] = -larmorFreq/2;
    
    for (int i=0; i < 3; i++)
    {   k0.S[i] = zeroes(nBands, nBands);
        k0.P[i] = zeroes(nBands, nBands);
    }
    
    // Sx
    k0.S[0].set(0,1, 1);
    k0.S[0].set(1,0, 1);
    
    // Sy
    k0.S[1].set(0,1, complex(0,-1));
    k0.S[1].set(1,0, complex(0, 1));

    // Sz
    k0.S[2].set(0,0, 1);
    k0.S[2].set(1,1,-1);
    
    double deltaOmegaX = 0*larmorFreq/10;
    double deltaOmegaY = 0*larmorFreq/5;
    matrix H = k0.E + 0.5*deltaOmegaX*k0.S[0] + 0.5*deltaOmegaY*k0.S[1];
    
    logPrintf("\n----k-----\n");
    k0.k.print(globalLog, "%lf");
    
    logPrintf("\n----H-----\n");
    H.print(globalLog);
    
    logPrintf("\n----E-----\n");
    k0.E.print(globalLog);
    
    logPrintf("\n----Sx-----\n");
    k0.S[0].print(globalLog);
    
    logPrintf("\n----Sy-----\n");
    k0.S[1].print(globalLog);

    logPrintf("\n----Sz-----\n");
    k0.S[2].print(globalLog);
    
    // Transform Sx, Sy, Sz into eigenbasis of H
    matrix V;
    H.diagonalize(V, k0.E);
    for (int i=0; i < 3; i++)
        k0.S[i] = dagger(V) * k0.S[i] * V;

    logPrintf("\n----HDiagonalized-----\n");
    const matrix HDiagonalized = dagger(V) * H * V;
    HDiagonalized.print(globalLog);
    
    logPrintf("\n----Eigenvectors-----\n");
    matrix daggerV = dagger(V);
    V.print(globalLog);
    logPrintf("--------\n");
    daggerV.print(globalLog);
    
    logPrintf("\n----SxDiagonalized-----\n");
    k0.S[0].print(globalLog);
    
    logPrintf("\n----SyDiagonalized-----\n");
    k0.S[1].print(globalLog);

    logPrintf("\n----SzDiagonalized-----\n");
    k0.S[2].print(globalLog);

    
    //Prepare the file header:
    LindbladFile::Header h;
    h.dmuMin = 0;
    h.dmuMax = 0;
    h.Tmax = 1e-1; // in Hartrees; 1e-3 ~ 320 K.
    h.pumpOmegaMax = 0;
    h.probeOmegaMax = 0;
    h.nk = 1;
    h.nkTot = 1;
    h.ePhEnabled = true;
    h.spinorial = true;
    h.spinWeight = 1;
    h.R = matrix3<>(1, 1, 1);
    
    //Compute offsets to each k-point within file:
    std::vector<size_t> byteOffsets(h.nk);
    byteOffsets[0] = h.nBytes() + h.nk*sizeof(size_t); //offset to first k-point (header + byteOffsets array)

    FILE* fp = fopen("ldbd.dat", "w");
    std::ostringstream oss;
    h.write(oss);
    
    fwrite(oss.str().data(), 1, h.nBytes(), fp);
    fwrite(byteOffsets.data(), sizeof(size_t), byteOffsets.size(), fp);

    oss.str(std::string());
    k0.write(oss, h);
    fwrite(oss.str().data(), 1, k0.nBytes(h), fp);
    
    fclose(fp);
    
    return 0;
}
