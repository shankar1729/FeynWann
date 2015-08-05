#!/bin/bash

#Puts together GammaAll-*.dat for * = expt, phonon and direct with drudeGamma

tmpDir="tmp.GammaCombine"
mkdir $tmpDir

outFile="GammaAll-combined.dat"

cut -f2 drudeGamma.dat > $tmpDir/resistive
cut -f3 drudeGamma.dat > $tmpDir/surface
cut -f2 GammaAll-direct.dat > $tmpDir/direct
cut -f2 GammaAll-phonon.dat > $tmpDir/phonon

echo "#omega Expt Direct Surface Phonon Resistive" > $outFile #header
paste GammaAll-expt.dat $tmpDir/direct $tmpDir/surface $tmpDir/phonon $tmpDir/resistive >> $outFile

rm -rf $tmpDir

gnuplot --persist <<EOF
	set term wxt enhanced
	set logscale y
	set yrange [1e-7:*]
	set xlabel "{/Symbol w} [eV]"
	set ylabel "{/Symbol G} [eV]"
	set key bottom right
	
	plot "GammaAll-combined.dat" \
		   u 1:2 w l title "Expt", \
		"" u 1:3 w l title "Direct", \
		"" u 1:(\$3+\$4) w l title "+Surface", \
		"" u 1:(\$3+\$4+\$5) w l title "+Phonon", \
		"" u 1:(\$3+\$4+\$5+\$6) w l title "+Resistive"
EOF