#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import glob

plt.rcParams['figure.figsize']  = 8, 8
plt.rcParams['figure.dpi']      = 100
plt.rcParams['image.cmap']      = 'jet'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.family']     = 'serif'
plt.rcParams['font.weight']     = 'bold'
plt.rcParams['font.size']       = 25
plt.rcParams['font.sans-serif'] = 'serif'
plt.rcParams['text.usetex']     = False
plt.rcParams['axes.linewidth']  = 1.5
plt.rcParams['axes.titlesize']  = 'medium'
plt.rcParams['axes.labelsize']  = 'medium'

plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['xtick.major.pad']  = 8
plt.rcParams['xtick.minor.pad']  = 8
plt.rcParams['xtick.color']      = 'k'
plt.rcParams['xtick.labelsize']  = 'medium'
plt.rcParams['xtick.direction']  = 'in'

plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['ytick.major.pad']  = 8
plt.rcParams['ytick.minor.pad']  = 8
plt.rcParams['ytick.color']      = 'k'
plt.rcParams['ytick.labelsize']  = 'medium'
plt.rcParams['ytick.direction']  = 'in'

omega = 41.3414 # taken from test.cpp

plt.figure(1, figsize=(12,6))
styles=['solid','solid', 'solid', 'solid','solid', 'solid', 'solid']
SxArray = []; SyArray = []; SzArray = []
SxPrimeArray = []; SyPrimeArray = []
for iFile, fname in enumerate(glob.glob('xRelax*.out')):
	print(iFile, fname); title = fname.replace('xRelax', '').replace('.out','')
	t = []
	Sx = []; Sy = []; Sz = []
	for line in open(fname):
		if line.startswith('Integrate: Step:'):
			tokens = line.split()
			t.append(float(tokens[4]))
			Sx.append(float(tokens[11])); Sy.append(float(tokens[12])); Sz.append(float(tokens[13]))
	tPS = np.array(t[1:])*0.001
	Sx = np.array(Sx[1:])
	Sy = np.array(Sy[1:])
	Sz = np.array(Sz[1:])
	SxArray.append(Sx)
	SyArray.append(Sy)
	SzArray.append(Sz)
	
	SxPrime =  Sx*np.cos(omega*tPS) + Sy*np.sin(omega*tPS)
	SyPrime = -Sx*np.sin(omega*tPS) + Sy*np.cos(omega*tPS)
	SxPrimeArray.append(SxPrime)
	SyPrimeArray.append(SyPrime)


plt.plot(tPS, SxArray[0])
plt.plot(tPS, SxPrimeArray[0])
plt.axvline(100 + 3.7995, linestyle='--', color='black')
plt.axvline(200 + 3.7995, linestyle='--', color='black')

#plt.plot(tPS, np.sin(omega*tPS))
#plt.plot(tPS, SxPrime)

#for i in range(len(SxArray)):
	#Sx = SxArray[i]
	#Sy = SyArray[i]
	#Sz = SzArray[i]

	##plt.plot(tPS, Sz/Sz[0], ls=styles[iFile])
	#plt.plot(tPS, Sx/Sz[0], label=title, ls=styles[iFile], alpha=1)
	##plt.plot(tPS, Sy/Sz[0], label=title, ls=styles[iFile])
	
#plt.legend('$\omega = 0$')
#plt.axhline(1, color='k', ls='dotted', lw=1)
#plt.axhline(0, color='k', ls='dotted', lw=1)
#plt.axhline(-1, color='k', ls='dotted', lw=1)
plt.xlabel(r'$t$ [ps]'); plt.xlim(0, 100)
#plt.ylabel(r'$\langle S_x(t) \rangle / \langle S_x(0) \rangle$')
plt.ylabel(r'$\langle S_x(t) \rangle$')
plt.xlim(xmax=tPS[-1])

plt.savefig('/home/mani/tmp/S_x_with_echo.png', bbox_inches='tight')
plt.show()
