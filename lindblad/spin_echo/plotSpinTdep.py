#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import glob

plt.rcParams['figure.figsize']  = 8, 8
plt.rcParams['figure.dpi']      = 100
plt.rcParams['image.cmap']      = 'jet'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.family']     = 'serif'
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

styles=['solid','solid', 'solid', 'solid','solid', 'solid', 'solid']
for iFile, fname in enumerate(glob.glob('*.out')):
	fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
	plt.subplots_adjust(hspace=0.1)
	title = fname.replace('.out','')
	print(f'Plotting {title}')
	t = []
	S = []
	for line in open(fname):
		if line.startswith('Integrate: Step:'):
			tokens = line.split()
			t.append(float(tokens[4]))
			S.append([float(tok) for tok in tokens[11:14]])
	tPS = np.array(t[1:])*0.001
	S = np.array(S[1:]).T
	
	#Rotating frame versions:
	Sprime = np.zeros(S.shape)
	cos_wt, sin_wt = np.cos(omega*tPS), np.sin(omega*tPS)
	rot = np.array([[cos_wt, sin_wt], [-sin_wt, cos_wt]])
	Sprime[:2] = np.einsum('ijt, jt -> it', rot, S[:2])
	Sprime[2] = S[2]

	#Plot each direction in separate panel:
	for iDir, ax in enumerate(axes):
		plt.sca(ax)
		plt.plot(tPS, S[iDir])
		plt.plot(tPS, Sprime[iDir])
		plt.axvline(100 + 3.7995, linestyle='--', color='black')
		plt.axvline(200 + 3.7995, linestyle='--', color='black')
		plt.ylabel(fr'$\langle S_{"xyz"[iDir]}(t) \rangle$')
	plt.xlabel(r'$t$ [ps]')
	plt.xlim(xmax=tPS[-1])
	plt.savefig(f'{title}.png', bbox_inches='tight')
plt.show()
