#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import glob

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
		plt.plot(tPS, S[iDir], label='Lab frame')
		plt.plot(tPS, Sprime[iDir], label='Rotating frame')
		plt.axvline(100 + 3.7995, linestyle='--', color='black')
		plt.axvline(200 + 3.7995, linestyle='--', color='black')
		plt.ylabel(fr'$\langle S_{"xyz"[iDir]}(t) \rangle$')
	plt.xlabel(r'$t$ [ps]')
	plt.xlim(tPS[0], tPS[-1])
	plt.savefig(f'{title}.png', bbox_inches='tight')
	plt.legend()
plt.show()
