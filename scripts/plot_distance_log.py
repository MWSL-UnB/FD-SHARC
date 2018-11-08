
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

file_name = "[SYS] CDF of system INR.txt"

direction = 'DL'

directories = ['.\\40_DL_center',
               '.\\40_DL_100',
               '.\\40_DL_200',
               '.\\40_DL_400',
               '.\\40_DL_600',
               '.\\40_DL_800',
               '.\\40_DL_1000',
               '.\\40_UL_center',
               '.\\40_UL_100',
               '.\\40_UL_200',
               '.\\40_UL_400',
               '.\\40_UL_600',
               '.\\40_UL_800',
               '.\\40_UL_1000']

cases = []
for dr in directories:
    case = ''
    if 'DL' in dr: case = case + 'IMT BS, '
    if 'UL' in dr: case = case + 'IMT UE, '
    
    if '200' in dr: case = case + '200m'
    if '400' in dr: case = case + '400m'
    if '600' in dr: case = case + '600m'
    if '800' in dr: case = case + '800m'
    if '1000' in dr: case = case + '1000m'
    elif '100' in dr: case = case + '100m'
    if 'center' in dr: case = case + 'center'
    
    cases.append(case)


plt.figure(figsize=(16,6), facecolor='w', edgecolor='k')
plt.subplot(121)
for dire, case in zip(directories,cases):
    
    if direction not in dire: continue
	
    file_path = os.path.join(dire,file_name)
    data = np.loadtxt(file_path,skiprows=1)
	
    frmt = ''
    if '200' in dire: frmt = frmt + 'k'
    if '400' in dire: frmt = frmt + 'm'
    if '600' in dire: frmt = frmt + 'c'
    if '800' in dire: frmt = frmt + 'b--'
    if '1000' in dire: frmt = frmt + 'k--'
    elif '100' in dire: frmt = frmt + 'b'
    if 'center' in dire: frmt = frmt + 'm--'
    
    inr = data[:,0]
    prob_exceed = 100-100*data[:,1]
	
    plt.semilogy(inr,prob_exceed,frmt,label=case)
	
    f = interp1d(prob_exceed, inr)
	
#    print(case)
#    print("1% = " + str(f(1)))
#    protec = -6
#    apport = protec - 3
#    print(" 1 Margin = " + str(f(1) - protec))
#    print(" 1 App. Margin = " + str(f(1) - apport))
#    print("20% = " + str(f(20)))
#    protec = -10
#    apport = protec - 3
#    print(" 20 Margin = " + str(f(20) - protec))
#    print(" 20 App. Margin = " + str(f(20) - apport))

protec_x = -10*np.ones(1000)
protec_y = np.linspace(5e-3,100,num=1000)
plt.semilogy(protec_x,protec_y,'r',label='Protection criteria: [-10dBm]')
apport_x = protec_x - 3
apport_y = protec_y
plt.semilogy(apport_x,apport_y,'r--',label='Protection criteria [-10dBm] + apportionment')

protec_x = -6*np.ones(1000)
protec_y = np.linspace(5e-3,100,num=1000)
plt.semilogy(protec_x,protec_y,'r-.',label='Protection criteria: [-6dBm]')
apport_x = protec_x - 3
apport_y = protec_y
plt.semilogy(apport_x,apport_y,'r:',label='Protection criteria [-6dBm] + apportionment')

plt.xlabel("$I/N$ [dB]")
plt.ylabel("Percentage of time I/N is exceeded")
plt.ylim([5e-2,100])
plt.grid()
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.savefig('distance_log_' + direction + '.png',bbox_inches='tight')