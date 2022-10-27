import matplotlib.pyplot as plt
import numpy as np
from reata_funcs import *

plt.rcParams['text.usetex'] = True
os.makedirs('outputs', exist_ok=1)
os.makedirs('out_rmsd', exist_ok=1)

fout = open('outputs/out.txt\n', 'w')
fout2 = open('outputs/out2.txt\n', 'w')
f_rmsd = open('out_rmsd/out.txt\n', 'w')


for z in ['1_fwrd']:
    os.chdir(z)
    vals = {'1_lig': [], '2_comp': []}
    fout.write(z[2:]+':\n')
    dg_l, dg_c = [], []
    for fle in ['1_lig', '2_comp']:
        os.chdir(fle)
        fout.write(f'  {fle[2:]}:\n')
        fig, ax = plt.subplots()
        fig1, ax1 = plt.subplots()
        dg = dg_l if 'lig' in fle else dg_c
        mdvdl, stdr = [], []
        for i, rn in enumerate([f'run{i}' for i in range(1, 5)], 1): #glob('run*')
            fout2.write(f'{fle}/{rn}: \n')
            os.chdir(rn)
            lams, tmdvdl, tmp, lvals = post_ti(istart=1000, iend=5000, fout=fout2, interp=1, f_rmsd=f_rmsd, wcomp=1 if i==1 else 0, sstp=1)
            # for i, x in zip(lams, lvals):
            #     plt.plot(cummean(x[1000:]) - mean(x), label=f'{i:.5f}')
            # plt.xlabel('time (ps)')
            # plt.ylabel(r'cumulative average for $\frac{dV}{d\lambda}$')
            # plt.legend(ncol=2)

            mdvdl.append(tmdvdl)
            dg.append(tmp)
            stdr.append([np.std(x[50:1000:10]) for x in lvals] / sqrt(len(lvals[0][50:1000:10])))
            ax1.plot(lams, tmdvdl, label=rn)
            fout.write(f'    {rn}: {dg[-1]:10.5f}\n')
            fout.flush()
            os.chdir('..')
        ax.errorbar(lams, mean(mdvdl, 0), yerr=np.std(mdvdl, 0), label='Average', capsize=4)
        ax1.errorbar(lams, mean(mdvdl, 0), label='Average')
        ax.set_title(f'{fle[2:]}: $\\langle \Delta G\\rangle={mean(dg_l if "lig" in fle else dg_c):.5f}$')
        ax.set_xlabel('$\lambda$')
        ax.set_ylabel('$\\frac{dV}{d\lambda} (kcal/mol)$')
        ax.legend()
        ax1.set_title(f'{fle[2:]}: $\\langle \Delta G\\rangle={mean(dg_l if "lig" in fle else dg_c):.5f}$')
        ax1.set_xlabel('$\lambda$')
        ax1.set_ylabel('$\\frac{dV}{d\lambda} (kcal/mol)$')
        ax1.legend()
        fig.savefig(f'../../outputs/{fle[2:]}')
        fig1.savefig(f'../../outputs/0_{fle[2:]}')
        vals[fle].append(mean(dg))
        print(f'{z[2:]}_{fle[2:]}: avg dg={mean(dg)}')
        fout.write(f'  sum : {sum(vals[fle]):10.5f}\n')
        os.chdir('..')
    os.chdir('..')
    for i, (x, y) in enumerate(zip(dg_l, dg_c), 1):
        fout.write(f'{i}: {y - x:.5f}\n')

    print(f"ddG_{z[2:]}={sum(vals['2_comp']) - sum(vals['1_lig'])}")
    fout.write(f"ddG={sum(vals['2_comp']) - sum(vals['1_lig']):10.5}\n")
    fout.write(f'std: {np.std(array(dg_c) - dg_l)/sqrt(len(dg_l))}')

