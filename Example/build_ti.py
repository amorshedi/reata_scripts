import os

from reata_funcs import *
from funcs import *
nl1, nl2 = 'rta', 'UNK'
snd = snd_gauss(snd, 12)
path = os.path.normpath(__file__)
fnum = re.findall('^\d+(?=_)', path.split(os.sep)[-2]) # folder number
tsnd = ''
lig1, lig2 = adjust_coords2(nl1, nl2, sel1='@O1,N1,C1', sel2=':131@O30,N36,C32', cpdb1=1, vac='vac.pdb', tol=0.3, mol2_1='rta_0.mol2', pdb1='rta.pdb')

sc1, sc2 = fix_order(nl1, nl2)

ln1 = '../../' + nl1
ln2 = '../../' + nl2
nres = 131

framefreq, printfreq = 5000000, 2000
nruns = {'1_lig': 1, '2_comp': 1}


nsteps = {'1_lig': 5000000, '2_comp': 5000000}
ibuffer = {'1_lig': 33.49162301007389, '2_comp': 12.112944438257816}
timask = {'1_lig': 1, '2_comp': nres}
pth2 = {'1_lig': '../rand.rst7', '2_comp': '../ti.rst7'}


to_run = {'1_lig': 1, '2_comp': 1}
to_run2 = {'1_fwrd': 1, '2_back': 1}

for z in ['1_fwrd']:
    mkch(z)
    if 'back' in z:
        ln1, ln2, sc1, sc2, nl1, nl2 = ln2, ln1, sc2, sc1, nl2, nl1

    for x in ['1_lig', '2_comp']:
        mkch(x)
        if to_run[x] and to_run2[z] and 'fwrd' in z:
            solvate(f'{"../../../4cxi.pdb " if "comp" in x else ""}{ln1}.mol2 {ln2}.mol2', nt_atoms=58456, ibuffer=ibuffer[x], nruns=nruns[x], maxiter=100, tol=2, frcmod='../../UNK')

        bz, fz = 'back' in z, 'fwrd' in z
        txt = f"\ntimask1=':{timask[x] + bz}', timask2=':{timask[x] + fz}', noshakemask=':{timask[x]},{timask[x] + 1}', \n"
        txt += f"ifsc = 1, scmask1='{(f':{timask[x] + bz}' if sc1 else '')+sc1}', scmask2='{(f':{timask[x] + fz}' if sc2 else '')+sc2}',"

        eq2, snd2 = (leq_dan if 'lig' in x else eq_dan), snd
        for g, h in {'reptxt': txt, 'nres': f'{timask[x] + 1}', 'jnme': f'{fnum}{z[2]}{x[2]}jnme', 'nsteps': f'{nsteps[x]}', 'framefreq': f'{framefreq}', 'printfreq': f'{printfreq}'}.items():
            eq2 = eq2.replace(g, h)
            snd2 = snd2.replace(g, h)

        sub = f'../../../1_fwrd/{x}/' if 'back' in z else '../../'
        for i in range(1, nruns[x] + 1):
            mkch(f'run{i}')
            if 'back' in z:
                eq3 = eq2.replace('pthparm', f'../../../../1_fwrd/{x}/solv_out/full.parm7').replace('pthrand', f'../../../../1_fwrd/{x}/run{i}/' + ('rand.rst7' if 'lig' in x else 'ti.rst7'))
                snd3 = snd2.replace('../../solv_out/full.parm7', f'../../../../1_fwrd/{x}/solv_out/full.parm7')
            else:
                eq3 = eq2.replace('pthrand', pth2[x]).replace('pthparm', '../../solv_out/full.parm7')
                snd3 = snd2


            if to_run[x] and to_run2[z] and 'fwrd' in z and 'comp' in x:
                m = pt.load(f'rand.rst7', f'../solv_out/full.parm7')
                adjust_coords(m, nres)
                m.save('ti.rst7')
                shell('mv ti.rst7.1 ti.rst7')
            os.makedirs('0_equil', exist_ok=1)
            open('0_equil/eq-dan.sh', 'w').write(eq3.replace('icfe = 1,', 'icfe = 1,gti_lam_sch = 1,'))
            open('send.sh', 'w').write(snd3.replace('jnme', f'{i}').replace('icfe = 1,', 'icfe = 1, gti_lam_sch = 1,'))
            shell('chmod +x send.sh 0_equil/eq-dan.sh')
            tsnd += f'sbatch --chdir={os.getcwd()} {os.getcwd()}/send.sh \n'
            os.chdir('..')
        os.chdir('..')
    os.chdir('..')
if os.uname().nodename == 'violet':
    open('sub_all.sh', 'w').write(tsnd.replace(os.getcwd(), '.'))
    shell('chmod +x sub_all.sh')
else:
    ngpu = 2
    for i, x in enumerate(np.array_split(tsnd.split('\n')[:-1], ngpu)):
        f = open(f'sub{i}.sh', 'w')
        f.write(f'export CUDA_VISIBLE_DEVICES={i}\n')
        shell(f'chmod +x sub{i}.sh')
        for y in x:
            direc = y.split()[1].replace('--chdir=', '').replace(f'{os.getcwd()}', '.')
            f.write(f'cd {direc}; ./send.sh; cd -\n')