from subprocess import check_output
import subprocess as sp
import regex as re
import numpy as np
import io, os, datetime, shutil
from copy import deepcopy
from glob import glob
import scipy.optimize
from numpy import arange, array
from plotly import graph_objects as go
# from plotly.offline import plot
import matplotlib.pyplot as plt
import parmed as pd
import pytraj as pt
from parmed.tools.actions import tiMerge
from funcs import *

#import plotly.express as px
norm = np.linalg.norm
if os.uname().nodename == 'violet':
    leapcmd = '/home/ali/software/amber20/bin/tleap -s -f - '
    cpptraj = '/home/ali/software/amber20/bin/cpptraj'
else:
    leapcmd = 'tleap -s -f - '
    cpptraj = 'cpptraj'
parmed = 'parmed -O '
leapin0 = 'source leaprc.protein.ff19SB\nsource leaprc.water.opc\n'

if 'c109' in shell('hostname'):
    pmemd_exec = '/root/shared/projects_ali/amber20/bin/pmemd.cuda_SPFP'
elif 'COMP' in shell('hostname'):
    pmemd_exec = '/home/chemistry/NFS/projects_ali/amber20/bin/pmemd.cuda_SPFP'
else:
    pmemd_exec = 'pmemd.cuda'

def shell(comm, inp=''):
    '''comm: command to run
    inp: a byte string to give as input'''
    try:
        o = check_output(comm, shell=1, input=inp.encode(), executable='/bin/bash') if inp else check_output(comm, shell=1, executable='/bin/bash')
    except sp.CalledProcessError as ex:
        o = ex.output
    return o.decode()

def mat_prt(mat, frm = '10.5f', prt=0, sep=' '):
    mat=np.array(mat)
    dims = mat.shape

    frm = f'{{:{frm}}}{sep}'
    out = dims[0] * frm
    if len(dims) > 1:
        out = out.replace(' ', ' \n')
        out = out.replace(frm, frm * dims[1])
    out = out.format(*mat.flatten())
    if prt:
        print(out)
    return out

def txt_to_mat(inp, dt=float):
    return np.genfromtxt(io.StringIO(inp), dtype=dt)

def cummean(A,axis=0):
    """ Cumulative averaging

    Parameters
    ----------
    A    : input ndarray
    axis : axis along which operation is to be performed

    Output
    ------
    Output : Cumulative averages along the specified axis of input ndarray
    """

    return np.true_divide(A.cumsum(axis),np.arange(1,A.shape[axis]+1))

def net_charge(lig):
    ch = 0
    for x in lig.atoms:
        ch += x.charge
    return ch

def get_ion_mask(ch1, ch2, ti=0, nres=0):
    if ti:
        system = pd.load_file('solv_out/full.parm7', 'run1/ti.rst7')
        # lig1, lig2 = system.residues[nres:nres+2]
    else:
        system = pd.load_file('solv_out/rand.pdb')

    mask = ['|:', '|:']
    for j, c in enumerate([ch1, ch2]):
        if int(c) < 0:
            ion = 'Na+'
        elif np.isclose(abs(int(c)), 0, atol=.1, rtol=0):
            continue
        else:
            ion = 'Cl-'
        cnt = 0
        for i, x in enumerate(system.residues, 1):
            if x.name == ion:
                mask[1-j] += f'{i},'
                cnt += 1
                if np.isclose(abs(int(c)), cnt, atol=.1, rtol=0):
                    break
        mask[1-j] = mask[1-j][:-1]
    for i in range(2):
        if mask[i] == '|:':
            mask[i] = ''
    return mask


def solvate(fle='', watnum=13618, molarity=0.15, catsym='Na+', ansym='Cl-', maxiter=20, tol=2,
            ncextra=0, naextra=0, ntrlz=1, seed=1, hmr=0, nparm='full.parm7', nhparm='full.hmr.parm7', use_pt=0,
            nruns=0, ff='source leaprc.protein.ff19SB\nsource leaprc.water.opc\nsource leaprc.gaff\n', tcom='', pleap=0, nt_atoms='',
             watbox='OPCBOX', ibuffer=0, frcmod='', mol2='', batoms1='', batoms2='', bcut=3, cpptp='', nonf=0, svsolv=0, lrand=1):
    '''
    ff: the forcefield parameters
    tcom: if fle is not provided, commands in this should make an "m" compound. If fle provided, tcom is added after "m" is construncted
    fle: file to be solvated
    watnum, molarity, rnares, catsym =num of water molecs, salt molarity, num of res in protein, cation symbol
    ncextra: extra cations
    naextra: extra anions
    ibuffer: initial buffer size
    outdir: creates a file for outputs
    ntrlz: do you want to neutralize first?
    load: custom load in leap. it should give an m UNIT for leap
    tol:tolerance on num of atoms
    maxiter: number of iterations in the newton root-finding procedure
    nruns: how many randomizations do we want?
    nparm: name of the written parm file
    nhparm: name fo the written parm of after hmr
    pleap: print the first leap exec
    nt_atoms: total number of atoms in the system
    frcmod: name of the frcmod file for leap
    batoms1, batoms2: atoms that we want bonded
    mol2: mol2 file that should be loaded
    cpptp: cpptraj path
    nonf: not the final files
    svsolv: do you want to save the pdb of the solution and the two ligands separately ?
    lrand: use addionsrand in leap'''
    cpptraj = os.path.join(cpptp, 'cpptraj')

    for x in frcmod.split(' '):
        ff += 'loadamberparams ' + x + '.frcmod\n'

    for x in mol2.split():
        tmp = pd.load_file(f'{x}.mol2', structure=True)
        ff += f'\n{tmp.residues[0].name}=loadmol2 ' + x + '.mol2\n'

    os.makedirs('solv_info', exist_ok=1)
    os.makedirs('solv_out', exist_ok=1)
    log = open('build.log', 'w')

    log.write(f'''log file created: {datetime.datetime.now()}

    Solvation parameters:
        Number of target water molecules: {watnum}
        Target molarity: {molarity}
        Cations, Anions: {catsym}, {ansym}
        Tolerance on number of atoms: {tol}
        Perform HMass repartition?: {hmr}
        Randomization seed: {seed}
    ''')

    if ' ' in fle:
        comb = "\nm=combine {"
        for i, x in enumerate(fle.split()):
            ff += f'comp{i}=' + ('loadpdb ' if 'pdb' in x else 'loadmol2 ') + x + '\n'
            comb += f'comp{i} '
        if 'comp' in comb:
            ff += comb[:-1] + '}\n' + tcom
    elif fle:
        ff += f"m={'loadpdb ' if 'pdb' in fle else 'loadmol2 '}{fle}"
    ff += tcom

    global leapout, leapin, watadded, ncats, nanions

    leapin = ff + f"\nsaveamberparm m temp.parm7 temp.rst7\nsavepdb m temp.pdb\nquit"
    leapout = shell(leapcmd, leapin)
    if pleap:
        print(leapin)
        print(leapout)
        open('solv_info/tleap.in', 'w').write(leapin)
        open('solv_info/tleap.out', 'w').write(leapout)
        os._exit(0)
    tmp = open('temp.parm7').read()
    nntrlz = abs(round(sum(np.array(re.search('(?s)%FLAG CHARGE.*?(?=%FLAG)', tmp).group().split()[3:], dtype=float)) / 18.2223))


    comp = pd.load_file('temp.pdb')
    atoms1, atoms2 = [], []
    for x in batoms1.split():
        atoms1.extend(comp[x].atoms)
    for x in batoms2.split():
        atoms2.extend(comp[x].atoms)

    for x in atoms1:
        cnt = 0
        for y in atoms2:
            if norm(np.subtract([x.xx, x.xy, x.xz], [y.xx, y.xy, y.xz])) < bcut:
                cnt += 1
                if cnt > 1:
                    raise Exception('More than one bond created for a single atom !')
                print('Bond command added: ' + (txt := f"bond m.{x.residue.number}.{x.name} m.{y.residue.number}.{y.name}"))
                ff += '\n' + txt


    rnares, natoms = len(comp.residues), comp.atoms.__len__()
    shell('rm temp.pdb temp.parm7 temp.rst7')
    # nparts = nt_atoms if nt_atoms else natoms + 4 * watnum - 3 * (ncats + nanions + nntrlz)


    def func(x, i):
        global leapout, leapin, watadded, ncats, nanions
        leapin = ff + f"\nsolvateoct m OPCBOX {x} iso\n"
        leapout = shell(leapcmd, leapin+'\nquit')
        volume = float(re.search('(?<=Volume:).*?(?=A)', leapout).group())
        tmp = int(np.around(6.022 * volume * molarity / 1e4))  # num of ion molecules needed to reach the specified molarity
        ncats, nanions = tmp + ncextra, tmp + naextra

        watadded = int(re.search('(?<=Added).*?(?=residues)', leapout).group())
        cur_atoms = natoms + 4 * watadded - 3 * (ncats + nanions + nntrlz) # current atoms considering replacement with ions that will happen later
        if nt_atoms:
            txt = f'{i}: Total num of atoms: {cur_atoms:6}; Diff: {cur_atoms-nt_atoms:6}; Added {watadded:6} water molecules'
            ndiff = cur_atoms - nt_atoms
        else:
            txt = f'{i}: Water molecules added: {watadded:6}; Diff: {watadded-watnum:6}'
            ndiff = watadded - watnum
        print(txt)
        log.write(txt + '\n')
        return ndiff

    iconv = 0
    x0, y0 = ibuffer, func(ibuffer, 0)
    if abs(y0) > tol:
        x1, y1 = ibuffer+3, func(ibuffer+3, 1)
        for i in range(maxiter):

            # try:
            df = (y1 - y0) / (x1 - x0)
            if np.isclose(x0, x1) or df == 0:
                while 1:
                    x0, x1 = (x0 + x1)/2 - .05, (x0 + x1)/2 + .05
                    if (y0:=func(x0, i+2))*(y1:=func(x1, i+2)) < 0:
                        break
                while abs(y1) > tol:
                    i += 1
                    c = (x0 + x1)/2
                    if y0*(y1:=func(c, i+2)) < 0:
                        x1 = c
                    else:
                        x0 = c
                        y0 = y1
                    if i == maxiter:
                        break
                iconv = 1


                    # x0 = x1
            # except ZeroDivisionError:
            #     # x1 += 1
            #     df = 50 * y1 * np.sign(y1 - y0)
            #     warn('If it is possible to get a smaller Diff, contact author')
            # if df == 0:
            #     df = 50 * y1
            #     # continue
            if iconv:
                break
            x0, x1 = x1, x1 - y1/df
            y0, y1 = y1, func(x1, i+2)
            if abs(y1) <= tol:
                break
    else:
        x1, y1 = x0, y0

    print(f'\nFinal buffer: {x1}')

    ntres = rnares + watadded
    # nwatext = divmod(y1, 3)[0]
    # for i in range(nwatext): # if y1 <=2, nothing is done here
    #     leapin += f'remove m m.{ntres - i}\n'
    # ntres -= nwatext

    if ntrlz:
        leapin += f'addions{"rand" if lrand else ""} m {catsym} 0\n'
    leapin += f'addions{"rand" if lrand else ""} m {catsym} {ncats} {ansym} {nanions}{" 3" if lrand else ""}\n'
    leapout = shell(leapcmd, leapin+'\nquit')
    for i in range(len(re.findall('No solvent overlap', leapout))):
        leapin += f'remove m m.{ntres - i}\n'
    leapin += f'\nsavepdb m solv_out/rand_leap.pdb\nsaveamberparm m solv_out/{nparm} solv_out/rand_leap.rst7\n'
    leapout = shell(leapcmd, leapin+'\nquit')
    open('solv_info/tleap.in', 'w').write(leapin+'\nquit')
    open('solv_info/tleap.out', 'w').write(leapout)

    nions = len(re.findall('Placed', leapout))  # total number of added ions
    print(f'Added {ncats} {catsym} atoms and {nanions} {ansym} atoms to get the molarity {molarity} ')
    log.write(f'\nAdded {ncats} {catsym} atoms and {nanions} {ansym} atoms to get the molarity {molarity}\n')

    if hmr:
        comp = pd.load_file(f'solv_out/{nparm}')
        pd.tools.actions.HMassRepartition(comp).execute()
        comp.save(f'solv_out/{nhparm}', overwrite=True, format='amber')
        log.write('\nHMassRepartition executed with python parmed\n')

    if use_pt:
        m = pt.load(f'solv_out/rand.rst7', f'solv_out/{nparm}')  # type: pt.trajectory.trajectory.Trajectory
        log.write(
            f'\nrandomized ions with pytraj: pt.randomize_ions(m, :{rnares + 1}-{rnares + nions}, around=:1-{rnares}, by=4, overlap=4, seed={seed})\n')
        pt.randomize_ions(m, mask=f':{rnares + 1}-{rnares + nions}', around=f':1-{rnares}', by=4, overlap=4, seed=seed)
        # m.autoimage('origin')
        m.save('solv_out/rand.rst7')
        m.save('solv_out/rand.pdb')
        m.strip(f':{rnares + 1}-{ntres}')
        m.save('solv_out/vac.rst7')
        m.save('solv_out/vac.pdb')
        m.save('solv_out/vac.parm7')
    else:
        cppin = f'parm solv_out/full.{"hmr." if hmr else ""}parm7 \ntrajin solv_out/rand_leap.rst7 \n'
        cppin += f'randomizeions :{rnares + 1}-{rnares + nions} around :1-{rnares} by 4.0 overlap 4.0 noimage seed {seed}\n' \
                 f'autoimage :1-{rnares} origin\n' \
                 f'outtraj solv_out/rand.rst7\nouttraj solv_out/rand.pdb\nstrip :{rnares + 1}-{ntres} ' \
                 f'parmout solv_out/vac.parm7\nouttraj solv_out/vac.rst7\nouttraj solv_out/vac.pdb\n'
        if svsolv:
            cppin += f'strip :1-{rnares},{rnares+2}-999999\nouttraj solv_out/solv_lig1.pdb\nunstrip\n'
            cppin += f'strip :1-{rnares},{rnares+1},{rnares+3}-999999\nouttraj solv_out/solv_lig2.pdb\n'
        open('solv_info/cpptraj.in', 'w').write(cppin)
        a = shell(f'{cpptraj} -i solv_info/cpptraj.in')
        open('solv_info/cpptraj.out', 'w').write(a)

        # parmin = f'parm full.hmr.parm7\nHMassRepartition\noutparm full.hmr.parm7\nquit'
        # shell(parmed, parmin)

    for x in ['build.log', 'leap.log']:
        shutil.move(x, f'solv_info/{x}')

    # for x in ['full.parm7', 'rand_leap.pdb']:
    #     shutil.move(x, f'solv_out/{x}')

    for i in range(1, nruns + 1):
        os.makedirs(f'run{i}', exist_ok=1)
        os.chdir(f'run{i}')
        cppin = f'parm ../solv_out/full.{"hmr." if hmr else ""}parm7 \ntrajin ../solv_out/rand_leap.rst7 \n'
        cppin += f'randomizeions :{rnares + 1}-{rnares + nions} around :1-{rnares} by 4.0 overlap 4.0 noimage seed {seed + i - 1}\n' \
                 f'autoimage :1-{rnares} origin\n' \
                 f'{"#" if nonf else ""}outtraj rand.rst7\nouttraj ../solv_out/rand_{i}.pdb\n'
        if svsolv:
            cppin += f'strip :1-{rnares+2} \nouttraj ../solv_out/solution_{i}.pdb'
        a = shell(cpptraj, cppin)
        open(f'../solv_info/cpptraj_{i}.in', 'w').write(cppin)
        open(f'../solv_info/cpptraj_{i}.out', 'w').write(a)
        print(f'\nCreated directory run{i}. Randomized ions with seed={seed + i - 1}')
        # m = pt.load('../rand.rst7', '../full.parm7')
        # pt.randomize_ions(m, mask=f':{rnares + 1}-{rnares + nions}', around=f':1-{rnares}', by=4, overlap=4, seed=seed+i)
        #
        # m.autoimage()
        # m.save('rand.rst7')
        # m.save('rand.pdb')
        # log.write(f'\n(directory run{i})randomized ions with pytraj: pt.randomize_ions(m, :{rnares + 1}-{rnares + nions}, around=:1-{rnares}, by=4, overlap=4, seed={seed+i})\n')

        os.chdir('..')

    if nonf:
        shell('rm solv_out/full.parm7')

    # os.system(f'mkdir -p {outdir};mv rand* {outdir};mv full* {outdir}')

def get_single_lam(lam, istart=0, iend=None):
    mdvdl = []
    for i, rn in enumerate(glob('run*'), 1):
        for y in glob(f'{rn}/*'):
            if a:=re.findall('(?<=.*?lam_).*', y):
                if txt_to_mat(a[0]) == lam:
                    f = open(y + '/run.out')
                    dvdl = []
                    for x in f:
                        if 'DV/DL  =' in x:
                            dvdl.append(re.search('(?<=DV/DL  =).*', x).group())
                        if 'A V E R A G E S' in x:
                            break
                    dvdl = array(dvdl, dtype=float)
                    g = dvdl[istart:iend] if iend else dvdl[istart:]
                    mdvdl.append(np.mean(g))
                    print(y)
    return mdvdl


def fix_order(nme1='rta', nme2='tx', cpdb1=0, cpdb2=0, ctol=.01, out_suffix=''):
    'the softcore atoms should be last'

    sc1, sc2, lig1, lig2 = get_sc(f'{nme1}', f'{nme2}', 1, ctol=ctol)

    tsc1, tsc2 = sc1[1:].split(','), sc2[1:].split(',')

    for inlig, sc, nme in zip((lig1, lig2), (tsc1, tsc2), (nme1, nme2)):
        lig = pd.Structure()
        sc_atoms = []
        for x in inlig.atoms:
            if x.name in sc:
                sc_atoms.append(x)
            else:
                lig.add_atom(x, x.residue.name, x.residue.number)
        for x in sc_atoms:
            lig.add_atom(x, x.residue.name, x.residue.number)
        lig.bonds = inlig.bonds
        lig.save(f'{nme}{out_suffix}.mol2', overwrite=1)
    return sc1, sc2


def fix_order2(nme='rta', full_pdb=1):
    if full_pdb:
        pdb = pd.load_file(f'init_{nme}.pdb')[f':{nme}']
    else:
        pdb = pd.load_file(f'{nme}.pdb')
    mol2 = pd.load_file(f'{nme}_1.mol2', structure=True)

    pdb.bonds = pd.TrackedList()

    atoms, charges, types = iodict(), iodict(), iodict()
    for x in mol2.atoms:
        charges[x.name] = x.charge
        types[x.name] = x.type
    for x in pdb.atoms:
        atoms[x.name] = x
        x.charge = charges[x.name]
        x.type = types[x.name]

    for x in mol2.bonds:
        at1, at2 = atoms[x.atom1.name], atoms[x.atom2.name]
        pdb.bonds.append(pd.Bond(at1, at2))
    pdb.save(f'{nme}.mol2', overwrite=1)


def get_sc(ln1, ln2, ret_ligs=0, chg=0, atp=0, ctol=.01):
    '''
    This function detects the softcore potentials and adjusts the coordinates according to a pdb file (which may or may not include the protein)

    ret_ligs: return ligands
    ln1, ln2: name of mol2 file
    chg: atoms with different charge are considered different
    atp: if atom types are different, they are considered different
    ctol: tolerance on coordinates
    '''

    m = [pd.load_file(f'{ln1}_1.mol2', structure=True), pd.load_file(f'{ln2}_1.mol2', structure=True)]

    at1, at2 = m[0].atoms, m[1].atoms
    natoms = len(at1) if len(at1) > len(at2) else len(at2)

    commons = []
    for x in at1:
        a1, a2 = m[0][f'@{x.name}'], m[1][f'@{x.name}']
        if not a1.atoms or not a2.atoms:
            continue
        a1, a2 = a1.atoms[0], a2.atoms[0]
        flg1 = np.isclose(a1.charge, a2.charge, atol=.001, rtol=0) if chg else 1
        flg2 = a1.type == a2.type if atp else 1
        if flg1 and flg2 and norm(array([a1.xx, a1.xy, a1.xz]) - array([a2.xx, a2.xy, a2.xz])) < ctol:
            commons.append(a1.name)

    sc1, sc2 = '@', '@'
    for x in at1:
        if x.name not in commons:
            sc1 += f'{x.name},'
    for x in at2:
        if x.name not in commons:
            sc2 += f'{x.name},'
    sc1, sc2 = sc1[:-1], sc2[:-1]
    if ret_ligs:
        return sc1, sc2, m[0], m[1]
    else:
        return sc1, sc2

def post_ti(lmax=1.03, istart=0, iend=None, iplt=0, sfig=0, fname='fig.jpeg', interp=0, rerun=0, gauss=1, fout='', f_rmsd='', wcomp=0, sstp=0):
    '''
    sfig: save the figure
    rerun: prep for rerun of end points
    fout: a file to print outputs
    rerun: if one of the lambdas didn't run
    wcomp: write the comp structures to compare with ref
    sstp: single step runs
    '''
    direcs = next(os.walk('.'))[1]
    tsnd = ''
    mdvdl, lams, interp_at = [], [], []
    vals = []
    for dd in direcs:
        snd = open('send.sh').readlines()
        if a := re.findall('(?<=.*?lam_).*', dd):  #\K in regex: forget all up to here
            # lams.append(re.search('(?<=clambda = ).*?(?=,)', open(f'{dd}/run.in').read()).group())
            f = open(dd + '/run.out')
            dvdl = []
            for x in f:
                if 'DV/DL  =' in x:
                    dvdl.append(re.search('(?<=DV/DL  =).*', x).group())
                if 'A V E R A G E S' in x:
                    break
            if '*' in ''.join(dvdl):
                g=[]
            else:
                dvdl = array(dvdl, dtype=float)
                g = dvdl[istart:iend] if iend else dvdl[istart:]


            if len(g)==0 and rerun:
                snd = [*snd[:11], snd[36]]
                if '0.00' in dd:
                    snd[-1] = snd[-1].replace('0_equil/step9', 'lam_0.05/res')
                    shell(f"sed -i 's/clambda = 0.00/clambda = 0.02/' {dd}/run.in")
                else:
                    snd[-1] = snd[-1].replace('0_equil/step9', 'lam_0.95/res')
                    shell(f"sed -i 's/clambda = 1.00/clambda = 0.98/' {dd}/run.in")
                open(f'{dd}/snd.sh', 'w').write(''.join(snd))
                tsnd += f'sbatch --chdir={os.getcwd()}/{dd} {os.getcwd()}/{dd}/snd.sh \n'

            if len(g) != 0:
                lams.extend(a)
                mdvdl.append(np.mean(g))
                vals.append(g)
            else:
                warn(f'There is a problem in {dd}')
            print(f'{dd}; len={len(dvdl)}')
            # if 'comp' in os.getcwd() and 'vdwl' in os.getcwd():

            if fout:
                fout.write(f'{dd}; len={len(dvdl)}\n')
    if 'comp' in os.getcwd() and wcomp:
        get_rmsd(os.getcwd(), fout=f_rmsd, sstp=sstp, gauss=gauss, nme=[x for x in direcs if 'lam' in x])
    lams, mdvdl = array(lams, dtype=float), array(mdvdl, dtype=float)
    idx = np.argsort(lams)
    lams, mdvdl, vals = lams[idx], mdvdl[idx], itemgetter(*idx)(vals)
    # for x in [[0.00922, 0.99078] if gauss else [0.0, 1.0]][0]:
    #     if x not in lams and interp:
    #         val = scipy.interpolate.interp1d(lams, mdvdl, kind='linear', fill_value='extrapolate')(x)
    #         lams = np.hstack([[x], lams] if x < 0.1 else [lams, [x]])
    #         mdvdl = np.hstack([[val], mdvdl] if x < 0.1 else [mdvdl, [val]])
    #         print(f'Extrapolated at {x}')
    # if iplt:
    #     f = go.Figure(go.Scatter(x=lams, y=mdvdl))
    #     f.update_xaxes(title_text='lambda')
    #     f.update_yaxes(title_text='dV/dlambda')
    #     f.update_layout(font=dict(size=20))
    #     f.show()
    if gauss:
        aa, w = np.polynomial.legendre.leggauss(len(mdvdl))
        w = w / 2
        integ = sum(mdvdl * w)
    else:
        integ = np.trapz(mdvdl, x=lams)
    if iplt or sfig:
        plt.plot(lams, mdvdl, marker='o')
        plt.title(f'dG={integ}')
    if iplt:
        plt.show()
    if sfig:
        plt.savefig(fname)
    print(integ)

    return lams, mdvdl, integ, vals

def equil_rmsd(ftraj='0_equil/step9.nc', ifrms=arange(0, 20, 5),rstfle='../rand.rst7', prmfle='../../solv_out/full.parm7', nres=131):
    '''ifrm: which frames to investigate
    ftraj: trajectory file'''
    ref = pt.load(rstfle, prmfle)
    fig, ax = plt.subplots()
    m = pt.load(ftraj, prmfle).autoimage('origin')
    m.rmsfit(frame_indices=ifrms, ref=ref[0], mask=f':1-130&!@H=')
    aa = pt.rmsd(m, ref=ref, mask=[f':{x}' for x in range(1, 132)], frame_indices=ifrms, nofit=True)
    ax.plot(aa)
    # ref[f':1-{nres}'].save(f'out_rmsd/ref.pdb')

    # for i in ifrms:
    #     pt.write_traj(f'frames_{i}.pdb', m[f':1-{nres}'], frame_indices=[i], overwrite=True)
    ax.set_xlabel('Residue number')
    ax.set_ylabel('RMSD')
    ax.legend([f'{x*100+500} ps' for x in ifrms])
    fig.savefig(f'fig.jpeg')


def get_rmsd(fle, gauss=1, fout='', sstp=0, nme=''):
    #0.43738 0.56262 0.31608 0.20634 0.11505 0.04794 0.00922 0.68392 0.79366 0.88495 0.95206 0.99078
    t1 = "_".join(fle.split(os.sep)[-3:-1])
    t2 = "_".join(fle.split(os.sep)[-3:])
    t3 = fle.split(os.sep)[-1]
    pthparm = '../solv_out/full.parm7' if 'fwrd' in fle else f'{"" if sstp else "../"}../../../1_fwrd/2_comp/solv_out/full.parm7'
    ref = pt.load('0_equil/step9.rst7' if 'fwrd' in fle else f'{"" if sstp else "../"}../../../1_fwrd/2_comp/{t3}/ti.rst7', pthparm)
    fig, ax = plt.subplots()
    fname = f'{"" if sstp else "../"}../../../out_rmsd/'

    ref[':1-132'].save(f'{fname}{t1}_ref.pdb')
    for i, y in enumerate(nme):
        m = pt.load(f'{y}/run.nc', pthparm, frame_indices=[-1])
        # pt._verbose()
        m.rmsfit(ref=ref[0], mask=f':1-130&!@H=')
        aa = pt.rmsd(m, ref=ref, mask=[f':{x}' for x in range(1, 133)], frame_indices=[len(m) - 1], nofit=True)
        ax.plot(aa, label=f'lam-{y[4:]}')
        fout.write(f'ligands rmsd: {aa[-2][0]:8.6f}, {aa[-1][0]:8.6f}\n')
        # m[':1-132'].save(f'{fname}{t2}_{y}.pdb')
    ax.set_xlabel('Residue number')
    ax.set_ylabel('RMSD')
    # m = pt.load(f'0_equil/step9.nc', '../solv_out/full.parm7', frame_indices=[-1])
    # m.rmsfit(ref=ref[0], mask=f':1-130&!@H=')
    # m[':1-132'].save(f'{fname}{t}_equil.pdb')
    # aa = pt.rmsd(m, ref=ref, mask=[f':{x}' for x in range(1, 133)], frame_indices=[len(m) - 1], nofit=True)
    # ax.plot(aa, label='equil')
    ax.legend(ncol=2)
    fig.savefig(f'{fname}{t2}.jpeg')


def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

def adjust_coords2(ln1, ln2, vac='vac.pdb', sel1='', sel2='', tol=.3, cpdb1=0, cpdb2=0, mol2_1='', mol2_2='', pdb1='', pdb2=''):
    '''adjust coordinates of atoms with the same name

    sel1: three atoms from the ligand you want to move
    sel2: three atoms from the ligand to move to
    vac: initial structure that has the ligand and the protein
    mol2_1: mol2 file for lig1
    mol2_2: mol2 file for lig2
    cpdb1: adjust ligand coords based on pdb files that only have the ligand
    cpdb2: adjust ligand coords based on pdb files that have both the ligand and the protein

    Return:
        lig1, lig2: parmed structures for ligand 1 and ligand2    '''

    if isinstance(ln1, str):
        lig1 = pd.load_file(mol2_1 if mol2_1 else f'{ln1}_0.mol2', structure=True)
    if isinstance(ln2, str):
        lig2 = pd.load_file(mol2_2 if mol2_2 else f'{ln2}_0.mol2', structure=True)
    m, pdb = [lig1, lig2], []
    if cpdb1:
        pdb.append(pd.load_file(pdb1 if pdb1 else f'{ln1}.pdb', structure=True))
        pdb.append(pd.load_file(pdb2 if pdb2 else f'{ln2}.pdb', structure=True))
    if cpdb2:
        pdb.append(pd.load_file(f'init_{ln1}.pdb')[f':{ln1}'])
        pdb.append(pd.load_file(f'init_{ln2}.pdb')[f':{ln2}'])

    if cpdb1 or cpdb2:
        for i in range(2):
            coords = np.zeros(m[i].coordinates.shape)
            for j, atom in enumerate(m[i].atoms):
                coords[j] = pdb[i][f'@{atom.name}'].coordinates[0]
            m[i].coordinates = coords

    comp = pd.load_file(vac, structure=True)
    lig1 = move_ligand(lig1, comp, sel1, sel2)
    lig2 = move_ligand(lig2, comp, sel1, sel2)

    lig = lig1 + lig2

    for x in lig2.atoms:
        out = lig[f':1@{x.name}<@{tol}'].atoms
        if len(out) == 2:
            if out[0].name != out[1].name:
                print(f'{out[0].name} is close to {out[1].name} but names are different ! dist: {norm(array([out[0].xx, out[0].xy, out[0].xz])-array([out[1].xx, out[1].xy, out[1].xz]))}')
            atom = lig1[f'@{x.name}'].atoms[0]
            x.xx = atom.xx
            x.xy = atom.xy
            x.xz = atom.xz
        else:
            print(f'{x.name} does not have a counterpart')

    lig1.save(f'{ln1}_1.mol2', overwrite=True)
    lig2.save(f'{ln2}_1.mol2', overwrite=True)
    return lig1, lig2

def adjust_coords(m, nres, sep=.9, tol=.8):
    '''resolve close contact between atoms
    sep: desired separation'''

    m.top.set_reference(m[0])
    atoms = np.array(list(m.top.atoms))
    coords = m.xyz[0]
    for ii in m.top.select(f':{nres},{nres + 1}'):
        x = atoms[ii]
        try:
            # top1 = m[f':{x.resid + 1}@{x.name}<@0.9'][f'!@{x.name}']
            tmp = atoms[m.top.select(f':{x.resid + 1}@{x.name}<@{tol}')]
            # at1 =   # any atom that's not one of the linearly transformed ones from the ligand
            # at2 = [g for g in tmp if g.name == x.name]
            for y in [g for g in tmp if g.name != x.name]:  # for each atom close to the atom of the ligand
                dr = coords[x.index] - coords[y.index]
                nrm = norm(dr)
                # if nrm > tol:
                    # print (nrm)
                if np.isclose(nrm, 0):
                    ndr = np.array([1, 0, 0])
                else:
                    ndr = dr / nrm
                vadd = sep - nrm
                coords[y.index] -= ndr * vadd
                print(f'{x.name}--{y.name}:', nrm, norm(coords[x.index] - coords[y.index]))
        except:
            continue

def recover_homogenous_affine_transformation(p, p_prime):
    '''
    Find the unique homogeneous affine transformation that
    maps a set of 3 points to another set of 3 points in 3D
    space:

        p_prime == np.dot(p, R) + t

    where `R` is an unknown rotation matrix, `t` is an unknown
    translation vector, and `p` and `p_prime` are the original
    and transformed set of points stored as row vectors:

        p       = np.array((p1,       p2,       p3))
        p_prime = np.array((p1_prime, p2_prime, p3_prime))

    The result of this function is an augmented 4-by-4
    matrix `A` that represents this affine transformation:

        np.column_stack((p_prime, (1, 1, 1))) ==
            np.dot(np.column_stack((p, (1, 1, 1))), A)

    Source: https://math.stackexchange.com/a/222170 (robjohn)
    '''

    # construct intermediate matrix
    Q       = p[1:]       - p[0]
    Q_prime = p_prime[1:] - p_prime[0]

    # calculate rotation matrix
    R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))),
               np.row_stack((Q_prime, np.cross(*Q_prime))))

    # calculate translation vector
    t = p_prime[0] - np.dot(p[0], R)

    # calculate affine transformation matrix
    # return np.column_stack((np.row_stack((R, t)),
    #                         (0, 0, 0, 1)))
    return R, t

# def euler_fit():
# from scipy.optimize import minimize, shgo

##    get rotation matrix by fitting euler angles
#     def func(x, c1, c2):
#         return np.sum((c1 @ Rotation.from_euler('xyz', x).as_matrix().T - c2) ** 2)
#
#     c1, c2 = lig['@O1,N1,C1'].coordinates, prot[':131@O30,N36,C32'].coordinates
#     ca, cb = np.mean(lig['@O1,N1,C1'].coordinates, 0), np.mean(prot[':131@O30,N36,C32'].coordinates, 0)
#     c1 -= ca
#     c2 -= cb
#     # minimize(func, [0, 0, 0], args=(c1, c2), bounds=((0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)))
#     angs = shgo(func, ((0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi)), args=(c1, c2)).x
#     R = Rotation.from_euler('xyz', angs).as_matrix()


def move_ligand(lig, base, ligmask, basemask):
    '''lig, base: parmed structure of ligand and the base protein
    ligmask: three atoms of the ligand (ambermask string)
    basemask: three atoms of the base protein (ambermask string)'''
    from scipy.spatial.transform import Rotation
    if isinstance(lig, str):
        lig = pd.load_file(lig, structure=True)
    if isinstance(base, str):
        base = pd.load_file(base)
    bc, lc = [], []
    for x, y in zip(re.findall('(?<=@).*', basemask)[0].split(','), re.findall('(?<=@).*', ligmask)[0].split(',')):
        bc.append(base['@'+x].coordinates[0])
        lc.append(lig['@'+y].coordinates[0])
    lc, bc = array(lc), array(bc)

    R, t = rigid_transform_3D(lc.T, bc.T)
    lig.coordinates = lig.coordinates @ R.T + t.T
    return lig

def neighbs_nonp(set1, set2=None, rcut=2, slf=0):
    coords_set1 = set1.coordinates
    coords_set2 = set2.coordinates

    idx = [[] for i in range(len(coords_set1))]
    odists = deepcopy(idx)
    for cnt, coord in enumerate(coords_set1):
        dists = norm(coord - coords_set2, axis=1)
        tmp = dists < rcut
        for x, y in zip(compress(set2, tmp), compress(dists, tmp)):
            if slf or set1[cnt] != x:
                idx[cnt].append(x)
                odists[cnt].append(y)
    return idx, odists


def snd_gauss(snd, n):
    '''only for when n is even'''
    l, _ = np.polynomial.legendre.leggauss(n)
    l = (l + 1) / 2

    snd = snd.replace('lams_list', f"{' '.join([f'{x:5f}' for x in l[n // 2 - 1::-1]])} {' '.join([f'{x:5f}' for x in l[n // 2:]])}")
    # txt = ''
    snd = snd.replace('mid1', f'{l[n//2-1]:5f}').replace('mid2', f'{l[n//2]:5f}')
    # idx = arange(n)
    #
    # for x, i in zip(l[n//2-2:0:-1], idx[n//2-2:0:-1]):
    #     txt += f'elif ["$x" == "{x:5f}"];then t = "../lam_{l[i+1]:5f}/init.rst7\n'
    #
    # for x, i in zip(l[n//2+1:], idx[n//2+1:]):
    #     txt += f'elif ["$x" == "{x:5f}"];then t = "../lam_{l[i-1]:5f}/init.rst7\n'

    return snd



eq_dan = '''#!/bin/bash

# Minimization for explicit solvent, The protocol told by Dan 

TOP='pthparm'
CRD='pthrand'
S=nres # Number of solute residues (assumes start at 1)
# Solute backbone mask
BACKBONEMASK=":1-$S@H,N,CA,HA,C=,O" # Protein

T=393.15 # Temperature

cat > step1.in <<EOF
Min explicit solvent relaxed heavy atom rest no shake 
 &cntrl
   imin = 1, ntmin = 2, maxcyc = 10000,ntxo=1,
   ntwx = 500, ioutfm = 1, ntpr = 500, 
   ntc = 1, ntf = 1, ntb = 1, cut = 8.0,  
   igb = 0, saltcon = 0.0, clambda = 0.5,
   ntr = 1, restraintmask = ':1-$S & !@H=', restraint_wt = 2.0,icfe = 1, 
   reptxt    
 &end
EOF

cat > step1.in <<EOF
MD explicit solvent heavy atom rest shake dt 0.001
 &cntrl
   imin = 0, nstlim = 15000, dt=0.001, 
   ntx = 1, irest = 0, ig = -1,vlimit=.05,
   ntwx = 5000, ioutfm = 1, ntpr = 5000, 
   iwrap = 1, nscm = 0,
   ntc = 1, ntf = 1, ntb = 1, cut = 8.0,  
   ntt = 1, tautp = 0.5, temp0 = $T, tempi = $T,
   ntp = 0, taup = 0.5,
   igb = 0, saltcon = 0.0,icfe = 1,
   ntr = 1, restraintmask = ':1-$S & !@H=', restraint_wt = 5.0, clambda = 0.5,
   reptxt     
 &end
EOF

# NTV MD with strong restraints on heavy atoms, shake, dt=.001, 15 ps
cat > step2.in <<EOF
MD explicit solvent heavy atom rest shake dt 0.001
 &cntrl
   imin = 0, nstlim = 1500, dt=0.001, 
   ntx = 1, irest = 0, ig = -1,
   ntwx = 500, ioutfm = 1, ntpr = 500,  
   iwrap = 1, nscm = 0,
   ntc = 1, ntf = 1, ntb = 1, cut = 8.0,  
   ntt = 1, tautp = 0.5, temp0 = 1, tempi = 1,
   ntp = 0, taup = 0.5,
   igb = 0, saltcon = 0.0,icfe = 1,
   ntr = 1, restraintmask = ':1-$S & !@H=', restraint_wt = 5.0, clambda = 0.5,
   reptxt   
 &end
EOF

cat > step2.in <<EOF
MD explicit solvent heavy atom rest shake dt 0.001
 &cntrl
   imin = 0, nstlim = 50000, dt=0.001, 
   ntx = 1, irest = 0, ig = -1,vlimit=.05,
   ntwx = 5000, ioutfm = 1, ntpr = 5000,
   iwrap = 1, nscm = 0,
   ntc = 1, ntf = 1, ntb = 1, cut = 8.0,  
   ntt = 1, tautp = 0.5, temp0 = $T, tempi = $T,
   ntp = 0, taup = 0.5,
   igb = 0, saltcon = 0.0,icfe = 1,
   ntr = 1, restraintmask = ':1-$S', restraint_wt = 5.0, clambda = 0.5,
   reptxt 
 &end
EOF

# Steepest Descent Minimization with relaxed restraints on heavy atoms, no shake
cat > step3.in <<EOF
Min explicit solvent relaxed heavy atom rest no shake 
 &cntrl
   imin = 1, ntmin = 2, maxcyc = 1000,ntxo=1,
   ntwx = 500, ioutfm = 1, ntpr = 500, 
   ntc = 1, ntf = 1, ntb = 1, cut = 8.0,  
   igb = 0, saltcon = 0.0, clambda = 0.5,
   ntr = 1, restraintmask = ':1-$S & !@H=', restraint_wt = 2.0,icfe = 1, 
   reptxt     
 &end
EOF

# Steepest Descent Minimization with minimal restraints on heavy atoms, no shake
cat > step4.in <<EOF
Min explicit solvent minimal heavy atom rest no shake 
 &cntrl
   imin = 1, ntmin = 2, maxcyc = 1000,ntxo=1,
   ntwx = 500, ioutfm = 1, ntpr = 500, 
   ntc = 1, ntf = 1, ntb = 1, cut = 8.0,  
   igb = 0, saltcon = 0.0, clambda = 0.5,
   ntr = 1, restraintmask = ':1-$S & !@H=', restraint_wt = 0.1,icfe = 1,
   reptxt   
 &end
EOF

# Steepest Descent Minimization with no restraints, no shake
cat > step5.in <<EOF
Min explicit solvent no heavy atom res no shake 
 &cntrl
   imin = 1, ntmin = 2, maxcyc = 1000,ntxo=1,
   ntwx = 500, ioutfm = 1, ntpr = 500, 
   ntc = 1, ntf = 1, ntb = 1, cut = 8.0, 
   igb = 0, saltcon = 0.0, clambda = 0.5,
   ntr = 0, icfe = 1,
   reptxt  
 &end
EOF

# NTP MD with shake and low restraints on heavy atoms, 5 ps dt=.001
cat > step6.in <<EOF
MD explicit solvent heavy atom low rest shake dt 0.001
 &cntrl
   imin = 0, nstlim = 5000, dt=0.001, ntxo=1,
   ntx = 1, irest = 0, ig = -1,
   ntwx = 500, ioutfm = 1, ntpr = 500, 
   iwrap = 1, nscm = 0,
   ntc = 1, ntf = 1, ntb = 2, cut = 8.0,  
   ntt = 1, tautp = 1.0, temp0 = $T, tempi = $T,
   ntp = 1, taup = 1.0,
   igb = 0, saltcon = 0.0,icfe = 1, clambda = 0.5,
   ntr = 1, restraintmask = ':1-$S & !@H=', restraint_wt = 1.0,
   reptxt
 &end
EOF

# NTP MD with shake and minimal restraints on heavy atoms
cat > step7.in <<EOF
MD explicit solvent heavy atom minimal rest shake dt 0.001, 1 ns
 &cntrl
   imin = 0, nstlim = 100000, dt=0.001, ntxo=1,
   ntx = 5, irest = 1, 
   ntwx = 10000, ioutfm = 1, ntpr = 5000, 
   iwrap = 1, nscm = 0,
   ntc = 2, ntf = 1, ntb = 2, cut = 8.0,  
   ntt = 1, tautp = 1.0, temp0 = $T, tempi = $T,
   ntp = 1, taup = 1.0,
   igb = 0, saltcon = 0.0, icfe = 1, clambda = 0.5,
   ntr = 1, restraintmask = ':1-$S & !@H=', restraint_wt = 0.5,
   reptxt
 &end
EOF

# NTP MD with shake and minimal restraints on backbone atoms, dt=0.001
cat > step8.in <<EOF
MD explicit solvent heavy atom minimal BB rest shake dt 0.001, 1ns
 &cntrl
   imin = 0, nstlim = 100000, dt=0.001, ntxo=1,
   ntx = 5, irest = 1, 
   ntwx = 10000, ioutfm = 1, ntpr = 5000, 
   iwrap = 1, nscm = 0,
   ntc = 1, ntf = 1, ntb = 2, cut = 9.0,
   ntt = 3, gamma_ln = 1, temp0 = $T, tempi = $T,
   ntp = 1, barostat = 2, taup = 5.0,
   icfe = 1, clambda = 0.5,
   ntr = 1, restraintmask = "$BACKBONEMASK", restraint_wt = 0.5,
   reptxt   
 &end
EOF

# NTP MD with shake and no restraints, dt=0.002
cat > step9.in <<EOF
MD explicit solvent heavy atom no rest shake dt 0.002, 1ns
 &cntrl
   imin = 0, nstlim = 100000, dt=0.001, ntxo=1,
   ntx = 5, irest = 1, 
   ntwx = 10000, ioutfm = 1, ntpr = 5000,  
   iwrap = 1, nscm = 1000,
   ntc = 1, ntf = 1, ntb = 2, cut = 9.0,
   ntt = 3, gamma_ln = 1, temp0 = $T, tempi = $T,
   ntp = 1, barostat = 2, taup = 5.0,
   clambda = 0.5, icfe = 1,
   reptxt
 &end
EOF

# Minimization Phase
for RUN in step1 step2 step3 step4 step5;do
  echo "  $RUN" 
  pmemd.cuda -O -i $RUN.in -p $TOP -c $CRD -ref $CRD -o $RUN.out -x $RUN.nc -r $RUN.rst7 -inf $RUN.mdinfo      
  CRD="$RUN.rst7"
done


# Equilibration phase - reference coords are last coords from minimize phase
REF="$CRD"
for RUN in step6 step7 step8 step9;do 
  echo "  $RUN" 
  pmemd.cuda -O -i $RUN.in -p $TOP -c $CRD -ref $REF -o $RUN.out -x $RUN.nc -r $RUN.rst7 -inf $RUN.mdinfo
  CRD="$RUN.rst7"
done
rm *info'''

leq_dan = '''#!/bin/bash

# Minimization for explicit solvent, The protocol told by Dan 

TOP='pthparm'
CRD='pthrand'
S=nres # Number of solute residues (assumes start at 1)
# Solute backbone mask
BACKBONEMASK=":1-$S@H,N,CA,HA,C=,O" # Protein

T=393.15 # Temperature

cat > step1.in <<EOF
Min explicit solvent relaxed heavy atom rest no shake 
 &cntrl
   imin = 1, ntmin = 2, maxcyc = 1000,ntxo=1,
   ntwx = 500, ioutfm = 1, ntpr = 500, 
   ntc = 1, ntf = 1, ntb = 1, cut = 8.0,  
   igb = 0, saltcon = 0.0, clambda = 0.5,
   ntr = 1, restraintmask = ':1-$S & !@H=', restraint_wt = 2.0,icfe = 1, 
   reptxt    
 &end
EOF

# NTP MD with shake and no restraints, dt=0.002
cat > step9.in <<EOF
MD explicit solvent heavy atom no rest shake dt 0.002, 1ns
 &cntrl
   imin = 0, nstlim = 100000, dt=0.001, ntxo=1,
   ntx = 1, irest = 0, ig=-1,
   ntwx = 100000, ioutfm = 1, ntpr = 50000, 
   iwrap = 1, nscm = 1000,
   ntc = 1, ntf = 1, ntb = 2, cut = 9.0,
   ntt = 3, gamma_ln = 1, temp0 = $T, tempi = $T,
   ntp = 1, barostat = 2, taup = 5.0,
   icfe = 1, clambda=.5,
   reptxt
 &end
EOF

REF="$CRD"
for RUN in step1 step9;do 
  echo "  $RUN" 
  pmemd.cuda -O -i $RUN.in -p $TOP -c $CRD -ref $REF -o $RUN.out -x $RUN.nc -r $RUN.rst7 -inf $RUN.mdinfo
  CRD="$RUN.rst7"
done
rm *info'''

snd ='''#!/bin/bash

#SBATCH --job-name=jnme      ## Name of the job. (-J)
#SBATCH -p free-gpu         ## partition/queue name
#SBATCH --nodes=1            ## (-N) number of nodes to use
#SBATCH --ntasks=1           ## (-n) number of tasks to launch
#SBATCH --cpus-per-task=1    ## number of cores the job needs
#SBATCH --output=out --error=err ## (-o and -e)
#SBATCH --gres=gpu:V100:1


cd 0_equil
rm step*
./eq-dan.sh
cd ..
#`seq 0.0 0.05 1.00`
for x in lams_list;do 
  fle="lam_$x"
  echo $fle
  mkdir $fle
  cd $fle

  cat > run.in <<cmd
 &cntrl
   imin = 0, nstlim = nsteps, dt=0.001,
   irest = 1, ntx = 5, ig = -1,
   ntwx = framefreq, ioutfm = 1, ntpr = printfreq, ntxo = 1,
   iwrap = 1, nscm = 1000,tol=0.0000001,
   ntc = 1, ntf = 1, ntb = 2, cut = 9.0,
   ntt = 3, gamma_ln = 1, temp0 = 393.15, tempi = 393.15,
   ntp = 1, barostat = 2, taup = 5.0,
   icfe = 1, clambda = $x, 
   reptxt 
 /
cmd
  cat > run1.in <<cmd
 &cntrl
   imin = 0, nstlim = 10000, dt=0.001,
   irest = 1, ntx = 5, ig = -1,
   ntwx = framefreq, ioutfm = 1, ntpr = printfreq, ntxo = 1,
   iwrap = 1, nscm = 1000,tol=0.0000001,
   ntc = 1, ntf = 1, ntb = 2, cut = 9.0,
   ntt = 3, gamma_ln = 1, temp0 = 393.15, tempi = 393.15,
   ntp = 1, barostat = 2, taup = 5.0,
   icfe = 1, clambda = $x, 
   reptxt 
 /
cmd
  if [ "$x" == "mid1" ] || [ "$x" == "mid2" ];then
    t='../0_equil/step9.rst7'    
  else
    t="../lam_${tmp}/init.rst7"
  fi
  tmp=$x
  pmemd.cuda -O -i run1.in -o run.out -p ../../solv_out/full.parm7 -c $t -r init.rst7 -x run.nc -inf run.mdinfo 
  pmemd.cuda -O -i run.in -o run.out -p ../../solv_out/full.parm7 -c $t -r res.rst7 -x run.nc -inf run.mdinfo 
  sleep 1
  rm *info run.in run1.in res.rst7

  #if [ "$x" != "0.00922" ] && [ "$x" != "0.99078" ];then
  #  rm run.nc res.rst7
  #fi
  cd ..
done
for x in lam*;do
rm ${x}/init*
done'''


snd_trapz='''#!/bin/bash

#SBATCH --job-name=jnme      ## Name of the job. (-J)
#SBATCH -p free-gpu         ## partition/queue name
#SBATCH --nodes=1            ## (-N) number of nodes to use
#SBATCH --ntasks=1           ## (-n) number of tasks to launch
#SBATCH --cpus-per-task=1    ## number of cores the job needs
#SBATCH --output=out --error=err ## (-o and -e)
#SBATCH --gres=gpu:V100:1


cd 0_equil
rm step*
./eq-dan.sh
cd ..
for x in `seq 0.50 -0.05 0.05` 0.00 `seq 0.55 .05 1.00`;do 
  fle="lam_$x"
  echo $fle
  mkdir $fle
  cd $fle

  cat > run.in <<cmd
 &cntrl
   imin = 0, nstlim = nsteps, dt=0.001,
   irest = 1, ntx = 5, ig = -1,
   ntwx = framefreq, ioutfm = 1, ntpr = printfreq, ntxo = 1,
   iwrap = 1, nscm = 1000,tol=0.0000001,
   ntc = 1, ntb = 1, cut = 8.0,
   ntt = 3, gamma_ln = 1, temp0 = 393.15, tempi = 393.15,
   icfe = 1, clambda = $x, 
   reptxt 
 /
cmd
  cat > run1.in <<cmd
 &cntrl
   imin = 0, nstlim = 10000, dt=0.001,
   irest = 1, ntx = 5, ig = -1,
   ntwx = framefreq, ioutfm = 1, ntpr = printfreq, ntxo = 1,
   iwrap = 1, nscm = 1000,tol=0.0000001,
   ntc = 1, ntb = 1, cut = 8.0,
   ntt = 3, gamma_ln = 1, temp0 = 393.15, tempi = 393.15,
   icfe = 1, clambda = $x, 
   reptxt 
 /
cmd
  if [ "$x" == "0.55" ] || [ "$x" == "0.50" ] || [ "$x" == "0.45" ];then
    t='../0_equil/step9.rst7'    
  else
    t="../lam_${tmp}/init.rst7"
  fi
  tmp=$x
  pmemd.cuda -O -i run1.in -o run.out -p ../../solv_out/full.parm7 -c $t -r init.rst7 -x run.nc -inf run.mdinfo 
  pmemd.cuda -O -i run.in -o run.out -p ../../solv_out/full.parm7 -c $t -r res.rst7 -x run.nc -inf run.mdinfo 
  sleep 1
  rm *info run.in run1.in res.rst7

  cd ..
done'''


snd_gauss_bak='''#!/bin/bash

#SBATCH --job-name=jnme      ## Name of the job. (-J)
#SBATCH -p free-gpu         ## partition/queue name
#SBATCH --nodes=1            ## (-N) number of nodes to use
#SBATCH --ntasks=1           ## (-n) number of tasks to launch
#SBATCH --cpus-per-task=1    ## number of cores the job needs
#SBATCH --output=out --error=err ## (-o and -e)
#SBATCH --gres=gpu:V100:1


cd 0_equil
rm step*
./eq-dan.sh
cd ..
#`seq 0.0 0.05 1.00`
for x in 0.43738 0.56262 0.31608 0.20634 0.11505 0.04794 0.00922 0.68392 0.79366 0.88495 0.95206 0.99078;do 
  fle="lam_$x"
  echo $fle
  mkdir $fle
  cd $fle

  cat > run.in <<cmd
 &cntrl
   imin = 0, nstlim = nsteps, dt=0.001,
   irest = 1, ntx = 5, ig = -1,
   ntwx = framefreq, ioutfm = 1, ntpr = printfreq, ntxo = 1,
   iwrap = 1, nscm = 1000,tol=0.0000001,
   ntc = 1, ntb = 1, cut = 8.0,
   ntt = 3, gamma_ln = 1, temp0 = 393.15, tempi = 393.15,
   icfe = 1, clambda = $x, 
   reptxt 
 /
cmd
  cat > run1.in <<cmd
 &cntrl
   imin = 0, nstlim = 10000, dt=0.001,
   irest = 1, ntx = 5, ig = -1,
   ntwx = framefreq, ioutfm = 1, ntpr = printfreq, ntxo = 1,
   iwrap = 1, nscm = 1000,tol=0.0000001,
   ntc = 1, ntb = 1, cut = 8.0,
   ntt = 3, gamma_ln = 1, temp0 = 393.15, tempi = 393.15,
   icfe = 1, clambda = $x, 
   reptxt 
 /
cmd
  if [ "$x" == "0.43738" ] || [ "$x" == "0.56262" ];then
    t='../0_equil/step9.rst7'    
  elif [ "$x" == "0.31608" ]; then
    t='../lam_0.43738/init.rst7'
  elif [ "$x" == "0.20634" ]; then
    t='../lam_0.31608/init.rst7'
  elif [ "$x" == "0.11505" ]; then
    t='../lam_0.20634/init.rst7'
  elif [ "$x" == "0.04794" ]; then
    t='../lam_0.11505/init.rst7'
  elif [ "$x" == "0.00922" ]; then
    t='../lam_0.04794/init.rst7'
  elif [ "$x" == "0.68392" ]; then
    t='../lam_0.56262/init.rst7'
  elif [ "$x" == "0.79366" ]; then
    t='../lam_0.68392/init.rst7'
  elif [ "$x" == "0.88495" ]; then
    t='../lam_0.79366/init.rst7'
  elif [ "$x" == "0.95206" ]; then
    t='../lam_0.88495/init.rst7'
  elif [ "$x" == "0.99078" ]; then
    t='../lam_0.95206/init.rst7'
  fi
  pmemd.cuda -O -i run1.in -o run.out -p ../../solv_out/full.parm7 -c $t -r init.rst7 -x run.nc -inf run.mdinfo 
  pmemd.cuda -O -i run.in -o run.out -p ../../solv_out/full.parm7 -c $t -r res.rst7 -x run.nc -inf run.mdinfo 
  sleep 1
  rm *info run.in run1.in res.rst7

  #if [ "$x" != "0.00922" ] && [ "$x" != "0.99078" ];then
  #  rm run.nc res.rst7
  #fi
  cd ..
done'''