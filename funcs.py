import numpy as np
from numpy.linalg import inv, norm, det, matrix_rank, svd, eig
from more_itertools import sliced
from multiprocessing import Pool
import sys
from shutil import copyfile, rmtree
from time import time, sleep
# from plotly import graph_objects as go
from plotly.offline import plot
from itertools import islice
import os, subprocess, math, io, fnmatch
from glob import glob
# from lammps import lammps
from numpy import dot, cross, sqrt, einsum, pi, arange, linspace, piecewise, exp, zeros, ones, mean
from numpy import logical_not as lnot
from math import degrees, radians
from collections.abc import Iterable
from collections import Counter, UserDict, defaultdict
from collections import OrderedDict as odict
from indexed import IndexedOrderedDict as iodict
from pathlib3x import Path
# from parmed import unit as u
from subprocess import check_output
# from molmod.ic import bond_length, bend_angle, dihed_angle
from itertools import combinations, chain, product, compress
# from scipy.spatial.transform import Rotation as R
from warnings import warn
from operator import itemgetter
# from lib.recipes.alkane import Alkane
from copy import deepcopy, copy
import regex as re
import importlib.util
import dill
from joblib import Parallel, delayed
from orderedset import OrderedSet as oset
# from pprofile import Profile
from operator import attrgetter as atget
from numpy import pi, cos, sin, arccos, arange, log, array
from scipy.special import erfc
from numpy import logical_not as lnot
from getpass import getuser as gu
from shortuuid import uuid
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.integrate as integrate
# from statsmodels.graphics.tsaplots import plot_acf

hpc = gu() == 'morsheda'

def mkch(nme):
    os.makedirs(nme, exist_ok=1)
    os.chdir(nme)

def find_nearest(array, value):
    'find the number is array that"s closest to value'
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def module_from_file(file_path):
    pth = re.findall('.*/', file_path)[0]
    cwd = os.getcwd()
    os.chdir(pth)
    
    spec = importlib.util.spec_from_file_location('', file_path.replace(pth, ''))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.chdir(cwd)
    return module

def sub_params2(ifile='gulp.in', ofile='gulp.out', fout='new_gulp.in', params=None):
    '''the flags should be in the form \d \d\n'''

    nparams, types = gulp_output_params(open(ofile))
    nparams = params if params is not None else nparams

    ev = 0

    inp = open(ifile).readlines()
    cnt, out = 0, ''

    for x in inp:
        if bool(flg := re.search(r'\d \d\n', x)):
            flg = txt_to_mat(flg.group(), int)
            for i, y in enumerate(x.split()):
                if not bool(re.search(r'[A-Za-z]', y)) or bool(re.search(r'\dE', y)):
                    icol = i + 1
                    break
            for i, iflg in enumerate(flg):
                if iflg == 1:
                    x = repnth(x, icol+i, ev_to_kcalpmol(nparams[cnt][1]) if ev else nparams[cnt][1])
                    cnt += 1
        out += x
    open(fout, 'w').write(out)



def sub_params(ifile='gulp.in', ofile='gulp.out', out='new_gulp.in', params=None):
    '''after you get fit output in gulp, make a new input file from fitted params'''

    nparams, types = gulp_output_params(open(ofile))
    nparams = params if params is not None else nparams

    equiv = ['Harmonic k2', 'Harmonic r0', 'Species epsilon', 'Species sigma', 'Lennard sigma', 'Lennard eps', 'Three-body cnst','Three-body angl']
    idx = [0, 1, 0, 1, 1, 0, 0, 1]  # which column the parameter is in input file. e.g: r0 is the second
    col = [3, 4, 3, 4, 4, 3, 5, 4]
    ev = [0, 0, 0, 0, 0, 0, 0, 0]
    search = ['harmonic', 'harmonic', 'epsilon', 'epsilon', 'lenn', 'lenn', 'three','three']


    inp = open(ifile).readlines()
    flg1 = 0
    # lines = {}
    nadv = np.zeros(len(equiv), dtype=int)  #how much the parameter has advanced
    for jj, (par, t) in enumerate(zip(nparams, types)):
        # f = open(ifile)
        for i, x in enumerate(equiv):
            if x in t:
                ii = i
                break

        p1, p2 = equiv[ii], search[ii]

        for i, x in enumerate(inp):

            if p2 in x:
                cnt = 0
                for x in inp[i+1:]:
                    # i += 1
                    x = re.sub('#.*', '', x)
                    if bool(flgs := re.search('\d\s+\d\s*?\n', x)) or x.strip():
                        if not flgs:
                            continue
                        flgs = txt_to_mat(flgs.group(), int)
                        if flgs[idx[ii]] == 1:
                            if nadv[ii] == cnt:
                                inp[i] = repnth(x, col[ii], ev_to_kcalpmol(par[1]) if ev[ii] else par[1])
                                # lines[i] = inp[i]
                            cnt += 1
                    else:
                        break
        nadv[ii] += 1

        # func(equiv[ii], search[ii])
    open(out, 'w').write(''.join(inp))


def skip_blanks(f):
    ''' f: iterator, skip the blank lines '''
    for x in f:
        if x.strip():
            return x


def expand_box_pair(comp, rcut, idx=[]):
    ''' expand the box such that within rcut gives unique pair distances '''
    from compound import Compound
    b = np.linalg.inv(comp.latmat)

    if not idx:
        idx = np.ceil(rcut * np.apply_along_axis(norm, 1, b))
    idx = np.array(idx, dtype=int) + 1

    comp2 = Compound()
    comp2.latmat = comp.latmat
    xx, yy, zz = np.meshgrid(range(-idx[0] + 1, idx[0]), range(-idx[1] + 1, idx[1]), range(1, idx[2]))

    for x, y, z in zip(xx.flatten(), yy.flatten(), zz.flatten()):
        c = deepcopy(comp)
        c.xyz += np.sum(np.array([x, y, z])[:, np.newaxis] * c.latmat, 0)
        comp2.add(c)
    xx, yy = np.meshgrid(range(0, idx[0]), range(0, idx[1]))
    for x, y in zip(xx.flatten(), yy.flatten()):
        c = deepcopy(comp)
        c.xyz += np.sum(np.array([x, y, 0])[:, np.newaxis] * c.latmat, 0)
        comp2.add(c)
        if [x, y] == [0, 0]:
            porig = c.particles()
    xx, yy = np.meshgrid(range(-idx[0] + 1, 0), range(1, idx[1]))
    for x, y in zip(xx.flatten(), yy.flatten()):
        c = deepcopy(comp)
        c.xyz += np.sum(np.array([x, y, 0])[:, np.newaxis] * c.latmat, 0)
        comp2.add(c)
    comp2.latmat *= (idx[:, np.newaxis]-1)*2+1

    # nlst1 = porig
    # nlst2 = comp2.particles()
    # n1 = len(nlst1)
    # n2 = len(nlst2)
    # cnt = 0
    # if comp2.bond_graph:
    #     for x in comp2.bond_graph.edges:
    #         cnt += 1
    #         equivalent_parts = [nlst2[i] for i in range(nlst2.index(x[1]) % n1, n2, n1)]
    #         p = comp2.closest_img(x[0], equivalent_parts)
    #         if x[1] is not p:
    #             comp2.bond_graph.remove_edge(*x)
    #             comp2.bond_graph.add_edge(x[0], p)
        # comp2.gen_angs_and_diheds()

    return comp2, porig


def rm_bonds(a, b, p):
    '''a:typed b:types'''
    if not isinstance(p, Iterable):
        p = [p]
        
    typed = dict()
    types = []
    for y in a:
        if np.any([(x in y) for x in p]):
            typed[y] = b[list(a).index(y)]
            types.append(b[list(a).index(y)])
    return typed, types



def points_in_sphere():
    d = .12
    tmp = np.arange(-d, d, .0266*d/.1)
    x, y, z = np.meshgrid(tmp, tmp, tmp);
    idx = x**2 + y**2 + z**2 < d**2;
    return x[idx], y[idx], z[idx]


def dist_sphere(num_pts):

    indices = arange(0, num_pts, dtype=float) + 0.5

    phi = arccos(1 - 2 * indices / num_pts)
    theta = pi * (1 + 5 ** 0.5) * indices

    return cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi);#x, y, z =

    import plotly.graph_objects as go
    go.Figure(go.Scatter3d(x=x,y=y,z=z,mode='markers')).show()



def gulp_confs_to_xyz(direc='gulp.in'):
    out = shell(f'''awk '/cart/{{getline;flg=1}}NF<3{{flg=0}}flg{{print $1, $2, $3, $4}}' {direc}''').decode().split('\n')
    n = shell(f'''awk '/cart/{{flg=1;getline;n=NR}} flg && NF<3 {{print NR-n;flg=0}}' {direc}''').decode().split()
    with open('out.xyz', 'w') as f:
        for i, x in enumerate(n):
            f.write('\n'+x+'\n')
            f.write('comment\n')
            f.write('\n'.join(out[i*int(x):(i+1)*int(x)]))

def get_gulp_configs(direc='gulp.in'):
    f = iter(open(direc))
    flg = 0
    out = ''
    ret = []
    for x in f:
        if 'vector' in x:
            flg = 1
        if 'end' in x and flg:
            flg = 0
            ret.append(out+x)
            out = ''
        if flg:
            out += x
    return ret
    

def normalize(x):
    return x/norm(x)

def hybrid_struct():
    '''create bonding for relaxed structure coming from vasp'''
    from compound import compload
    comp = hybrid_silane_portlandite()
    comp.xyz_label_sorted = compload('/home/ali/ongoing_research/polymer4/45_hybrid_hexane/2_opt_allfree_prec/CONTCAR').xyz_label_sorted
    
    comp.generate_bonds(comp.particles_by_name('C'), comp.particles_by_name('Si'), 1.5, 1.9)
    comp.generate_bonds(comp.particles_by_name('Ca'), comp.particles_by_name('O'), 2.23, 2.54)
    comp.create_bonding_all(kb=0, ka=0, kd=0, acfpath='/home/ali/ongoing_research/polymer4/45_hybrid_hexane/4_charge/')
    rm = []
    for x in comp.propers_typed.keys():
        if [y for y in x if y.name == 'Ca']:
            rm.append(x)
    
    for x in rm:
        comp.ff.proper_types.remove(comp.propers_typed[x])
        comp.propers_typed.pop(x)
    return comp


def rot_ax_ang(ax, ang, vec, deg=1):
    from scipy.spatial.transform import Rotation as R
    ax = normalize(ax)
    if deg:
        ang = math.radians(ang)
    rot = R.from_rotvec(ax * ang)
    return rot.apply(vec)

def get_fitted(comp, direc='.'):
    pth = os.path.join(direc, 'gulp.in')
    tmp = check_output("""awk '/harm/||/three/{getline;flg=1}NF==0{flg=0}flg' """+pth+
                       """| awk '/harm/||/three/{getline;flg=1}NF==0{flg=0}flg' """+pth+""" |
                          grep -Po '(?<=\s)[0-9].*(?=\#)'""", shell=1)
    qlist = [*comp.bonds_typed, *comp.angles_typed, *comp.propers_typed]
    k, rest, flags = [],[],[]
    rest = deepcopy(k)
    for x, y in zip(tmp.decode().replace('\t','').split('\n')[:-1], qlist):
        x = x.split()
        k.append(x[0])
        rest.append(x[1])
        flags.append(x[-2:])

    pth = os.path.join(direc, 'gulp.out')
    tmp = check_output("""awk '/Parameter        P/{getline;getline;getline;flg=1}/-----/{flg=0}flg{print $3}'  """+pth, shell=1)
    # vals = np.genfromtxt(io.BytesIO(tmp))
    
    return k, rest, np.array(flags, dtype=int), tmp.decode().split('\n')[:-1]


def gulp_strs(fout):
    strs0 = []
    for x in fout.splitlines(True):
        if 'xx' in x or 'yy' in x or 'zz' in x:
            strs0.extend(array(get_cols(x, [2, 4]), dtype=float))
    return array(strs0)



def brace(inp, misc='eps sig bond angle'):
    tmp = inp.replace('core', '').splitlines(True)
    flg1, flg2, flg3 = 0, 0, 0
    pars0 = []
    thcols = array([4, 5])
    for i, x in enumerate(tmp):
        if 'lenn' in x:
            if flg1 or 'lenn' in tmp[i + 2]:
                pars0.extend(get_cols(tmp[i + 1], lencols+2))
                tmp[i + 1] = repnth(tmp[i + 1], lencols+2, len(lencols)*['{}'])
                flg1 = 1
            else:
                ii = 1
                while 1:
                    cols = re.split('\s+', tmp[i + ii])
                    if len(cols) < 6 or not bool(re.search('[A-DF-Za-z]', cols[0]) and
                                                 re.search('[A-DF-Za-z]', cols[1]) and not re.search('[A-DF-Za-df-z]', cols[2]) ):
                        break
                    pars0.extend(itemgetter(2, 3)(cols))
                    tmp[i + ii] = repnth(tmp[i + ii], [3, 4], ['{}', '{}'])
                    ii += 1
        if 'three' in x:
            if flg2 or 'three' in tmp[i + 2]:
                pars0.extend(get_cols(tmp[i + 1], thcols+3))
                tmp[i + 1] = repnth(tmp[i + 1], thcols+3, ['{}', '{}'])
                flg2 = 1
            else:
                ii = 1
                while 1:
                    cols = re.split('\s+', tmp[i + ii])
                    if len(cols) < 5 or not bool(re.search('[A-Za-z]', cols[0]) and re.search('[A-Za-z]', cols[1]) and
                                                 re.search('[A-Za-z]', cols[2]) and not re.search('[A-Za-df-z]', cols[3])):
                        break
                    pars0.extend(itemgetter(3, 4)(cols))
                    tmp[i + ii] = repnth(tmp[i + ii], [4, 5], ['{}', '{}'])
                    ii += 1
        if 'harm' in x:
            if flg3 or 'harm' in tmp[i + 2]:
                pars0.extend(get_cols(tmp[i + 1], lencols+2))
                tmp[i + 1] = repnth(tmp[i + 1], lencols+2, ['{}', '{}'])
                flg3 = 1
            else:
                ii = 1
                while 1:
                    cols = re.split('\s+', tmp[i + ii])
                    if len(cols) < 5 or not bool(re.search('[A-Za-z]', cols[0]) and re.search('[A-Za-z]', cols[1]) and
                                                 not re.search('[A-Za-df-z]', cols[2]) ):
                        break
                    pars0.extend(itemgetter(2, 3)(cols))
                    tmp[i + ii] = repnth(tmp[i + ii], [3, 4], ['{}', '{}'])
                    ii += 1
    return ''.join(tmp), array(pars0, dtype=float)


def get_cell(txt):
    tot = ''
    for i, x in enumerate(txt):
        if bool(re.search('cell\s', x)):
            tot += ' '.join(get_cols(txt[i+1], arange(1, 7))) + ' '
    return txt_to_mat(tot)

    
def shell(comm, inp=''):
    '''comm: command to run
    inp: a byte string to give as input'''
    try:
        o = check_output(comm, shell=1, input=inp.encode(), executable='/bin/bash') if inp else check_output(comm, shell=1, executable='/bin/bash')
    except subprocess.CalledProcessError as ex:
        o = ex.output

    return o.decode()

def amass(nme):
    ''' nme: a string with the name of the element '''
    import pymatgen.core.periodic_table as pt
    return pt.Element(nme).atomic_mass.real


def sort_data(fle='data.dat', fout='data_new.dat'):

    out = ''
    tmp = ''
    with open(fle) as f:
        for x in f:
            if 'Atoms' in x:
                next(f)
                out += 'Atoms\n\n'
                for y in f:
                    if not y.strip():
                        break
                    tmp += y
                tmp = txt_to_mat(tmp)
                tmp = tmp[tmp[:, 0].argsort()]
                for z in tmp:
                    out += '{} {} {} '.format(*array(z[:3], dtype=int)) + '{} {} {} {} \n'.format(*z[3:7])
                # out += mat_prt(tmp,frm='{:15.11e} ')
                x = '\n'
            out += x
    open(fout, 'w').write(out)



def dist(c1, c2):
    from compound import Compound
    if isinstance(c1, Compound):
        c1, c2 = c1.pos, c2.pos
    else:
        c1, c2 = map(np.array, [c1, c2])
    return norm(c1 - c2)

def modsem2(comp, outcar_pth, vibrational_scaling=1, ff_xml_pth=None):
    ''' initial version is in /home/ali/ongoing_research/my_python_codes/modsem'''
    from indexed import IndexedOrderedDict
    # faulthandler.enable()

    vibrational_scaling_squared = vibrational_scaling ** 2  # Square the vibrational scaling used for frequencies
    parts = comp.particles()
    eigenvectors, eigenvalues = dict(), dict()
    hessian = -get_vasp_hessian(outcar_pth)
    n = comp.n_particles()
    hessian_partial = cut_array2d(hessian, [n, n])
    for i, p1 in enumerate(parts):
        for j, p2 in enumerate(parts):
            eigenvalues[p1, p2], eigenvectors[p1, p2] = eig(hessian_partial[i, j])

    def force_constant_bond(atom_A, atom_B, eigenvalues, eigenvectors):
        # Force Constant - Equation 10 of Seminario paper - gives force constant for bond
        c0, c1 = comp.closest_img_bond(atom_A, atom_B)
        diff_AB = c1 - c0
        unit_vector_AB = diff_AB / norm(diff_AB)  # Vector along bond

        return eigenvalues[atom_A, atom_B] @ np.abs(
            unit_vector_AB @ eigenvectors[atom_A, atom_B]) / 2  # divide by 2 cause k/2 = K

    k_b = np.zeros(len(comp.bonds))
    for i, bnd in enumerate(comp.bonds):
        AB = force_constant_bond(bnd[0], bnd[1], eigenvalues, eigenvectors)
        BA = force_constant_bond(bnd[1], bnd[0], eigenvalues, eigenvectors)

        k_b[i] = np.real((
                                     AB + BA) / 2) * vibrational_scaling_squared  # Order of bonds sometimes causes slight differences, find the mean

    val = [0, 2]
    k_a, theta = np.zeros(len(comp.angles)), np.zeros(len(comp.angles))

    for iang, ang in enumerate(comp.angles):
        same_centers = [x for x in comp.angles if x[1] == ang[1]]
        c0, c1, c2 = comp.closest_img_angle(*ang)
        upa, upc, theta[iang] = unit_perps(c0, c1, c2)
        upac = upa, upc

        invk = 0
        for i, bond in enumerate([ang[: 2], ang[2:0:-1]]):
            tmp = np.abs(upac[i] @ eigenvectors[bond[0], bond[1]])
            ki = tmp @ eigenvalues[bond[0], bond[1]]

            coeff = cnt = 0
            for x in [y for y in same_centers if bond[0] in y and set(y) != set(
                    ang)]:  # among bonds with the same center as ang, the ones share bond with ang
                c0, c1, c2 = comp.closest_img_angle(*bond, *[i for i in x if i not in bond])
                up = unit_perps(c0, c1, c2)
                coeff += np.abs(upac[i] @ up[0]) ** 2
                cnt += 1
            fact = 1 + coeff / cnt if cnt else 1
            invk += fact / (dist(*comp.closest_img_bond(*bond)) ** 2 * ki)
        k_a[iang] = 1 / np.real(invk) / 2

    k_d, phi = np.zeros(len(comp.diheds)), np.zeros(len(comp.diheds))
    for i, dihed in enumerate(comp.diheds):
        points = comp.closest_img_dihed(*dihed[0:4])
        normals = plane_normal(*points[0:3]), plane_normal(*points[1:])
        phi[i] = np.degrees(np.arccos(normals[0] @ normals[1]))

        invk = 0
        for j in range(2):  # loop over each dihedral arm (AB or CD)
            blen = dist(*comp.closest_img_bond(*bond))
            sint_sq = np.sin(
                angle_between_vecs(points[j] - points[j + 1], points[j + 2] - points[j + 1], degrees=0)) ** 2
            tmp = np.abs(normals[j] @ eigenvectors[dihed[2 * j], dihed[2 * j + 1]])
            tmp = tmp @ eigenvalues[dihed[2 * j], dihed[2 * j + 1]]

            invk += 1 / blen / sint_sq / tmp
        k_d[i] = 1 / invk.real / 2

    return np.array([*k_b, *k_a, *k_d])

def modsem(comp, outcar_pth, vibrational_scaling=1, ff_xml_pth=None):
    ''' initial version is in /home/ali/ongoing_research/my_python_codes/modsem'''
    from indexed import IndexedOrderedDict
    # faulthandler.enable()

    vibrational_scaling_squared = vibrational_scaling ** 2 # Square the vibrational scaling used for frequencies
    parts = comp.particles_label_sorted()
    eigenvectors, eigenvalues = dict(), dict()
    hessian = -get_vasp_hessian(outcar_pth)
    n = comp.n_particles()
    hessian_partial = cut_array2d(hessian, [n, n])
    for i, p1 in enumerate(parts):
        for j, p2 in enumerate(parts):
            eigenvalues[p1, p2], eigenvectors[p1, p2] = eig(hessian_partial[i, j])

    bonds_list, angles_list, diheds_list = map(list, (comp.bonds_typed.keys(), comp.angles_typed.keys(), comp.propers_typed))
    bonds_length_list = IndexedOrderedDict()

    for x in bonds_list:
        bonds_length_list[frozenset(x)] = dist(*comp.closest_img_bond(*x))
    # atom_names = [x.type['name'] for x in comp.particles_label_sorted()]

    def force_constant_bond(atom_A, atom_B, eigenvalues, eigenvectors, c0, c1):
        # Force Constant - Equation 10 of Seminario paper - gives force constant for bond

        diff_AB = c1 - c0
        unit_vector_AB = diff_AB / norm(diff_AB)  # Vector along bond

        return eigenvalues[atom_A, atom_B] @ np.abs(
            unit_vector_AB @ eigenvectors[atom_A, atom_B]) / 2  # divide by 2 cause k/2 = K

    k_b = np.zeros(len(bonds_list))
    for i, bnd in enumerate(bonds_list):
        c0, c1 = comp.closest_img_bond(bnd[0], bnd[1])
        AB = force_constant_bond(bnd[0], bnd[1], eigenvalues, eigenvectors, c0, c1)
        BA = force_constant_bond(bnd[1], bnd[0], eigenvalues, eigenvectors, c0, c1)

        k_b[i] = np.real((AB + BA) / 2) * vibrational_scaling_squared # Order of bonds sometimes causes slight differences, find the mean

    val = [0, 2]
    k_a, theta = [np.zeros(len(angles_list)) for _ in range(2)]

    for iang, ang in enumerate(angles_list):
        same_centers = [x for x in angles_list if x[1] == ang[1]]
        c0, c1, c2 = comp.closest_img_angle(*ang)
        upa, upc, theta[iang] = unit_perps(c0,c1,c2)
        upac = upa, upc

        invk = 0
        for i, bond in enumerate([ang[0: 2], ang[2:0:-1]]):
            tmp = np.abs(upac[i] @ eigenvectors[bond[0], bond[1]])
            ki = tmp @ eigenvalues[bond[0], bond[1]]

            coeff = cnt = 0
            for x in [y for y in same_centers if bond[0] in y and set(y) != set(ang)]: #among bonds with the same center as ang, the ones share bond with ang
                c0, c1, c2 = comp.closest_img_angle(*bond, *[i for i in x if i not in bond])
                up = unit_perps(c0, c1, c2)
                coeff += np.abs(upac[i] @ up[0]) ** 2
                cnt += 1
            fact = 1 + coeff / cnt if cnt else 1
            invk += fact / (bonds_length_list[frozenset(bond)] ** 2 * ki)
        k_a[iang] = 1 / np.real(invk) / 2

    k_d, phi = [np.zeros(len(diheds_list)) for _ in range(2)]
    for i, dihed in enumerate(diheds_list):
        points = comp.closest_img_dihed(*dihed[0:4])
        normals = plane_normal(*points[0:3]), plane_normal(*points[1:])
        phi[i] = np.degrees(np.arccos(normals[0] @ normals[1]))

        invk = 0
        for j in range(2): # loop over each dihedral arm (AB or CD)
            blen = bonds_length_list[frozenset(dihed[2*j:2*j+2])]
            sint_sq = np.sin(angle_between_vecs(points[j]-points[j+1], points[j+2]-points[j+1], degrees=0))**2
            tmp = np.abs(normals[j] @ eigenvectors[dihed[2*j], dihed[2*j+1]])
            tmp = tmp @ eigenvalues[dihed[2*j], dihed[2*j+1]]

            invk += 1/blen/sint_sq/tmp
        k_d[i] = 1/invk.real

    return np.array([*k_b, *k_a, *k_d])



def parms_from_labels(lbls, comp):
    '''you have connections in label form, you want them in compound tuple'''
    n = len(lbls)
    v = [comp.bonds_typed, comp.angles_typed, comp.propers_typed][n - 2]
    gg = tuple([z for z in comp.particles() if z.type['name'] == qq][0] for qq in lbls)
    if gg not in v:
        gg = gg[-1::-1]
    return float(v[gg]['k']), abs(float(v[gg]['rest']))

def swap_rows_cols(mat, idx):
    idx = list(map(list, idx))
    for i in range(2):
        for x in idx:
            if len(set(x)) == 1:
                continue
            mat[x] = mat[np.flip(x)]
        mat = mat.T

    return mat

def rm_dependent_rows(mat):
    mat = np.array(mat)
    
    out = []
    idx = []
    for cnt, x in enumerate(mat):
        if np.linalg.matrix_rank([*out, x]) == len(out) + 1:
            out.append(x)
            idx.append(cnt)
    return idx, np.array(out)[:, idx]


def zet(s):
    return (s[0] == s[1]) - (s[0] == s[2])

def dihed_derivs2(p1,p2,p3,p4):

    u = p1 - p2
    w = p3 - p2
    v = p4 - p3
    nu,nv,nw = np.array([norm(x) for x in [u,v,w]])
    u,v,w = u/nu, v/nv, w/nw

    cphi_u = dot(u,w)
    sphi_u = sqrt(1-cphi_u**2)
    cphi_v = -dot(v,w)
    sphi_v = sqrt(1-cphi_v**2)
    cuw = cross(u,w)
    cvw = cross(v,w)

    sphi_u4 = sphi_u**4
    sphi_v4 = sphi_v**4

    t1 = einsum('i,j', cuw, w*cphi_u-u)
    e1 = (t1 + t1.T)/sphi_u4/nu**2

    t2 = einsum('i,j', cvw, w*cphi_v-v)
    e2 = (t2+t2.T)/sphi_v4/nv**2

    t3 = einsum('i,j', cuw, w - 2*u*cphi_u + w*cphi_u**2)
    e3 = (t3+t3.T)/sphi_u4/2/nu/nw

    t4 = einsum('i,j', cvw, w + 2*u*cphi_v + w*cphi_v**2)
    e4 = (t4+t4.T)/sphi_v4/2/nv/nw

    t5 = einsum('i,j', cuw, u + u*cphi_u**2 - 3*w*cphi_u + w*cphi_u**3)
    e5 = (t5+t5.T)/sphi_u4/2/nw**2

    t6 = einsum('i,j', cvw, v + v*cphi_v**2 + 3*w*cphi_v - w*cphi_v**3)
    e6 = (t6+t6.T)/sphi_v4/2/nw**2
    
    e7 = np.zeros([3,3])
    e8 = deepcopy(e7)
    for i in range(3):
        for j in [x for x in range(3) if x!=i]:
            k = [x for x in range(3) if x not in [i,j]][0]
            e7[i,j] = (j-i) * (-1/2)**np.abs(j-i) * (w[k]*cphi_u - u[k])/nu/nw/sphi_u
            e8[i,j] = (j-i) * (-1/2)**np.abs(j-i) * (w[k]*cphi_v - v[k])/nv/nw/sphi_v

    val = np.zeros([12,12])
    for cnt1,a in enumerate('mopn'):
        for cnt2,b in enumerate('mopn'):
           val[3*cnt1:3*(cnt1+1),3*cnt2:3*(cnt2+1)] = \
            zet(a+'mo')*zet(b+'mo')*e1 + zet(a+'np')*zet(b+'np')*e2+\
            (zet(a+'mo')*zet(b+'op') + zet(a+'po')*zet(b+'om'))*e3 +\
            (zet(a+'np')*zet(b+'po') + zet(a+'po')*zet(b+'np'))*e4+\
            zet(a+'op')*zet(b+'po')*e5+\
            zet(a+'op')*zet(b+'op')*e6+\
                  (1- (a==b))*(zet(a+'mo')*zet(b+'op') + zet(a+'po')*zet(b+'om'))*e7+\
                  (1- (a==b))*(zet(a+'no')*zet(b+'op') + zet(a+'po')*zet(b+'om'))*e8
    return val

def angle_derivs(rs, n=0):
    p1,p2,p3 = rs
    u = p1 - p2
    v = p3 - p2
    nu,nv = norm(u), norm(v)
    u, v = u/nu, v/nv
    ang = np.arccos(dot(u, v))
    if n == 0:
        return ang
        
    w = cross(u, v)
    if np.allclose(w, 0):
        # w = cross(u, [1, -1, 1])
        u = u + np.array([1, -1, 1])*.0001
        w = cross(u, v)
    if np.allclose(w, 0):
        w = cross(u, [-1, 1, 1])
    nw = norm(w)
    w = w/nw

    val = np.zeros(9)
    if n > 0:
        for cnt, a in enumerate('mon'):
            val[3*cnt:3*(cnt+1)] = zet(a+'mo')*cross(u, w)/nu + zet(a+'no')*cross(w, v)/nv
        if n == 1:
            return ang, val

    cqa = dot(u, v)
    sqa = sqrt(1-cqa**2)

    sval = np.zeros([9, 9])
    if n==2:
        try:
            for cnt1, a in enumerate('mon'):
                for cnt2, b in enumerate('mon'):
                    sval[3*cnt1:3*(cnt1+1),3*cnt2:3*(cnt2+1)] =\
                        zet(a+'mo')*zet(b+'mo')*(einsum('i,j', u, v) + einsum('j,i', u, v) - 3*einsum('i,j', u, u)*cqa + np.eye(3)*cqa)/nu**2/sqa +\
                        zet(a+'no')*zet(b+'no')*(einsum('i,j', v, u) + einsum('j,i', v, u) - 3*einsum('i,j', v, v)*cqa + np.eye(3)*cqa)/nu**2/sqa +\
                        zet(a+'mo')*zet(b+'no')*(einsum('i,j', u, u) + einsum('j,i', v, v) - einsum('i,j', u, v)*cqa - np.eye(3))/nu/nv/sqa +\
                        zet(a+'no')*zet(b+'mo')*(einsum('i,j', v, v) + einsum('j,i', u, u) - einsum('i,j', v, u)*cqa - np.eye(3))/nu/nv/sqa -\
                        cqa/sqa*einsum('i,j', val[3*cnt1:3*(cnt1+1)], val[3*cnt2:3*(cnt2+1)])
        except FloatingPointError:
            sval = np.zeros([9, 9])

        return ang, val, sval
        


def plt_eigenvecs(coords, ev):
    import plotly.graph_objects as go

    # eigenvec = eigenvec.reshape([-1, 3])

    fig = go.Figure(data=go.Cone(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], u=ev[0::3], v=ev[1::3], w=ev[2::3], anchor='tip', hoverinfo='u+v+w'))
    # fig.add_trace(go.Scatter3d(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode='markers'))
    # fig.update_layout(scene_camera_eye=dict(x=-0.76, y=1.8, z=0.92))

    fig.show()


def sub_list(lst, idx):
    return [x for x in lst if lst.index(x) in idx]


class defd(defaultdict):
    def __repr__(self):
        return dict.__repr__(self)


class mt(type):
    def __call__(self, inp1, *inp2):
        inp = [inp1, *inp2] if inp2 else inp1
        return type.__call__(self, inp)

class bt(tuple, metaclass=mt):
    ''' why use list__eq__ directly? you can't call the same method you're defining ! '''

    # def __init__(self, *inp):
    #     pass

    def __eq__(self, other):
        return tuple.__eq__(self, other) or tuple.__eq__(self[-1::-1], other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f'bt{tuple.__repr__(self)}'

    def __hash__(self):
        return 0 #tuple.__hash__(self)


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

    # string = np.array_str(mat, precision=prec)
    # string = re.sub('(\[|\])', '', string)
    # out = re.sub('\n\s+', '\n', string)


    # tmp = '\n'.join([(len(x)*frm).format(*x) for x in mat])
    # if prt:
    #     print(tmp)
    # return tmp
    
def brace_to_curly(lst):
    return str(lst).replace('[','{').replace(']', '}')


def scatter(x, y, plt=1, log=1, **sargs):
    '''sargs: scatter arguments'''

    f = go.Figure(go.Scatter(x=x, y=y, **sargs))
    if log:
        f.update_xaxes(type='log')

    if plt:
        plot(f, show_link=1)

    return f



def gulp_hessian(direc='gulp.out', dm=0, sin=0):
    '''extract hessian from gulp output
    dm: return dynamical matrix rather than the Hessian
    sin: string input instead of file directory'''

    f = direc if sin else open(direc).read()

    return txt_to_mat(re.search('(?s)my derv.*?end', f).group().replace('my derv', '').replace('end',''))

    n = int(re.search('(?<=Number of irreducible atoms\/shells =).*', f).group())

    tmp = re.findall(r'(?<=Internal-internal second derivative matrix \: \(eV\/Angstrom\*\*2\)\n\n).*?\n\n', f, re.DOTALL)[-1].replace('\n', ' ')
    derv2 = txt_to_mat(tmp).reshape([3*n, 3*n])
    if not dm:
        hessian = np.array(tmp.split(), float).reshape(3*n, 3*n)
        # hessian = np.hstack([np.zeros([3*(n-1), 3]), hessian])
        # hessian = np.vstack([np.zeros([3, 3*n]), hessian])
        return hessian
############# from here on needs edit ####################
    tmp = re.search('(?<= Real Dynamical matrix :\n)(?s).*?\n\n', f).group()
    dynmat = np.array(tmp.split(), float).reshape(3*n, 3*n)

    if dm:
        return dynmat

    f = iter(f.split('\n'))
    i = 0
    masses = np.zeros(n)
    for x in f:
        if 'Species output' in x:
            for x in f:
                if bool(re.search('\d', x)):
                    masses[i] = get_cols(x, 4)
                    if i == n - 1:
                        break
                    i += 1

    hessian = dynmat.copy()
    for i in range(n):
        for j in range(n):
            hessian[i, j] = np.sqrt(masses[i]*masses[j]) * dynmat[i, j]
    return hessian


def nearestr(r, latmat):
    '''nearest for vector r
    rv = latmat
    '''
    lst = np.zeros([27, 3])
    for i, n in enumerate(product([-1,0,1], [-1,0,1], [-1,0,1])):
        lst[i, :] = r + n @ latmat
    return lst[np.argmin(norm(lst, axis=1))]
    # rv = rv.T
    # rmin = 10000.0
    # xcdi = xdc - 2.0 * rv[0, 0]
    # ycdi = ydc - 2.0 * rv[1, 0]
    # zcdi = zdc - 2.0 * rv[2, 0]
    # # !
    # # !  Loop over unit cells
    # # !
    # for ii in range(-1, 2):
    #     xcdi = xcdi + rv[0, 0]
    #     ycdi = ycdi + rv[1, 0]
    #     zcdi = zcdi + rv[2, 0]
    #     xcdj = xcdi - 2.0 * rv[0, 1]
    #     ycdj = ycdi - 2.0 * rv[1, 1]
    #     zcdj = zcdi - 2.0 * rv[2, 1]
    #     for jj in range(-1, 2):
    #         xcdj = xcdj + rv[0, 1]
    #         ycdj = ycdj + rv[1, 1]
    #         zcdj = zcdj + rv[2, 1]
    #         xcrd = xcdj - 2.0 * rv[0, 2]
    #         ycrd = ycdj - 2.0 * rv[1, 2]
    #         zcrd = zcdj - 2.0 * rv[2, 2]
    #         for kk in range(-1, 2):
    #             xcrd = xcrd + rv[0, 2]
    #             ycrd = ycrd + rv[1, 2]
    #             zcrd = zcrd + rv[2, 2]
    #             r = xcrd * xcrd + ycrd * ycrd + zcrd * zcrd
    #             if r <= rmin:
    #                 rmin = r
    #                 xdc = xcrd
    #                 ydc = ycrd
    #                 zdc = zcrd
    # return xdc, ydc, zdc
               

def join_silane(portsil, alk, port, tpairs, bpairs):
    from compound import Port
    
    for j, x in enumerate([tpairs[1], bpairs[1]]):
        pos1 = np.mean([i.pos for i in x], axis=0)
        p = Port(pos=pos1, orientation=x[1].pos-x[0].pos)
        port.add(p, expand=0)
        port.remove(x)
        calk = deepcopy(alk)
        portsil.add(calk)
        portsil.force_overlap(calk, calk[-1], p)
        if j:
            calk.reflect(pos1, [0, 0, 1])
            tmp1 = calk.particles()
            for z in tmp1:
                if z.name == 'C':
                    tmp = list(portsil.bond_graph.neighbors(z))
                    while tmp:
                        y = tmp[0]
                        if y.name == 'H':
                            portsil.remove(y)
                        tmp.remove(y)
                portsil.remove(z)


def biosym_to_lammpstrj(fle):
    from ase.cell import Cell
    from compound import Box, Compound
    inp = open(fle).read()

    pbcs = re.findall(r'(?<=PBC).*(?=\()', inp)
    coords, cnt = [], 0
    inp2 = inp.splitlines(1)
    while cnt < len(inp2)-1:
        cnt += 1
        if 'PBC ' in inp2[cnt]:
            tmp = ''
            for x in inp2[cnt+1:]:
                cnt += 1
                if 'end' in x:
                    coords.append(tmp)
                    break
                tmp += x

    n = coords[0].count('\n')
    out = ''
    for p, c in zip(pbcs, coords):  # for each frame

        out += ('ITEM: TIMESTEP\n'
                '0\n'
                'ITEM: NUMBER OF ATOMS\n'
                f'{n}\n'
                'ITEM: BOX BOUNDS xy xz yz pp pp pp\n')
        b = Compound(latmat=Cell.fromcellpar(txt_to_mat(p)).array).box
        out += (f'{b.xlo_bound} {b.xhi_bound} {b.xy}\n'
                f'{b.ylo_bound} {b.yhi_bound} {b.xz}\n'
                f'{b.zlo_bound} {b.zhi_bound} {b.yz}\n')
        # --- begin body ---
        out += 'ITEM: ATOMS element x y z\n'
        out += re.sub('(?<=CORE).*', '', c)
    breakpoint()



def biosym_to_lammpstrj2(comp, fle):
    from ase.cell import Cell
    from compound import Box, Compound
    inp = open(fle).read()

    pbcs = re.findall(r'(?<=PBC ).*?(?=\(|\n)', inp)
    coords, cnt = [], 0
    inp2 = inp.splitlines(1)
    while cnt < len(inp2)-1:
        cnt += 1
        if 'PBC ' in inp2[cnt]:
            tmp = ''
            for x in inp2[cnt+1:]:
                cnt += 1
                if 'end' in x:
                    coords.append(tmp)
                    break
                tmp += x

    out = ''
    for p, c in zip(pbcs, coords):  # for each frame
        cc = ''
        for x in c.splitlines():
            cc += ' '.join(get_cols(x, [2, 3, 4])) + '\n'
        comp.add_frame(txt_to_mat(cc), Cell.fromcellpar(txt_to_mat(p)).array)


def get_gul_elast(fout):
    '''out: string of gulp output'''
    etmp = []
    for j, y in enumerate(fout):
        if 'Elastic Constant Matrix' in y:
            c = []
            for jj in range(6):
                c.append(list(sliced(fout[j + 5:j + 11][jj], 10)))
            c = array(c, dtype=float)
            c = np.vstack([np.zeros([1, 7]), c])
            etmp.extend([c[1, 1], c[2, 2], c[3, 3], c[4, 4], c[1, 2], c[1, 3]])
    return array(etmp)


def gen_pol_bonds(comp):
    c, h, o, si = comp['c\d+'], comp['h\d+'], comp['o\d+'], comp['si\d+']
    comp.generate_bonds(c, h, 1, 1.4)
    comp.generate_bonds(c, c, 1.4, 1.6)
    comp.generate_bonds(h, o, .5, 1.2)
    comp.generate_bonds(si, o, 1.4, 1.9)
    comp.generate_bonds(si, c, 1.4, 2)


def hybrid_sil_port_dimer(nc=6, direc=''):
    '''create hybrid polymer+portlandite system with bonding
    nc: number of carbon in chain
    direc: directory to write the poscar in'''
    from compound import compload, Port, Compound

    if nc == 4:
        bz = 11.5
    elif nc == 6:
        bz = 14  # box z
    elif nc == 8:
        bz = 16.5

    port = compload('/home/ali/ongoing_research/polymer4/14_mbuild/port.cif', ad_names=1)  # type:Compound
    port.add_bond(list(zip(port['o(1|2)'], port['h(1|2)'])))

    port.supercell([[2, 0, 0], [1, 2, 0], [0, 0, 1]])

    port.xyz += np.array([0, 0, 2.2])
    port.wrap_atoms()

    alk = dimer(nc)

    port.latmat[2, 2] += 12

    port.remove(port['h(1|5|2|6)'])

    pos1 = np.mean([alk['o3'].pos, alk['o4'].pos], 0)  # the oxygens that connect to portlandite
    pos2 = np.mean([port['o1'].pos, port['o5'].pos], 0)  # the oxygens that connect to portlandite
    alk.add(Port(pos=pos1, orientation=alk['o3'].pos - alk['o4'].pos), expand=0)
    port.add(Port(pos=pos2, orientation=port['o1'].pos - port['o5'].pos), expand=0)

    portsil = Compound()
    portsil.latmat = port.latmat
    portsil.latmat[2, 2] = bz  # 13 for hexane
    portsil.add(port)
    portsil.add(alk)
    portsil.force_overlap(alk, alk['p.*'], port['p.*'])

    sil = compload('/home/ali/ongoing_research/polymer4/22_lammps_py_fitting/sil.mol2')  # type: Compound
    pos1 = np.mean([sil['o3'].pos, sil['o4'].pos], 0)  # the oxygens that connect to portlandite
    pos2 = np.mean([port['o2'].pos, port['o6'].pos], 0)  # the oxygens that connect to portlandite
    sil.add(Port(pos=pos1, orientation=sil['o3'].pos - sil['o4'].pos))
    port.add(Port(pos=pos2, orientation=port['o2'].pos - port['o6'].pos))
    portsil.add(sil)
    portsil.force_overlap(sil, sil['p.*'], port['p.*'], rotate_ang=180)

    vec = alk[f'c{nc - 2}'].pos - alk[f'c{nc}'].pos
    alk['alkane1'].rotate_vecs(vec, [0, 0, 1], rotpnt=alk[f'c{nc}'])  # rotate polymers
    alk['alkane2'].rotate_vecs(vec, [0, 0, 1], rotpnt=alk[f'c{2 * nc}'])  # rotate polymers

    vec = alk[f'c{nc - 2}'].pos - alk[f'c{nc}'].pos
    alk['alkane1'].rotate_around(vec, -30, pnt=alk[f'c{nc}'].pos)
    alk['alkane2'].rotate_around(vec, -30, pnt=alk[f'c{2 * nc}'].pos)

    vec = portsil['si1'].pos - portsil['si2'].pos
    ang = 20 if nc == 4 else 10
    alk['alkane1'].rotate_around(vec, ang, pnt=alk[f'c{nc}'].pos)
    alk['alkane2'].rotate_around(vec, ang, pnt=alk[f'c{2 * nc}'].pos)

    portsil.add_bond([[alk['c1'], sil['si3']], [alk[f'c{nc + 1}'], sil['si4']]])

    # portsil.vmd(commands='\nmol showperiodic 0 0 z\nmol numperiodic 0 0 1')

    port.remove(port['o(1|5|2|6)'])

    if direc:
        portsil.write_poscar(direc, fixed_atoms=portsil.sel('ca.*|si.*|o.*'))
    return portsil



def hybrid_sil_port_trimer(nc=6, direc=''):
    '''create hybrid polymer+portlandite system with bonding
    nc: number of carbon in chain
    direc: directory to write the poscar in'''
    from compound import compload, Port, Compound

    if nc == 4:
        bz = 10.5
    elif nc == 6:
        bz = 14  # box z
    elif nc == 8:
        bz = 16.5

    port = compload('/home/ali/ongoing_research/polymer4/14_mbuild/port.cif', ad_names=1)  # type:Compound
    port.add_bond(list(zip(port['o(1|2)'], port['h(1|2)'])))

    port.supercell([[3, 0, 0], [1, 2, 0], [0, 0, 1]])

    port.xyz += np.array([0, 0, 2.2])
    port.wrap_atoms()

    alk = trimer(nc)

    port.latmat[2, 2] += bz

    port.remove(port['h(3|9|11|2|6|10)'])

    pos1 = np.mean([alk['o7'].pos, alk['o4'].pos], 0)  # the oxygens that connect to portlandite
    pos2 = np.mean([port['o3'].pos, port['o9'].pos], 0)  # the oxygens that connect to portlandite
    alk.add(fport(pos=pos1, orientation=alk['o7'].pos - alk['o4'].pos), expand=0)
    port.add(fport(pos=pos2, orientation=port['o3'].pos - port['o9'].pos), expand=0)

    portsil = Compound()
    portsil.latmat = port.latmat
    portsil.latmat[2, 2] = bz  # 13 for hexane
    portsil.add(port)
    portsil.add(alk)
    portsil.force_overlap(alk, alk['p.*'], port['p.*'])

    for x, y in zip(['o15', 'o16', 'o19'], ['o11', 'o9', 'o3']):
        portsil[x].pos = portsil[y].pos

    lst = alk['o\d+|si\d+|^h7$|^h32|c4|c8|c12']
    trisil = Compound(atoms = deepcopy(lst))
    portsil.add(trisil)

    for x in combinations(lst, 2):
        if portsil.bond_graph.has_edge(*x):
            x, _ = portsil.neighbs(x, rcut=.1)
            for i, t in enumerate(x):
                if len(t) > 1:
                    x[i] = [z for z in t if z.name.lower() not in ['o11', 'o9', 'o3']]
            portsil.add_bond([y[0] for y in x])

    trisil.reflect(portsil.sel(['o11', 'o9', 'o3']))
    trisil.move(portsil['o26'], portsil['o2'])

    lst = portsil.sel('c13|c14|c15', lst=0)
    lst.latmat = portsil.latmat
    tmp = (lst.frac_coords - np.floor(lst.frac_coords)) @ lst.latmat

    for i, (x, y) in enumerate(zip([1, 5, 9], [4, 8, 12]), 1):
        portsil[f'alkane{i}'].rotate_vecs(portsil[f'c{x}'].pos - portsil[f'c{y}'].pos, tmp[i-1] - portsil[f'c{y}'].pos, rotpnt=portsil[f'c{y}'])

    portsil['alkane2'].rotate_around(portsil['c5'].pos - portsil['c8'].pos, 90, pnt=portsil['c8'])

    portsil.remove(lst.particles())

    a, b = portsil.neighbs(portsil.particles(), rcut=.2)

    t = list(flatten([[*x] for x in portsil.bonds]))
    for x in flatten(a):
        if x not in t:
            portsil.remove(x)

    if direc:
        portsil.write_poscar(direc, fixed_atoms=portsil.sel('ca.*|si.*|o.*'))
    return portsil


def hessian_to_dynmat(comp, hess):
    '''multiply by 1/sqrt(m_im_j)'''
    n = comp.n_particles() * 3
    dynmat = np.zeros([n, n])
    parts = comp.particles()
    masses = np.repeat(np.array([x.mass for x in parts], dtype=float), 3)
    for i in range(n):
        for j in range(n):
            dynmat[i, j] = 1/np.sqrt(masses[i] * masses[j]) * hess[i, j]
    return dynmat

def bond_derivs(rs, deriv=1):
    ''' seeing bond length as generalized coordinate, this gives dq/dx_i or d^2q/dx_i*dx_j'''
    rs = np.array(rs)
    del_r = rs[0] - rs[1]
    ndr = norm(del_r)
    ndrv = del_r/ndr #normalized dr vector
    if deriv == 1:
        return ndr, np.hstack([ndrv, -ndrv])

    hessian = np.zeros([6, 6])
    if deriv == 2:
        tmp = (np.outer(ndrv, ndrv) - np.eye(3))/ndr
        for i in range(2):
            for j in range(2):
                hessian[3*i:3*(i+1), 3*j:3*(j+1)] = (-1)**(i == j) * tmp
        return hessian

def dihed_derivs(p1,p2,p3,p4,d='dum', only_dphi = 0):
    '''from Karplus' paper'''
    p1,p2,p3,p4 = map(np.array,[p1,p2,p3,p4])

    F = p1 - p2
    G = p2 - p3
    H = p4 - p3
    A = cross(F, G)
    B = cross(H, G)

    phi = np.arccos(dot(A,B)/norm(A)/norm(B))
    if d==0:
        return phi

    nG = norm(G); nB = norm(B)
    nAsq = norm(A)**2
    nBsq = norm(B)**2

    dfg = dot(F,G)
    dhg = dot(H,G)

    dphi_dr1 = -nG/nAsq*A
    dphi_dr2 = nG/nAsq*A + dfg/nAsq/nG*A - dhg/nBsq/nG*B
    dphi_dr3 = dhg/nBsq/nG*B - dfg/nAsq/nG*A - nG/nBsq*B
    dphi_dr4 = nG/nBsq*B

    dphi_dr = np.array([*dphi_dr1, *dphi_dr2, *dphi_dr3, *dphi_dr4])
    if only_dphi:
        return phi, dphi_dr

    s1 = range(0,3); s2 = range(3,6); s3 = range(6,9);

    dT_dr=np.array([[1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1]])
    d2phi_dT2 = np.zeros([9, 9])
    cga = cross(G,A)
    cfa = cross(F,A)
    chb = cross(H,B)
    cgb = cross(G,B)
    t1 = einsum('i,j', A, cga) + einsum('i,j', cga, A)
    t2 = einsum('i,j', A, cfa) + einsum('i,j', cfa, A)
    t3 = einsum('i,j', B, cgb) + einsum('i,j', cgb, B)
    t4 = einsum('i,j', B, chb) + einsum('i,j', chb, B)

    d2phi_dT2[np.ix_(s1, s1)] = nG/nAsq**2*t1 #eq 32

    d2phi_dT2[np.ix_(s2, s2)] = 1/2/nG**3/nAsq*t1 + dfg/nG/nAsq**2*t2 -  1/2/nG**3/nBsq*t3 - dhg/nG/nBsq**2*t4 #eq 44

    d2phi_dT2[np.ix_(s3, s3)] = -nG/nBsq**2*t3 #eq 33

    d2phi_dT2[np.ix_(s1, s2)] = -1/nG/nAsq**2*(nG**2*einsum('i,j', cfa, A) + dfg*einsum('i,j', A, cga)) #eq 38
    d2phi_dT2[np.ix_(s2, s1)] = d2phi_dT2[np.ix_(s1, s2)]

    d2phi_dT2[np.ix_(s2, s3)] = 1/nG/nBsq**2*(nG**2*einsum('i,j', chb, B) + dhg*einsum('i,j', B, cgb)) #eq 39
    d2phi_dT2[np.ix_(s3, s2)] = d2phi_dT2[np.ix_(s2, s3)]

    # kk = np.zeros([12,12])
    # for i in range(12):
    #     for j in range(12):
    #         tmp = 0
    #         for k in range(9):
    #             for l in range(9):
    #                 tmp += dT_dr[k, i]*dT_dr[l, j]*d2phi_dT2[k, l]
    #         kk[i, j] = tmp
    tmp = einsum('ij,kl', dT_dr, dT_dr)
    return phi, dphi_dr, einsum('ij,ikjl',d2phi_dT2,tmp)

def make_unique(original_list):
    unique_list, idx = [], []
    for i, obj in enumerate(original_list):
        if obj not in unique_list:
            unique_list.append(obj)
            idx.append(i)
    return unique_list, idx

def gulp_out_coords(direc='gulp.out'):
    coords = shell('''awk '/Fractional c/{for (i=0;i<6;i++) getline; flg=1}/---/{flg=0}flg{print $4, $5, $6}' '''+direc)
    lat = shell('''awk '/Cartesian lattice/{getline;getline; for (i=0;i<3;i++) {print;getline}}' '''+direc)
    n = shell('''awk '/Number of irred/ {print $6}' '''+direc).decode('utf-8')
    n = int(n)
    return (np.genfromtxt(io.BytesIO(coords)) @ (lat:=np.genfromtxt(io.BytesIO(lat)))) [-n:], lat 

def plane_normal(p1, p2, p3):
    ''' normal to plane from three points '''
    p1, p2, p3 = map(np.array, [p1, p2, p3])
    v1 = p1 - p2
    v2 = p3 - p2
    vec = np.cross(v1, v2)
    return vec/norm(vec)


def ab_to_es(A, B, reverse=0):
    '''
  #          A       B                            /  / s \^a    / s \^b \
  #  U(r) = ---  -  ---           =       4 * e * |  |---|   -  |---|   |
  #         r^a     r^b                           \  \ r /      \ r /   /

  #    A =  4*e*s^a
  #    B = -4*e*s^b

  For A,B>0:
  #    s = (A/B)^(1/(a-b))
  #    e = -B / (4*s^b)
  #  Setting B=0.0001, and A=1.0   (for sigma=2^(-1/6))
  #  ---> e=2.5e-09, s=4.641588833612778
  #  This is good enough for us.  (The well depth is 2.5e-9 which is almost 0)'''
    A, B = map(np.array, (A, B))
    if reverse:
        e, s = A, B
        A = 4*e*s**12
        B = 4*e*s**6
    else:
        id1 = np.isclose(A, 0, rtol=0, atol=1e-8)
        id2 = np.isclose(B, 0, rtol=0, atol=1e-8)
        id3 = np.logical_and(id1, id2)
        A[id3], B[id3] = 1, 1

        s = (A/B)**(1/6)
        e = B/(4*s**6)

        e[id3], s[id3] = 0, 0

    return (A, B) if reverse else (e, s)

def get_2d_mat(inpstr):
    aa = inpstr.strip().split('\n')
    return np.array([x.split() for x in aa], dtype=float)

def b_to_mat(inp, dt=float):
    return np.genfromtxt(io.BytesIO(inp), dtype=dt)

def txt_to_mat(inp, dt=float):
    return np.genfromtxt(io.StringIO(inp), dtype=dt)

def flatten(l):
    for el in l:
        if isinstance(el, list) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def pairs_in_rcut(compound, pairs, rcut):
    tmp = set()
    tpairs = []
    for p in pairs:
        _, dist = compound.neighbs([p[0]], [p[1]], rcut=3.6)
        if dist and not [x for x in p if x in tmp]:
            tpairs.append(p)
            [tmp.add(x) for x in p]
    return tpairs

def transform_mat(A, B):
    ''' two sets of 3D points, gives the matrix that rigid transforms one into the other.
     for details see http://nghiaho.com/?page_id=671'''
    A, B = map(np.array, [A, B])

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    centroid_A.shape = (1, 3)
    centroid_B.shape = (1, 3)

    H = np.zeros((3, 3), dtype=float)
    for i in range(A.shape[0]):
        H = H + (A[i, :] - centroid_A).reshape(3,1) @ (B[i, :] - centroid_B)
    U, _, V = svd(H)
    R = np.transpose(U @ V)

    t = (centroid_B.reshape(3,1) - R @ centroid_A.reshape(3,1))
    return np.vstack([np.hstack([R, t]), [0, 0, 0, 1]])

def apply_transform(T, points):
    points = np.array(points)
    one_row = np.ones([1,points.shape[0]])
    return (T @ np.vstack([points.T, one_row]))[0:3,:].T


def alkane(n, num_caps=0):
    from compound import Compound, Port
    from scipy.spatial.transform import Rotation as R

    ch2 = Compound(name='ch', names=['C', 'H', 'H'], pos=[[0,0.5935,0],[0,1.2513,0.8857],[ 0,1.2513,-0.8857]])

    dr1, dr2 = [-0.64315, -0.42845, 0], [ 0.64315, -0.42845, 0]
    ch2.add(fport(anchor=ch2['C1'], orientation=dr1, pos=ch2['C1'].pos+dr1))
    ch2.add(fport(anchor=ch2['C1'], orientation=-np.array(dr2), pos=ch2['C1'].pos+dr2))
    ch2.add_bond([[ch2['C1'], ch2['h1']],[ch2['c1'], ch2['h2']]])
    alk = Compound(name='alkane')
    alk.add(deepcopy(ch2), expand=0)
    for i in range(1, n):
        c_ch2 = deepcopy(ch2)
        alk.add(c_ch2, expand=0)
        alk.force_overlap(c_ch2, c_ch2[f'p!{2*i+1}'], alk.sel(f'p!{2*i}', op=0)[0], rotate_ang=0 if np.mod(i-1, 2) else 180)


    ch3 = Compound(names=['C', 'H', 'H', 'H'],
                         pos=[[-0.0000,   0.0000,   0.0000],
                              [ 0.6252,   0.6252,   0.6252],
                              [-0.6252,  -0.6252,   0.6252],
                              [-0.6252,   0.6252,  -0.6252]])
    
    [ch3.add_bond(x) for x in product([ch3['c1']], ch3['h.*'])]

    dr = np.array([ 0.6252,  -0.6252,  -0.6252])/2
    ch3.add(fport(anchor=ch3['C1'],pos=dr, orientation=-dr), expand=0)

    if n == 0:
        alk = Compound(name='alkane')
        cp1, cp2 = deepcopy(ch3), deepcopy(ch3)
        alk.add([cp1, cp2])
        alk.force_overlap(cp1, cp1['p!'], cp2['p!'], flip=0)
        return alk
    for i in range(num_caps):
        cp = deepcopy(ch3)
        try:
            p = alk['p!.*'][0]
        except IndexError:
            p = alk['p!.*']
        alk.force_overlap(cp, cp['p!.*'], p, flip=i)
        alk.add(cp)

    # alk.unique_names()
    return alk

def fport(anchor=None, orientation=[1, 0, 0], pos=None, name='p!'):
    'fport: function port as opposed to class Port'
    from compound import Compound
    if pos is None:
        pos = anchor.pos

    port = Compound(name, pos=pos)
    orientation, loc_vec = map(np.asarray, [orientation, pos])

    port.anchor, port.orientation = anchor, orientation / norm(orientation)
    return port


def get_gulp_derv(fname='gulp.out'):
    f = open(fname)

    flg = 1
    out = []
    for x in f:
        if 'Final internal derivatives :' in x:
            tmp = ''
            for x in islice(f, 5, None):
                if '---' in x:
                    break
                tmp += ' '.join(get_cols(x, [4, 5, 6])) + '\n'
            out.append(txt_to_mat(tmp))
            # if flg:
            #     flg = 0
            #     out = txt_to_mat(tmp)[:, :, None]
            # else:
            #     out = np.dstack([out, txt_to_mat(tmp)])
    return out


def trimer2(nchain):
    from compound import Compound, compload

    if nchain == 4:
        nz = 6.5
    elif nchain == 6:
        nz = 9.2
    elif nchain == 8:
        nz = 11.1
    trimer = Compound(name='trimer', latmat=10*np.eye(3))
    th = Compound (name='thed', pos=array([[-3.46944695e-16, -6.34041430e-16,  1.59030300e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [-7.50805127e-01,  1.29970043e+00,  2.12162845e+00],
       [-7.50091343e-01, -1.29920327e+00,  2.11790897e+00],
       [ 1.50089646e+00, -4.97161173e-04,  2.12167459e+00]]), names=['Si', 'O', 'O', 'O', 'O', 'O'], latmat=10*np.eye(3))
    trimer.add(th)

    for i in range(1, 3):
        th2 = deepcopy(th)
        th2.add(fport(th2['si1'], orientation=[*(th2['si1'].pos - th2['o3'].pos)[:2], 0], pos=th2['o3'].pos))
        th2.remove(th2['o3'])
        trimer.add(fport(trimer[f'o{4*i-2}'], orientation=[*(-trimer[f'si{i}'].pos + trimer[f'o{4*i-2}'].pos)[:2], 0]))

        trimer.add(th2)
        trimer.force_overlap(th2, th2['p.*'], trimer['p!1'], add_bond=0)

    trimer.generate_bonds(trimer['si\d+'], trimer['o\d+'], 1, 1.8)

    trimer.xyz = txt_to_mat('''5.680201281    2.493852597    4.459681180
6.304877140    2.578771031    3.000000000
4.804307170    3.791406420    4.737643546
4.761340732    1.200815942    4.568545657
6.850280148    2.404416871    5.532535615
4.298824359    5.282673289    4.516941096
4.542686984    5.706512975    3.004000000
2.746217439    5.370527126    4.848291557
5.102086201    6.262246684    5.477829474
1.191632386    5.551420811    4.567859637
0.952825686    5.737931326    3.007000000
0.408443823    4.263452040    5.073677567
0.659042754    6.833773294    5.342469565''') # got these coordinates from gulp in polymer5/69_trimer_tet

    trim = Compound(name='trimers')
    trim.add([trimer, deepcopy(trimer)])

    for i in range(1, 4):
        trimer.add(fport(trimer[f'si{i}'], orientation=[0, 0, 1], pos=trimer[f'o{3*i+1}'].pos))
        trimer.remove(trimer[f'o{3*i+1}'])
        alk = alkane(nchain)
        alk.remove(alk['p.*'])
        alk.add(fport(alk['c1'], orientation=alk['c3'].pos - alk['c1'].pos))
        trimer.add(alk)
        trimer.force_overlap(alk, alk['p.*'], trimer['p!1'])

    port = compload('/home/ali/ongoing_research/polymer4/14_mbuild/port.cif', ad_names=1)  # type:Compound
    port.add_bond(list(zip(port['o(1|2)'], port['h(1|2)'])))

    port.supercell([[3, 0, 0], [1, 2, 0], [0, 0, 1]])

    port.xyz += np.array([0, 0, 2.2])
    port.wrap_atoms()
    port.latmat[2, 2] += nz
    # port.remove(port['h(3|9|11|4|10|12)'])
    port.remove(port['h(3|9|11|2|6|10)'])

    port.add(fport(port['o11'], orientation=port['o11'].pos-port['o9'].pos))
    trimer.add(fport(trimer['o1'], orientation=trimer['o1'].pos-trimer['o5'].pos))

    t = Compound(latmat=port.latmat)
    t.add([port, trim])

    t.force_overlap(trim, trimer['p.*'], port['p.*'])

    t['trimers1']['trimer2'].reflect(t['ca1|ca2|ca3'])
    t['trimers1']['trimer2'].move(t['o20'], t['o10'])
    # t['trimers1']['trimer2'].move(t['o24'], t['o10'])

    for x in t['o\d+']:
        a, b = t.neighbs(x, rcut=.1)
        if a[0]:
            t.remove(x)

    for i, (x, y, z) in enumerate(zip([1, 1+nchain, 1+2*nchain], nchain*array([1, 2, 3]), [4, 5, 6]), 1):
        t[f'alkane{i}'].rotate_vecs(t[f'c{y}'].pos - t[f'c{x}'].pos, t.latmat[2, :]+t[f'si{z}'].pos - t[f'c{x}'].pos, rotpnt=t[f'c{x}'])

    t.remove(t['o(29|26|23)'])

    parts = [Compound(name='H', pos=g.pos+[0, 0, -1]) for g in t['o(15|19)']]
    parts.extend([Compound(name='H', pos=g.pos+[0, 0, 1]) for g in t['o(22|28)']])
    t.add(parts)
    t.add_bond(zip(t['o(15|19|22|28)'], parts))

    for x in t.particles():
        t.bond_graph.add_node(x)
        t.bond_graph.nodes._nodes[x] = {x: x}
    return t

def dimer(nchain):
    from compound import Compound, compload, Port
    from funcs import alkane
    alksil = Compound(name='alksil', latmat=np.eye(3)*50)

    sil = compload('/home/ali/ongoing_research/polymer4/22_lammps_py_fitting/sil.mol2')  # type: Compound
    sil.name = 'sil1'
    c2, c1 = [6.3001, 4.1639, 6.0570], [8.5253, 8.0180, 6.0570]
    tmp1, tmp2 = c1 - sil['Si1'].pos, c2 - sil['Si2'].pos
    sil.add(fport(sil['Si1'], pos=sil['Si1'].pos+tmp1/2, orientation=tmp1, name='p!1'), expand=0)
    sil.add(fport(sil['Si2'], pos=sil['Si2'].pos+tmp2/2, orientation=tmp2, name='p!2'), expand=0)
    alksil.add(sil, expand=0)

    alkane = alkane(nchain)
    alkane['p!1'].orientation = alkane['p!1'].pos - alkane['C1'].pos
    alksil.add(deepcopy(alkane), expand=0)
    alksil.add(deepcopy(alkane), expand=0)

    alksil.force_overlap(alksil['alkane1'], alksil['p!4'], alksil['p!1'], rotate_ang=90)
    alksil.force_overlap(alksil['alkane2'], alksil['p!6'], alksil['p!2'], rotate_ang=90)
    alksil.remove([x for x in alksil.particles(1) if '!' in x.name])
    # alksil.unique_names()
    return alksil

def trimer(nchain):
    from compound import Compound, compload, Port
    alksil = dimer2(nchain)
    sel = deepcopy(alksil.sel('alkane1|o4|o1|si1', 0, 0))
    sel.add(fport(sel['si\d'], pos=alksil['o5'].pos))
    alksil.add(sel, 1)
    alksil.add(fport(alksil['o1']))
    alksil.force_overlap(sel, sel['p!1'], alksil['p!2'])
    sel.rotate_around([0, 0, 1], 60, pnt=alksil['o1'])
    alksil.remove(['h1'])
    alksil.add(Compound('H', pos=alksil['o6'].pos+[0, 1, 0]))

    alksil.add_bond([['si3', 'o7'], ['si3', 'o6'], ['c9', 'h19'], ['c9', 'h20'],
                     ['c10', 'h21'], ['c10', 'h22'], ['c11', 'h23'], ['c11', 'h24'],
                     ['o6', 'h27'], ['c12', 'h26'], ['c12', 'h25'], ['c10', 'c11'], ['c11', 'c12'], ['si3', 'c12'], ['c9', 'c10']])

    return alksil

def regng(a, b):
    from regex_engine import generator as rgen
    return rgen().numerical_range(a, b).replace('^', '').replace('$', '')

def mod_com(commands, intervals, tt, opts, pstr: 'partial string', par_lst, par_lst2):
    a, b = [eval(str(tt['f'+x])) for x in opts[:2]]
    temp = (opts[2] + pstr + ('* ' if opts[2] == 'pair' else '') + '{} {}\n')\
        .format('{}' if a else tt[opts[0]], '{}' if b else tt[opts[1]])
    intervals.extend([x for x in (a, b) if x])
    if '{' not in temp:
        return commands
    commands += temp
    def get_str(x, y):
        if 'bond' in opts:
            return f"{opts[2]} ~ {tt['type1']}-{tt['type2']} ~ {y} ~ {str(x)} "
        elif 'pair' in opts:
            return f"{opts[2]} ~ {tt['type']} ~ {y} ~ {str(x)} "
        elif 'angle' in opts:
            return f"{opts[2]} ~ {tt['type1']}-{tt['type2']}-{tt['type3']} ~ {y} ~ {str(x)} "
    if a:
        par_lst.append(get_str(a, opts[0]))
        par_lst2.append((dict(tt), opts[0]))
    if b:
        par_lst.append(get_str(b, opts[1]))
        par_lst2.append((dict(tt), opts[1]))
    return commands


def get_lmps(configs, inp_fle='runfile'):
    '''
    :param configs: nframes*natoms*3
    :return: list of lammps objects for each config, forces for these configurations
    '''
    lmps = []
    frc_lmps = np.empty(configs.shape, dtype=float)
    for i, pos in enumerate(configs):
        # lmp = lammps(cmdargs=['-echo', 'none', '-screen', 'none'])
        lmp = lammps()
        lmp.file(inp_fle)
        x = lmp.extract_atom("x", 3)
        for j, p in enumerate(pos):
        	for k, c in enumerate(p):
        		x[j][k] = c
        lmps.append(lmp)
        lmp.command('run 0')
        frc_lmps[i] = np.ctypeslib.as_array(lmp.extract_atom("f", 3).contents, shape=configs[0].shape)
    return lmps[0] if len(lmps)==1 else lmps, frc_lmps

def ev_to_kcalpmol(input, reverse=0, gulp=1):
    ''' gulp: based on what gulp uses'''
    from parmed import unit as u
    input = np.array(input)
    if gulp:
        evtokcal = 23.060373516758343    # 23.06037352 :
        return (input/evtokcal if reverse else evtokcal*input)

    return u.AVOGADRO_CONSTANT_NA._value**-1 * (
		input * u.kilocalorie).in_units_of(u.elementary_charge * u.volts)._value if reverse else u.AVOGADRO_CONSTANT_NA._value * (
		input * u.elementary_charge * u.volts).in_units_of(u.kilocalorie)._value


def pol2cart(r, phi, reverse=0):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y
    if reverse:  #needs fixing
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return (rho, phi)

def get_file_num(outdir):
    ''' if the file name is sth like out23, it returns 23'''
    files = next(os.walk(f'./{outdir}'))[-1]
    try:
        return np.sort(np.array([int(re.match('[^_a-zA-Z]*', x).group())
                            for x in files if re.match('[^_a-zA-Z]*', x).group() != '']))[-1] + 1
    except:
	    return 1

def vasp_output(direc, save_npy=0, get_freqs=1, get_eigvecs=1):
    '''direc: outcar directory
    latmat, coords, frcs, derv2, freqs, eigr'''

    pth = os.path.join(direc,"OUTCAR")

    with open(pth) as f:
        latmat, coords, frcs, types = [], [], [], []
        fcnt = 0
        for ln in f:
            if 'direct lattice vectors' in ln:
                latmat.append([])
                for i in range(3):
                    latmat[-1].append(get_cols(next(f), range(1, 4)))

            if 'position of ions in cartesian coordinates' in ln:
                tmp = ''
                for ln in f:
                    if not ln.strip():
                        break
                    tmp += ln
                coords.append(get_2d_mat(tmp))
                n = len(coords[-1])
                freqs = [None] * 3*n
                eigr = np.zeros([3*n, 3*n])

            if 'TOTAL-FORCE' in ln:
                next(f)
                tmp = ''
                for i in range(n):
                    tmp += next(f)
                tmp = get_2d_mat(tmp)
                coords.append(tmp[:, :3])
                frcs.append(tmp[:, 3:])

            if 'cm-1' in ln:
                freqs[fcnt] = re.search(r'[0-9\.\-]*\s+(?=cm-1)', ln).group()
                if 'f/i=' in ln:
                    freqs[fcnt] = '-' + freqs[fcnt]
                next(f)
                tmp = ''
                for i in range(n):
                    tmp += next(f)
                eigr[:, fcnt] = get_2d_mat(tmp)[:, 3:].flatten()
                fcnt += 1

            if 'SECOND DERIVATIVES' in ln:
                f = islice(f, 2, None)
                tmp = ''
                for x in f:
                    if not x.strip():
                        break
                    tmp += x[6:]
                derv2 = get_2d_mat(tmp)

            if 'ions per type' in ln:
                nt = np.array(re.search(r'\d.*', ln).group().split(), dtype=int)
            if 'VRHFIN' in ln:
                types.append(re.search(r'(?<==).*?(?=:)', ln).group())

    tm = []
    for x in types:
        tm.append(amass(x))
    elems = []
    masses = []
    for i, x in enumerate(types):
        for j in range(nt[i]):
            elems.append(x)
            masses.append(tm[i])
    return np.array(latmat, dtype=float), coords, frcs, derv2, np.array(freqs, dtype=float), eigr, elems, masses

    # n = shell(f' grep -oP -m 1 \'(?<=NIONS =)\\s*[0-9]*\' {os.path.join(direc,"OUTCAR")} ')
    # n = int(n.decode())
    # dft_pos_frc = shell(
    #     f'''awk '/TOTAL-/{{getline;getline;flg=1}};/--/{{flg=0}}flg{{print $1, $2, $3, $4, $5, $6}}' {direc}/OUTCAR ''')
    # dft_pos_frc = np.vstack([np.fromstring(x, np.float, sep=' ')
    #                     for x in dft_pos_frc.decode().split('\n')[:-1]])
    # dft_pos_frc = cut_array2d(dft_pos_frc, [len(dft_pos_frc) // n, 1])
    #
    # return np.array([x[:, :3] for x in dft_pos_frc]), np.array([x[:, 3:] for x in dft_pos_frc])

def cut_array2d(array, shape):
    xstep = array.shape[0]//shape[0]
    ystep = array.shape[1]//shape[1]

    if 1 in shape:
        blocks = np.repeat(None, np.max(shape))
    else:
        blocks = np.tile(None, shape)

    cnt = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            val = array[i*xstep:(i+1)*xstep, j*ystep:(j+1)*ystep]
            if 1 in shape:
                blocks[cnt] = val
                cnt += 1
            else:
                blocks[i, j] = val
    return blocks

def get_vasp_hessian(direc):
    n = shell(f' grep -oP -m 1 \'(?<=NIONS =)\\s*[0-9]*\' {os.path.join(direc,"OUTCAR")} ')
    n = int(n)
    hessian = check_output(f''' awk '/^  1X/{{flg=1}};/^ Eigen/{{flg=0}}flg{{$1="";print $0}}' {direc}/OUTCAR ''', shell=1)
    return -(u.AVOGADRO_CONSTANT_NA._value *(np.genfromtxt(io.BytesIO(hessian)) *
                                            u.elementary_charge * u.volts).in_units_of(u.kilocalorie))._value

def  unit_perps(c0, c1, c2):
    #This gives the vector in the plane A,B,C and perpendicular to A to B

    diff_AB = c1 - c0
    u_AB = diff_AB / norm(diff_AB)

    diff_CB = c1 - c2
    u_CB = diff_CB / norm(diff_CB)

    cross_product = np.cross(u_CB, u_AB)
    u_N = cross_product / norm(cross_product)

    u_PA = np.cross(u_N, u_AB)
    u_PC = -np.cross(u_N, u_CB)

    return u_PA / norm(u_PA), u_PC / norm(u_PC), math.degrees(math.acos(np.dot(u_AB, u_CB)))

def repnth(s_in, n, sub, sep='(\s+)'):
    '''replace nth col of string with sth elase
    sub: a string or a list of strings'''
    if not isinstance(n, Iterable):
        sub = [sub]
        n = [n]
    s_in = re.sub('^\s+', '', s_in)
    res = re.split(sep, s_in)
    for i, x in zip(n, sub):
        res[2*(i-1)] = str(x)
    return  ''.join(res)

def gulp_elem_names(ln):
    tmp = re.findall('^.*?\s+(?=[\d-])', ln)[0].split()
    n = len(tmp)
    if n == 3:
        tmp = list(np.roll(tmp, 1))
    return tmp, n

def direc_list(direc, depth):
    files = []
    dir2 = os.path.join(direc, "/".join("*" * (depth+1)))
    return glob(dir2)
    
def get_cols(s, n, sep='\s+'):
    s = re.sub('^\s+', '', s)
    if not isinstance(n, Iterable):
        n = [n]
    n = [x-1 for x in n]
    return itemgetter(*n)(re.split(sep, s))
    
    
def write_pprofile(titles='', pp=None):
    pp.disable()
    pp.dump_stats('pprofile_tmp')
    txt = open('pprofile_tmp').read()
    if isinstance(titles, str):
        titles = [titles]
    out = ''
    for x in titles:
        out += re.findall(f'(?s){x}.*?(?=File:)', txt)[0] + '\n\n next func\n'
        
    open('pprofile', 'w').write(out)
    # Path('pprofile_tmp').unlink(missing_ok=1)
    
    
def equivalence_classes(objects, func):
    ''' func should take two arguments of objects and return true or false'''
    objects = list(objects)
    out = []
    used = np.zeros(len(objects))

    for i, obj in enumerate(objects):
        if not used[i]:
            eq_class = [(i,x) for i, x in enumerate(objects) if func(x, obj)]
            used[[x[0] for x in eq_class]] = 1
            out.append(eq_class)
    return [[y[1] for y in x] for x in out], [[y[0] for y in x] for x in out]


def gulp_output_params(f, read_current=1):
    nparms,flg = [],0
    types = []
    if isinstance(f, str):
        f = io.StringIO(f)
    for x in f:
        if 'Variables :' in x:
            [next(f) for _ in range(4)]
            for y in f:
                if '---' in y: break
                nparms.append([get_cols(y, 2)])
                types.append(' '.join(get_cols(y, [3, 4])))

        if 'Parameter        P' in x:
            [next(f) for _ in range(2)]
            for i, y in enumerate(f):
                if '---' in y: break
                nparms[i] = [nparms[i][0], get_cols(y, 3)]
                # types.append(' '.join(re.findall('[A-Za-z].*?(?=\s)', y)))

        if read_current and 'Current' in x:
            s = len(nparms)
            for i, y in enumerate(f):
                if flg:
                    nparms[i][1] = get_cols(y, 2)
                else:
                    nparms[i].append(get_cols(y, 2))
                if i+1 == s: break
            flg = 1

    return np.array(nparms, dtype=float), types


class cdict(UserDict):
    ''' you give it something like ['Ca', 'Si'] and it treats it as Counter'''
    def __setitem__(self, key, value):
        if isinstance(key, list):
            self.data[frozenset(Counter(key).items())] = value
        else:
            self.data[key] = value

    def __getitem__(self, key):
        if isinstance(key, list):
            return self.data[frozenset(Counter(key).items())]
        else:
            return self.data[key]

def gulp_kcal_conv(val, ikcal, okcal):
    if ikcal == okcal:
        return val
    elif ikcal and not okcal:
        return ev_to_kcalpmol(val, 1)
    else:
        return ev_to_kcalpmol(val)

def comb_params(params, ctype='sqrt'):
    idx = list(combinations(range(len(params)), 2))
    idx = [*enumerate(range(len(params))), *idx]
    out = np.zeros(len(idx))
    for i, x in enumerate(idx):
        if ctype == 'sqrt':
            out[i] = sqrt(params[x[0]] * params[x[1]])
        elif ctype == 'sum':
            out[i] = params[x[0]] + params[x[1]]

    idx = np.sort(idx, 1) + 1

    return idx, out


def stem(string):
    return re.search('[^\d]+', string).group()


def gen_grimme(nomix=0):
    # indices = [20, 1, 8, 14, 6]
    # conversion = 10.364425449557316
    atoms = ['Ca', 'H', 'O', 'Si', 'C']

    cc = dict(zip(atoms, [10.8, 0.14, 0.7, 9.23, 1.75]))
    r0 = dict(zip(atoms, [1.474, 1.001, 1.342, 1.716, 1.452]))

    for x in atoms:
        cc[x] = (cc[x] * u.joule * u.nanometer ** 6).in_units_of(u.angstrom ** 6 * u.kilocalorie)._value

    if nomix:
        return {x: .75*y for x, y in cc.items()}, r0

    n = len(atoms)
    out = {}
    for i in range(n):
        for j in range(i, n):
            t1, t2 = atoms[i], atoms[j]
            out[bt([t1, t2])] = {'c6': np.sqrt(cc[t1] * cc[t2])*0.75, 'r0': r0[t1] + r0[t2], 'd': 20, 'rcut': 12}  # 0.75: the global scaling factor used in vasp

    return out

def dictxml(d):
    out = ''

    for x in d:
        if isinstance(d[x], dict):
            if not np.any([isinstance(t, dict) for t in d[x].values()]):  # none of the dict members are dict
                out += f'<{x} '
                for z in d[x].items():
                    out += f'{z[0]}={z[1]} '
                out += f'/>'
            else:
                out += dictxml(d[x])
        else:
            out += f'<{x}>{d[x]}</{x}>\n'


def read_dict(file):
    return eval(open(file).read())


def prt_dict(d, lvl=0, file=''):

    out = '{'

    for x in d:
        r1, r2 = x.__repr__(), d[x].__repr__()
        if isinstance(d[x], dict):
            if np.any([isinstance(d[x][t], dict) for t in d[x]]):
                t = '\t' + prt_dict(d[x], lvl+1).replace('\n', '\n\t')
                out += f'{r1}: \n{(lvl+1)*"  "}{t},\n'
            else:
                out += f'{r1}: {r2},\n'
        else:
            out += f'{r1}: {r2},\n'
    out = out[:-2] + '}'

    tmp = re.findall('\d+\.\d+', out)
    for x in tmp:
        out = re.sub(f'{x}'+'(,|})', f'{float(x):.4f}'+r'\1', out)

    if file:
        open(file, 'w').write(out)
    else:
        return out




def get_shift(n, glp):
    '''glp: path of lammps executable'''
    out = open(f'gulp{n}.in').read()
    open('tmp', 'w').write(re.sub('\d\s+\d\s+#', '0 0 #', out))

    gout = shell(glp + ' < tmp')
    os.remove('tmp')
    f = iter(gout.split('\n'))
    for ln in f:
        if 'r      P' in ln:
            [next(f) for _ in range(2)]
            ln = next(f)
            ener = get_cols(ln, 3)
            break
    shell(f'''sed -i 's/shift .*/shift {ener}/' gulp{n}.in''')


def get_flags(ln):
    return re.findall(r'\d\s+\d(?=\s+#)', ln)[0].split()


def update_gulp_fit(file=1, prob_flg=0, ig_lupdate=0):
    ''' 
    iat: atom number
    no_file: I don't want a new file written
    prob_flg: returns 1 when either k or r0 are problematic
    ig_lupdate: ignore lupdate
    '''
    jj = max_file_num()

    # a = shell("squeue --noheader --format='%i' --user=morsheda")
    # try:
    #     b = re.search(f'.*?_\[?{iat}\]?', a).group()
    # except AttributeError:
    #     b = ''

    if not Path(f'gulp{jj}.out').exists():
        print(f'gulp{jj}.in exists but not gulp{jj}.out')
        shell('touch 1')
        exit(0)
        # if b:
        #     exit(0)
        # else:
        #     Path(f'gulp{jj}.in').unlink()
        # jj -= 1

    # if shell('cat lupdate') == f'{jj}' and not ig_lupdate:  # last updated
    #     exit(0)
    # else:
    #     open('lupdate', 'w').write(f'{jj}')
    
    nparms, types1 = gulp_output_params(open(f'gulp{jj}.out'), 1)
    if len(nparms) == 0 or nparms.shape[1] == 1:
        print(f'gulp{jj}.out is incomplete')
        shell('touch 2')
        exit(0)
        # if b:
        #     exit(0)
        # else:
        #     shell(f'rm gulp{jj}.out gulp{jj}.in')
        #     jj -= 1
        #     nparms, types1 = gulp_output_params(open(f'gulp{jj}.out'), 1)

    def set_flags(ln, f1, f2):
        return re.sub('\d\s+\d(?=\s+#)', f'{f1} {f2}', ln)

    f = open(f'gulp{jj}.in').read()
    flj = re.search(r'\d\s+\d(?=\s+#lj)', f).group().split() if 'lj' in f else [0]
    if '1' in flj:
        ftype = 1
    ff = re.search(r'\d\s+\d(?=\s+#[bad])', f).group().split()
    if ff[0] == '1':
        ftype = 3
    elif ff[1] == '1':
        ftype = 2

    cnt = 0  # flg2:0:remove the new gulp\d.in, 1:keep cause negative, 2:keep cause the run had fixed params
    out = ''
    sv_bnd = []
    flg1, flg2, flg3, cnt = 0, 0, 0, 0
    for ln in f.split('\n'):
        if 'lj' in ln:
            if ftype == 1:
                ln = repnth(ln, 2, nparms[cnt, 1])
                ln = set_flags(ln, 0, 0)
                cnt += 1
            elif ftype == 3:
                ln = set_flags(ln, 1, 0)

        if 'r0' in ln:
            flags = get_flags(ln)
            tmp, n = gulp_elem_names(ln)
            tol = .2 if n == 2 else 8  # tolerance for rejection or acceptance of r0
            if ftype == 2:
                r0 = abs(float(re.findall(r'(?<=r0:).*?(?=\s)', ln)[0]))
                if nparms[cnt, 1] > r0 + tol or nparms[cnt, 1] < r0 - tol:
                    nparms[cnt, 1] = r0
                    flg1 = 1
                ln = set_flags(ln, 1, 0)

            if ftype == 3:
                k0 = abs(float(re.findall(r'(?<=k:).*?[\n\s]', ln+' ')[0]))
                nparms[cnt, :] *= 23.06054972536769
                if nparms[cnt, 1] > 2*k0 or nparms[cnt, 1] < k0/2 or nparms[cnt, 1] > 1300:
                    nparms[cnt, 1] = k0
                    flg2 = 1

                if 'atomab' in f:  # you have repulsion
                    ln = set_flags(ln, 0, 0)
                else:
                    ln = set_flags(ln, 0, 1)

            if ftype != 1:
                ln = repnth(ln, n + (1 if ftype == 3 else 2), nparms[cnt, 1])  # sub k or r
                cnt += 1
            else:
                ln = set_flags(ln, 0, 1)

        if 'shift ' in ln:
            ln = repnth(ln, 2, nparms[-1, 1])
        out += ln + '\n'

    out = out.rstrip('\n')
    if not file:
        return out


        # open('fle', 'w').write((f' has negative stiffness' if flg1 else '') +
        #                        (f' & has problematic r0 for {sv_bnd}' if sv_bnd else ''))
    if ftype == 3:
        f = open(f'gulp{jj - 1}.out')
        nparams2, types2 = gulp_output_params(f, 1)
        types = [*types1, *types2]
        params = np.vstack([nparms, nparams2])

        for i, tp in enumerate(types):
            tol = .05 if 'r0' in tp else 5

            if abs(params[i, 1] - params[i, 0]) > tol:
                flg3 = 1
                break
    if flg1 or flg2 or jj < 2 or ftype != 3 or flg3 == 1:
        # shell(f'scancel {b}')
        open(f'gulp{jj + 1}.in', 'w').write(out)


def max_file_num(direc='.', pat = '(?<=gulp)\d+(?=.in)'):
    direc = str(direc)
    files = os.listdir(direc) #  next(os.walk(direc))[-1]
    nums = []
    for x in files:
        y = re.findall(pat, x)
        if y:
            nums.append(int(y[0]))
    if not nums:
        return ''
    else:
        return np.max(nums)


def find_file(name, paths):
    results = []
    for path in paths:
        for root, dirs, files in os.walk(path):
            if name in files:
                results.append(os.path.join(root, name))
    return results


def get_vasp_freqs(direc, nunit=2, clean=1):
    '''
    unit: units of output (each column): THz - 2PiTHz - cm-1  - meV
    clean: make it into a matrix for writing in gulp
    '''

    command = f"awk '/2PiTHz/{{if (\"f\"==$2) {{print $4, $6, $8, $10}} else {{print $3, $5, $7, $9}}}}' {os.path.join(direc,'')}OUTCAR"
    file = shell(command)
    t = np.sort(txt_to_mat(file)[:, nunit])
    return np.vstack([arange(3,len(t)), t[3:]]).T if clean else t

def get_gulp_freqs_eigs2(direc='gulp.out'):
    '''eigs needs some fixing'''

    n = shell('''awk '/Number of irred/ {print $6;exit}' ''' + direc)
    n = int(n)
    # f = open(direc)
    out = open(direc).readlines()
    # freqs = re.search('(?s)(?<=Frequencies \(cm-1\) :).*(?=No)', out).group().replace('\n', ' ')

    reg = re.compile('^\s*Frequency\s*(?=-?\d)')
    reg2 = re.compile('(?<=[xyz]).*')
    freqs = ''
    flg = 0
    eigstr = np.zeros([3*n, 3*n])
    cnt, lower = 0, 0
    lbl = dict()
    for i in range(1, n+1):
        for j, a in enumerate(['x', 'y', 'z']):
            lbl[f'{i}{a}'] = (i-1)*3 + j

    fcnt = 0
    while cnt < len(out):
        if reg.match(out[cnt]):
            freqs0 = re.findall(r'(?<=Frequency).*', out[cnt])[0]
            freqs += freqs0
            cnt += 7
            idx1 = lbl[out[cnt][:8].replace(' ', '')]
            txt = ''
            while 1:
                txt += out[cnt][8:]
                cnt += 1
                if out[cnt] == '\n':
                    idx2 = lbl[out[cnt-1][:8].replace(' ', '')] + 1
                    lf = len(freqs0.split())
                    eigstr[idx1:idx2, fcnt:fcnt+lf] = txt_to_mat(txt)
                    fcnt += lf
                    break
        cnt += 1

    return txt_to_mat(freqs), eigstr

def get_gulp_freqs_eigs3(direc='gulp.out'):
    n = shell('''awk '/Number of irred/ {print $6;exit}' ''' + direc)
    n = int(n)
    out = open(direc).read()
    aa = re.findall('\ +\d+\s+[xyz]\s+.*', out)

    vals = dict()
    for i in range(n):
        for j in ['x', 'y', 'z']:
            vals[f'{i+1} {j}'] = ''

    for x in aa:
        s = x.split()
        # if f' {" ".join(s[2:])} ' in vals[' '.join(s[:2])] :
        #     breakpoint()
        vals[' '.join(s[:2])] += f' {" ".join(s[2:])} '
    for x, y in vals.items():
        if len(y.split()) != 6912:
            print(x, len(y.split()))
    breakpoint()


def get_gulp_freqs_eigs(direc='gulp.out'):
    '''eigs needs some fixing'''

    n = shell('''awk '/Number of irred/ {print $6;exit}' ''' + direc)
    n = int(n)
    f = open(direc)
    out = open(direc).read()
    # freqs = re.search('(?s)(?<=Frequencies \(cm-1\) :).*(?=No)', out).group().replace('\n', ' ')

    reg = re.compile('^\s*Frequency\s*(?=-?\d)')
    reg2 = re.compile('(?<=[xyz]).*')
    freqs = ''
    flg = 0
    eigstr = ['' for _ in range(3*n)]
    for x in f:
        if reg.match(x):
            freqs += re.findall(r'(?<=Frequency).*', x)[0]
            f = islice(f, 6, None)
            tmp = ''
            for i, y in enumerate(f):
                if y == '\n':
                    break
                eigstr[i] += reg2.findall(y)[0]

    return txt_to_mat(freqs), txt_to_mat('\n'.join(eigstr))

def get_gulp_thermal(direc='gulp.out'):
    n = shell('''awk '/Number of irred/ {print $6;exit}' ''' + direc)
    n = int(n)
    out = open(direc).readlines()

    cnt = 0
    while cnt < len(out):
        if 'Mode    : Frequency' in out[cnt]:
            tmp = ''
            for cc in range(cnt+3, cnt+3*n):
                if '---' in out[cc]:
                    break
                tmp += out[cc]
            cnt = cc
        cnt += 1
    return txt_to_mat(tmp)

def allen_feldman(direc='gulp.out'):

    evtoj, avogadro, speedl, planck, boltz = 1.6021917e-19, 6.022045e23, 2.99792458e10, 6.62617636e-34, 1.38066244e-23
    #                cm/s     Js      J/K
    fcut = 2.5e12 / speedl

    fscale = sqrt(1.0e23 * evtoj * avogadro) / (2.0 * pi * speedl)
    constant = 1e-20 * (1.0e23 * evtoj * avogadro) ** 0.5 * fscale ** 3 / 48 * pi

    temperature, kaps, cv = 300, [], []

    broad = 2
    tmp = gulp_hessian(direc)  # latmat, coords, frcs, derv2, freqs, eigr, elems, masses
    comp = from_gulp(direc.replace('out', 'in'))
    latmat, coords, elems, masses = comp.latmat, comp.xyz, [x.elem for x in comp.particles()], [x.mass for x in comp.particles()]
    derv2 = gulp_hessian(direc)
    freqs, eigr = get_gulp_freqs_eigs(direc)

    # freqs = np.sort(freqs)
    # freqs[freqs < fcut] = 0
    # eigr = eigr[:, -1::-1]

    n = len(elems)
    # for i in range(n):
    #     for j in range(n):
    #         derv2[3 * i:3 * i + 3, 3 * j:3 * j + 3] /= sqrt(masses[i] * masses[j])

    ndof = len(freqs)
    fidx = freqs > 0
    # ff = freqs[fidx]
    # ff = freqs[8:45]
    # dwavg = np.sum(ff - np.insert(ff[:-1], 0, 0)) / (ndof - 1)
    # broad = dwavg  # .2

    # broad = .4
    # go.Figure(go.Histogram(x=freqs,nbinsx=100)).show()

    # !  Create inverse frequency factors while trapping translations and imaginary modes
    freqinv = np.zeros(ndof)
    # nfreqmin = np.argmax(fidx)
    # nfreqmin = 20
    freqinv[fidx] = 1 / sqrt(freqs[fidx])

    nats = ndof // 3

    def func(ixyz):
        Di = np.zeros(ndof)
        cnt = ixyz * nats * (nats - 1) // 2
        Vij = np.zeros([ndof, ndof])
        tt = []
        for ig in range(nats):
            ix = 3 * ig
            for jg in range(ig):
                jx = 3 * jg
                r = nearestr(coords[jg, :] - coords[ig, :], latmat)
                dxyz = r[ixyz]
                Vij[jx:jx + 3, ix:ix + 3] = derv2[jx:jx + 3, ix:ix + 3] * dxyz
                Vij[ix:ix + 3, jx:jx + 3] = - derv2[ix:ix + 3, jx:jx + 3] * dxyz
                cnt += 1
                tt.append(dxyz)

        Sij = einsum('ij,i,j,ij->ij', eigr.T @ -Vij @ eigr, freqinv, freqinv, freqs[None, :] + freqs[:, None])

        Lor_tol = 1e-2
        lor = 1.0 / np.pi * broad / ((freqs[:, None] - freqs[None, nfreqmin:]) ** 2 + broad ** 2)
        Di[nfreqmin:] = constant * np.sum(lor * Sij[:, nfreqmin:] ** 2 * (lor > Lor_tol), 0) / freqs[nfreqmin:] ** 2
        return Di


    out = Parallel(1)([delayed(func)(i) for i in range(3)])
    Di = np.sum(out, axis=0)
    # go.Figure(go.Scatter(x=freqs, y=Di, mode='markers')).show()
    kappafct = 1.0e30 / det(latmat)  # 1/vol


    cfreq = planck * speedl / (2 * boltz * temperature) * freqs[nfreqmin:]
    kappa_af = kappafct * np.sum(boltz * (cfreq / np.sinh(cfreq)) ** 2 * Di[nfreqmin:])

    kaps.append(kappa_af)

    cv.append(np.sum(boltz * (cfreq / np.sinh(cfreq)) ** 2)/sum(masses)*avogadro)  # J/K/g
    # fig = go.Figure(go.Scatter(x=np.arange(.2, 10, .1), y=kaps[0], name='hex'))
    # fig.add_trace(go.Scatter(x=np.arange(.2, 10, .1), y=kaps[1], name='oct'))
    # fig.show()
    print(kaps)


def closest_num(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def angle_between_vecs(v1, v2, degrees=1):
    v1, v2 = map(np.array,[v1, v2])
    t = v1 @ v2/(norm(v1) * norm(v2))
    if t > 1 and t < 1.001:
        t = 0
    ang = np.arccos(t)

    return np.degrees(ang) if degrees else ang


def optim_half(f, tol=1e-6, x0=0, maxiter=200, dx0=1):

    iconv = 0
    y0 = f(x0)
    if abs(y0) > tol:
        x1, y1 = x0+dx0, f(x0+dx0)
        for i in range(maxiter):

            df = (y1 - y0) / (x1 - x0)
            if np.isclose(x0, x1) or df == 0:
                while 1:
                    x0, x1 = (x0 + x1)/2 - .05, (x0 + x1)/2 + .05
                    if (y0:=f(x0))*(y1:=f(x1)) < 0:
                        break
                while abs(y1) > tol:
                    i += 1
                    c = (x0 + x1)/2
                    if y0*(y1:=f(c)) < 0:
                        x1 = c
                    else:
                        x0 = c
                        y0 = y1
                    if i == maxiter:
                        break
                iconv = 1

            if iconv:
                break
            x0, x1 = x1, x1 - y1/df
            y0, y1 = y1, f(x1)
            if abs(y1) <= tol:
                break
    else:
        x1, y1 = x0, y0
    return x1, y1

# CONV_FACT = ev_to_kcalpmol(1e10 * 1.602176634e-19 / (4 * pi * 8.8541878128e-12))  ## Converts unit of q*q/r into kcal/mol               eps_0: vacuum electric permittivity
