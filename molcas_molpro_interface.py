from read_molcas_h5 import *
from lut import element_Z_from_sym
from misc import contraction_string
import re


toang = 0.529177

file = '/home/katukuri/work/Molcas/CaCuO2/1site/ANO-R-Zn-sp/B1g/MCPDFT/mcpdft.rasscf.h5'
unique_atoms = [1, 2, 6, 14, 20]
point_charges_file = 'cacuo2.evjen.lat'
ecp_file = 'ecp.dat'
molpro_inp_file = 'molpro.inp'


prefix = '''***, Title
MEMORY, 1000,M;

GPRINT, BASIS

GDIRECT

!Symmetry, NOSYM
ANGSTROM
GEOMETRY={
'''
out_file = open(molpro_inp_file,'w')
print(prefix,end='',file=out_file)

orbs = Orbitals(file,'hdf5')
basis = []
z = []
names = []
coo = []
# read all the information from the molcas orbitals class
for i in range(len(orbs.centers)):
    basis.append(orbs.centers[i]['basis'])
    z.append(orbs.centers[i]['Z'])
    names.append(orbs.centers[i]['name'])
    coo.append(orbs.centers[i]['xyz'])

names_new = []
for name in names:
    names_new.append(re.sub(r'\d+', '', name.split()[0]))

actual_z = []
for name in names_new:
    actual_z.append(element_Z_from_sym(name))
print(actual_z)

print(len(coo), file=out_file)
print('', file=out_file)
for j, xyz in enumerate(coo):
    if j+1 in unique_atoms:
        suffix = unique_atoms.index(j+1) + 1
    print('%s  %3.7f %3.7f  %3.7f'%(names_new[j]+str(suffix), toang*xyz[0], toang*xyz[1], toang*xyz[2]),file=out_file)
print('}', file=out_file)
if point_charges_file :
    print('{lattice,usesym,infile='+point_charges_file+'}', file=out_file)

#check if ECPS are present
ecp = []
for i in range(len(basis)):
    if element_Z_from_sym(names_new[i]) != z[i]:
        ecp.append(True)
        hasecp = True
    else:
        ecp.append(False)

def reformat_basis(basis):
    data = dict()
    data['electron_shells'] = []
    # print(len(basis))
    nam = 0
    for l in basis:
        if len(l) != 0:
            nam += 1
    for am in range(nam):
        shell = {}
        exp = [str(basis[am][0][1][k][0]) for k in range(len(basis[am][0][1]))]
        shell['angular_momentum'] = [am]
        shell['exponents'] = exp
        coeff = []
        for n_cont in range(len(basis[am])):
            coeff.append([str(basis[am][n_cont][1][k][1]) for k in range(len(basis[am][n_cont][1]))])
        shell['coefficients'] = coeff
        data['electron_shells'].append(shell)
    return data


def create_data_old(basis):
    data = dict()
    data['electron_shells'] = []
    for n_am in range(len(basis)):
        shell = {}
        exp = [basis[n_am][0][1][k][0] for k in range(len(basis[n_am][0][1]))]
        shell['angular_momentum'] = [n_am]
        shell['exponents'] = exp
        coeff = []
        for n_cont in range(len(basis[n_am])):
            # print(a1_basis[n_am])
            # print([a1_basis[n_am][n_cont][1][k][1] for k in range(len(a1_basis[n_am][n_cont][1]))])
            coeff.append([str(basis[n_am][n_cont][1][k][1]) for k in range(len(basis[n_am][n_cont][1]))])
        shell['coefficients'] = coeff
        data['electron_shells'].append(shell)
    for shell in data['electron_shells']:
        print(shell)
    return data


s = 'basis={\n'
for i, b in enumerate(basis):
    if i+1 in unique_atoms :
        suffix = str(unique_atoms.index(i+1) + 1)
        if not ecp[i]:
            data = reformat_basis(b)
            # s += '! {:20} {}\n'.format(names_new[i] + suffix, contraction_string(data))
            s += write_molpro(names_new[i] + suffix, data=data)
            # print(s)
        else:
            data = reformat_basis(b)
            # s += '! {:20} {}\n'.format(names_new[i] + suffix, contraction_string(data))
            s += write_molpro(names_new[i] + suffix, data=data)
            # print(s)
if hasecp:
    with open(ecp_file) as fp:
        ecp_dat = fp.read()

    s += ecp_dat
s += '}\n'
print(s,file=out_file)
out_file.close()