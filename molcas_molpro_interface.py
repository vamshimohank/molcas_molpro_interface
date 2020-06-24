toang = 0.529177


def read_basis_data_from_h5(file=''):
    from read_molcas_h5 import Orbitals
    from lut import element_Z_from_sym

    import re

    orbs = Orbitals(file, 'hdf5')
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
    # print(actual_z)

    return coo, names_new, z, actual_z, basis


# check if ECPS are present
def check_for_ecps(basis, q):
    from lut import element_Z_from_sym
    ecp = []
    hasecp = False
    for i in range(len(basis)):
        if element_Z_from_sym(names[i]) != q[i]:
            ecp.append(True)
            hasecp = True
        else:
            ecp.append(False)
    return hasecp, ecp


def reformat_basis(basis):
    '''
    Reformats the basis read from the hdf5 file. This is essential to use the basis_set_exchange molpro writer
    :param basis: basis read from hdf5 file
    :return: basis data in the format as in basis_set_exchange
    '''

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
    '''
    Obselete function
    :param basis:
    :return:
    '''
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


def create_molpro_input(coo, names, basis, molpro_inp_file, hasecp=False, pc_file=''):
    from read_molcas_h5 import write_molpro
    out_file = open(molpro_inp_file, 'w')

    ecp_file = 'ecp.dat'
    prefix = '''***, Title\n MEMORY, 1000,M; \n \nGPRINT, BASIS \nGDIRECT\n!Symmetry, NOSYM \nANGSTROM \nGEOMETRY={
    '''

    print(prefix, end='', file=out_file)

    print(len(coo), file=out_file)
    print('', file=out_file)
    suffix = ''
    for j, xyz in enumerate(coo):
        if j + 1 in unique_atoms:
            suffix = unique_atoms.index(j + 1) + 1
        print('%s  %3.7f %3.7f  %3.7f' % (names[j] + str(suffix), toang * xyz[0], toang * xyz[1], toang * xyz[2]),
              file=out_file)
    print('}', file=out_file)

    if pc_file != '':
        print('{lattice,usesym,infile=' + pc_file + '}', file=out_file)

    s = 'basis={\n'
    for i, b in enumerate(basis):
        if i + 1 in unique_atoms:
            suffix = str(unique_atoms.index(i + 1) + 1)
            if not ecp[i]:
                data = reformat_basis(b)
                s += write_molpro(names[i] + suffix, data=data)
            else:
                data = reformat_basis(b)
                s += write_molpro(names[i] + suffix, data=data)
    if hasecp:
        with open(ecp_file) as fp:
            ecp_dat = fp.read()
        print('ECPs are taken for the provided ecp.dat file')
        s += ecp_dat
    s += '}\n'
    print(s, file=out_file)
    out_file.close()


def ao_ovlp_evals(file):
    '''
    Given a hdf5 file of molcas calculation, this module computes the eigenvalues of the overlap matrix symmetry irrep. wise
    :param file: *.h5 file from molcas run
    :return: evals: a list of dimension #irreps. each element contains eigenvalues of that particular irrep.
     evecs: a list of of dimension #irreps., each containing eigenvectors of that particular irrep.
    '''
    import h5py
    import numpy as np

    hf = h5py.File(file, 'r')
    ao_ovlp = np.array(hf.get('AO_OVERLAP_MATRIX'))
    nsym = hf.attrs['NSYM']
    nbas = hf.attrs['NBAS']

    ao_ovlp_dim = np.asarray([np.square(i) for i in nbas])
    print(ao_ovlp_dim)

    ovlp_mats = []
    for i in range(nsym):
        temp = ao_ovlp[np.sum(ao_ovlp_dim[:i]): np.sum(ao_ovlp_dim[:i + 1])]
        ovlp_mats.append(np.reshape(temp, (-1, nbas[i])))
        print(ovlp_mats[i].shape)

    evals = []
    evecs = []
    for i in range(nsym):
        e, ev = np.linalg.eig(ovlp_mats[i])
        idx = e.argsort()
        e = e[idx]
        ev = ev[:, idx]
        evals.append(e)
        evecs.append(ev)
    return evals, evecs


if __name__ == '__main__':
    unique_atoms = [1, 2, 6, 14, 20]
    point_charges_file = 'cacuo2.evjen.lat'
    file = '/home/katukuri/work/Molcas/CaCuO2/1site/ANO-R-Zn-sp/B1g/MCPDFT/mcpdft.rasscf.h5'

    eval, evecs = ao_ovlp_evals(file)
    print(eval)
    # molpro_inp_file = 'molpro.inp'
    #
    # coo, names, q, z, basis = read_basis_data_from_h5(file=file)
    # hasecp, ecp = check_for_ecps(basis, q)
    # # print(hasecp, ecp)
    # create_molpro_input(coo, names, basis, molpro_inp_file, hasecp=hasecp, pc_file=point_charges_file)
