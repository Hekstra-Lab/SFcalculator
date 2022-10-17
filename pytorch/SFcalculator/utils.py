import torch
import numpy as np
import gemmi

def r_factor(Fo, Fmodel, rwork_id, rfree_id):
    '''
    A function to calculate R_work and R_free

    Parameters
    ----------
    Fo: torch.tensor, [N_hkl,], complex
        1D tensor containing Fo corresponding to HKL list
    Fmodel: torch.tensor, [N_hkl,], complex
        1D tensor containing Fmodel corresponding to HKL list
    rwork_id: np.array, [N_work,]
        1D array, a list of index used to do work calculation, usually come from np.argwhere()
    rfree_id: np.array, [N_free,]
        1D array, a list of index used to do test calculation, usually come from np.argwhere()

    Returns
    -------
    R_work, R_free: Both are floats
    '''
    R_work = torch.sum(torch.abs(Fo[rwork_id] - Fmodel[rwork_id])) / torch.sum(Fo[rwork_id])
    R_free = torch.sum(torch.abs(Fo[rfree_id] - Fmodel[rfree_id])) / torch.sum(Fo[rfree_id])
    return R_work, R_free

def diff_array(a, b):
    '''
    Return the elements in a but not in b, when a and b are array-like object

    Parameters
    ----------
    a: array-like
       Can be N Dimensional

    b: array-like

    return_diff: boolean
       return the set difference or not

    Return
    ------
    Difference Elements
    '''
    tuplelist_a = list(map(tuple, a))
    tuplelist_b = list(map(tuple, b))
    set_a = set(tuplelist_a)
    set_b = set(tuplelist_b)
    return set_a - set_b

def asu2HKL(Hasu_array, HKL_array):
    '''
    A fast way to find indices convert array Hasu to array HKL
    when both Hasu and HKL are 2D arrays. 
    HKL is the subset of Hasu.
    Involves two steps:
    1. an evil string coding along axis1, turn the 2D array into 1D
    2. fancy sortsearch on two 1D arrays
    '''
    def tostr(array):
        string = ""
        for i in array:
            string += "_"+str(i)
        return np.asarray(string, dtype='<U20')

    HKL_array_str = np.apply_along_axis(tostr, axis=1, arr=HKL_array)
    Hasu_array_str = np.apply_along_axis(tostr, axis=1, arr=Hasu_array)
    xsorted = np.argsort(Hasu_array_str)
    ypos = np.searchsorted(Hasu_array_str[xsorted], HKL_array_str)
    indices = xsorted[ypos]
    return indices

def DWF_iso(b_iso, dr2_array):
    '''
    Calculate Debye_Waller Factor with Isotropic B Factor
    DWF_iso = exp(-B_iso * dr^2/4), Rupp P640, dr is dhkl in reciprocal space

    Parameters:
    -----------
    b_iso: 1D tensor, float32, [N_atoms,]
        Isotropic B factor

    dr2_array: numpy 1D array or 1D tensor, [N_HKLs,]
        Reciprocal d*(hkl)^2 array, corresponding to the HKL_array

    Return:
    -------
    A 2D [N_atoms, N_HKLs] float32 tensor with DWF corresponding to different atoms and different HKLs
    '''
    return torch.exp(-b_iso.view([-1, 1])*dr2_array/4.).type(torch.float32)


def DWF_aniso(b_aniso, reciprocal_cell_paras, HKL_array):
    '''
    Calculate Debye_Waller Factor with anisotropic B Factor, Rupp P641
    DWF_aniso = exp[-2 * pi^2 * (U11*h^2*ar^2 + U22*k^2*br^2 + U33*l^2cr^2
                                 + 2U12*h*k*ar*br*cos(gamma_r)
                                 + 2U13*h*l*ar*cr*cos(beta_r)
                                 + 2U23*k*l*br*cr*cos(alpha_r))]

    Parameters:
    -----------
    b_aniso: 2D tensor float32, [N_atoms, 6]
        Anisotropic B factor U, [[U11, U22, U33, U12, U13, U23],...], of diffferent particles

    reciprocal_cell_paras: list of float or tensor float, [6,]
        Necessary info of Reciprocal unit cell, [ar, br, cr, cos(alpha_r), cos(beta_r), cos(gamma_r)

    HKL_array: array of HKL index, [N_HKLs,3]

    Return:
    -------
    A 2D [N_atoms, N_HKLs] float32 tensor with DWF corresponding to different atoms and different HKLs
    '''
    # U11, U22, U33, U12, U13, U23 = b_aniso
    h, k, l = HKL_array.T
    ar, br, cr, cos_alphar, cos_betar, cos_gammar = reciprocal_cell_paras
    log_value = -2 * np.pi**2 * (b_aniso[:, 0][:, None] * h**2 * ar**2
                                 + b_aniso[:, 1][:, None] * k**2 * br**2
                                 + b_aniso[:, 2][:, None] * l**2 * cr**2
                                 + 2*b_aniso[:, 3][:, None] *
                                 h*k*ar*br*cos_gammar
                                 + 2*b_aniso[:, 4][:, None]*h*l*ar*cr*cos_betar
                                 + 2*b_aniso[:, 5][:, None]*k*l*br*cr*cos_alphar)
    DWF_aniso_vec = torch.exp(log_value)
    return DWF_aniso_vec.type(torch.float32)

def vdw_rad_tensor(atom_name_list):
    '''
    Return the vdw radius tensor of the atom list
    '''
    unique_atom = list(set(atom_name_list))
    vdw_rad_dict = {}
    for atom_type in unique_atom:
        element = gemmi.Element(atom_type)
        vdw_rad_dict[atom_type] = torch.tensor(element.vdw_r)
    vdw_rad_tensor = torch.tensor([vdw_rad_dict[atom] for atom in atom_name_list]).type(torch.float32)
    return vdw_rad_tensor

def vdw_distance_matrix(atom_name_list):
    '''
    Calculate the minimum distance between atoms by vdw radius
    Use as a criteria of atom clash

    Parameters
    ----------
    atom_name_list: array-like, [N_atom,]
        atom names in order, like ['C', 'N', 'C', ...]

    Returns
    -------
    A matrix with [N_atom, N_atom], value [i,j] means the minimum allowed 
    distance between atom i and atom j
    '''
    vdw_rad = vdw_rad_tensor(atom_name_list)
    vdw_min_dist = vdw_rad[None, :] + vdw_rad[:, None]
    return vdw_min_dist

def nonH_index(atom_name_list):
    '''
    Return the index of non-Hydrogen atoms
    '''
    index = np.argwhere(np.array(atom_name_list) != 'H').reshape(-1)
    return index

def unitcell_grid_center(unitcell, spacing=4.5, frac=False, return_tensor=True):
    '''
    Create a grid in real space given a unitcell and spacing
    output the center positions of all grids

    Parameters
    ----------
    unitcell: gemmi.UnitCell
        A unitcell instance containing size and fractionalization/orthogonalization matrix

    spacing: float, default 4.5
        grid size

    frac: boolean, default False
        If True, positions are in fractional space; Otherwise in cartesian space

    return_tensor: boolean, default True
        If True, convert to tf.tensor and return

    Returns
    -------
    [N_grid, 3] array, containing center positions of all grids
    '''
    a, b, c, _, _, _ = unitcell.parameters
    na = int(a/spacing)
    nb = int(b/spacing)
    nc = int(c/spacing)
    u_list = np.linspace(0, 1, na)
    v_list = np.linspace(0, 1, nb)
    w_list = np.linspace(0, 1, nc)
    unitcell_grid_center_frac = np.array(
        np.meshgrid(u_list, v_list, w_list)).T.reshape(-1, 3)
    if frac:
        result = unitcell_grid_center_frac
    else:
        result = np.dot(unitcell_grid_center_frac, np.array(
            unitcell.orthogonalization_matrix).T)

    if return_tensor:
        return torch.tensor(result)
    else:
        return result
