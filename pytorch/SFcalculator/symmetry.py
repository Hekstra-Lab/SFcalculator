import numpy as np
import torch
import reciprocalspaceship as rs
import pandas as pd

from .utils import try_gpu

ccp4_hkl_asu = [
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 7, 6, 7, 6, 7, 7, 7,
    6, 7, 6, 7, 7, 6, 6, 7, 7, 7, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9
]

asu_cases = {
    0: lambda h, k, l: (l > 0) | ((l == 0) & ((h > 0) | ((h == 0) & (k >= 0)))),
    1: lambda h, k, l: (k >= 0) & ((l > 0) | ((l == 0) & (h >= 0))),
    2: lambda h, k, l: (h >= 0) & (k >= 0) & (l >= 0),
    3: lambda h, k, l: (l >= 0) & (((h >= 0) & (k > 0)) | ((h == 0) & (k == 0))),
    4: lambda h, k, l: (h >= k) & (k >= 0) & (l >= 0),
    5: lambda h, k, l: ((h >= 0) & (k > 0)) | ((h == 0) & (k == 0) & (l >= 0)),
    6: lambda h, k, l: (h >= k) & (k >= 0) & ((k > 0) | (l >= 0)),
    7: lambda h, k, l: (h >= k) & (k >= 0) & ((h > k) | (l >= 0)),
    8: lambda h, k, l: (h >= 0) & (((l >= h) & (k > h)) | ((l == h) & (k == h))),
    9: lambda h, k, l: (k >= l) & (l >= h) & (h >= 0),
}

def expand_to_p1(spacegroup, Hasu_array, Fasu_tensor, dmin_mask=6.0, Batch=False, unitcell=None):
    '''
    Expand the reciprocal ASU array to a complete p1 unit cell, with phase shift on the Complex Structure Factor
    In a fully differentiable manner (to Fasu_tensor), with tensorflow

    Parameters:
    -----------
    spacegroup: gemmi.SpaceGroup
        A gemmi spacegroup object

    Hasu_array: np.int32 array
        The HKL list of reciprocal ASU

    Fasu_tensor: torch.complex64 tensor, single model or batched model
        Corresponding structural factor tensor

    Batch: Boolean, Default False
        Use to show if a single or a batch of Fasu_tensor are given. If True, the Fasu_tensor 
        should be in shape [N_batch, N_Hasu]; If False, the Fasu_tensor should be in shape
        [N_Hasu,]

    dmin_mask: np.float32, Default 6 angstroms.
        Minimum resolution cutoff, in angstroms, for creating the solvent mask

    Return:
    -------
    Hp1_array, Fp1_tensor
        HKL list in p1 unit cell and corresponding complex structural factor tensor
    '''

    groupops = spacegroup.operations()
    allops = [op for op in groupops]

    if Batch:
        # Batched calculation
        assert Fasu_tensor.dim() == 2, "Give batch Fasu if you set Batch=True!"
        concat_axis = 1
    else:
        # Single model calculation
        assert Fasu_tensor.dim() == 1, "Give single Fasu if you set Batch=False!"
        concat_axis = 0
    if dmin_mask is not None:
        # expands to p1 with resolution set by dmin_mask, to remove high-frequency noise.
        dHKL = unitcell.calculate_d_array(Hasu_array).astype("float32") # type: ignore
        new_hkl_inds_bool = (dHKL >= dmin_mask)
        Hasu_array = Hasu_array[new_hkl_inds_bool]

        # removes entries of Fasu_tensor that correspond to resolutions above dmin_mask
        if Batch:
            Fasu_tensor = Fasu_tensor[:,new_hkl_inds_bool]
        else: 
            Fasu_tensor = Fasu_tensor[new_hkl_inds_bool]

    Fp1_tensor = Fasu_tensor
    len_asu = len(Hasu_array)
    Hasu_tensor = torch.tensor(Hasu_array, device=try_gpu()).type(torch.float32)

    ds = pd.DataFrame()
    ds["H"] = Hasu_array[:, 0]
    ds["K"] = Hasu_array[:, 1]
    ds["L"] = Hasu_array[:, 2]
    ds["index"] = np.arange(len_asu)
    zero_tensor = torch.tensor(0.,device=try_gpu())
    for i, op in enumerate(allops):
        if i == 0:
            continue
        rot_temp = np.array(op.rot)/op.DEN
        tran_temp = torch.tensor(np.array(op.tran)/op.DEN, device=try_gpu()).type(torch.float32)
        H_temp = np.matmul(Hasu_array, rot_temp).astype(np.int32)
        ds_temp = pd.DataFrame()
        ds_temp["H"] = H_temp[:, 0]
        ds_temp["K"] = H_temp[:, 1]
        ds_temp["L"] = H_temp[:, 2]
        ds_temp["index"] = np.arange(len_asu)+len_asu*i
        # exp(-2*pi*j*h*T)
        phaseshift_temp = torch.exp(torch.complex(
            zero_tensor, -2*np.pi*torch.tensordot(Hasu_tensor, tran_temp, dims=1)))
        Fcalc_temp = Fasu_tensor * phaseshift_temp
        ds = pd.concat([ds, ds_temp])
        Fp1_tensor = torch.concat((Fp1_tensor, Fcalc_temp), dim=concat_axis)

    # Friedel Pair
    ds_friedel = ds.copy()
    ds_friedel["H"] = -ds_friedel["H"]
    ds_friedel["K"] = -ds_friedel["K"]
    ds_friedel["L"] = -ds_friedel["L"]
    ds_friedel["index"] = ds_friedel["index"] + len(ds)
    F_friedel_tensor = torch.conj(Fp1_tensor)

    # Combine
    ds = pd.concat([ds, ds_friedel])
    Fp1_tensor = torch.concat((Fp1_tensor, F_friedel_tensor), dim=concat_axis)

    ds = ds.drop_duplicates(subset=["H", "K", "L"])

    HKL_1 = ds[["H", "K", "L"]].values
    idx_1 = torch.tensor(ds["index"].values, device=try_gpu())
    Fp1_tensor = torch.index_select(Fp1_tensor, dim=concat_axis, index=idx_1)
    in_asu = asu_cases[0]  # p1 symmetry
    idx_2 = in_asu(*HKL_1.T)
    HKL_2 = HKL_1[idx_2]
    idx_2i = torch.where(torch.tensor(idx_2, device=try_gpu()))[0]
    Fp1_tensor = torch.index_select(Fp1_tensor, dim=concat_axis, index=idx_2i)

    return HKL_2, Fp1_tensor


def generate_reciprocal_asu(cell, spacegroup, dmin, anomalous=False):
    """
    Generate the Miller indices of the reflections in the reciprocal ASU.
    If `anomalous=True` the Miller indices of acentric reflections will be
    included in both the Friedel-plus and Friedel-minus halves of reciprocal
    space. Centric Miller indices will only be included in the Friedel-plus
    reciprocal ASU.
    Parameters
    ----------
    cell : gemmi.UnitCell
        UnitCell object
    spacegroup : str, int, gemmi.SpaceGroup
        Space group to identify asymmetric unit
    dmin : float
        Maximum resolution of the data in Å
    anomalous : bool
        Whether to include Friedel-minus Miller indices to represent anomalous data
    Returns
    -------
    hasu : np.array (np.int32)
        n by 3 array of miller indices in the reciprocal ASU.
    """
    p1_hkl = generate_reciprocal_cell(cell, dmin)
    # Remove absences
    hkl = p1_hkl[~rs.utils.is_absent(p1_hkl, spacegroup)] #type: ignore
    # Map to ASU
    hasu = hkl[rs.utils.in_asu(hkl, spacegroup)] #type: ignore
    if anomalous:
        hasu_minus = -hasu[~rs.utils.is_centric(hasu, spacegroup)] #type: ignore
        return np.unique(np.concatenate([hasu, hasu_minus]), axis=0)
    return np.unique(hasu, axis=0)


def generate_reciprocal_cell(cell, dmin, dtype=np.int32):
    """
    Generate the miller indices of the full P1 reciprocal cell.
    Parameters
    ----------
    cell : gemmi.UnitCell
        Unit cell object
    dmin : float
        Maximum resolution of the data in Å
    dtype : np.dtype (optional)
        The data type of the returned array. The default is np.int32.
    Returns
    -------
    hkl : np.array(int32)
    """
    hmax, kmax, lmax = cell.get_hkl_limits(dmin)
    hkl = np.meshgrid(
        np.linspace(-hmax, hmax + 1, 2 * hmax + 2, dtype=dtype),
        np.linspace(-kmax, kmax + 1, 2 * kmax + 2, dtype=dtype),
        np.linspace(-lmax, lmax + 1, 2 * lmax + 2, dtype=dtype),
    )
    hkl = np.stack(hkl).reshape((3, -1)).T

    # Remove reflection 0,0,0
    hkl = hkl[np.any(hkl != 0, axis=1)]

    # Remove reflections outside of resolution range
    dHKL = cell.calculate_d_array(hkl).astype("float32")
    hkl = hkl[dHKL >= dmin]

    return hkl


def asu2p1_tf(atom_pos_orth, unitcell, spacegroup,
              incell=True, fractional=True):
    '''
    Apply symmetry operations to real space asu model coordinates

    Parameters
    ----------
    atom_pos_orth: tensor, [N_atom, 3]
        ASU model ccordinates
    
    unitcell: gemmi.UnitCell

    spacegroup: gemmi.SpaceGroup

    incell: boolean, default True
        If True, will move all atoms inside of the unit cell

    fractional: boolean, default True
        If True, will return fractional coordinates; Otherwise will return orthogonal coordinates

    Return
    ------
    atom_pos_sym_oped, [N_atoms, N_ops, 3] tensor in either fractional or orthogonal coordinates
    '''
    orth2frac_tensor = tf.constant(unitcell.fractionalization_matrix.tolist())
    frac2orth_tensor = tf.constant(unitcell.orthogonalization_matrix.tolist())
    operations = spacegroup.operations()  # gemmi.GroupOps object
    R_G_tensor_stack = tf.stack([tf.constant(
        sym_op.rot, dtype=tf.float32)/sym_op.DEN for sym_op in operations], axis=0)
    T_G_tensor_stack = tf.stack([tf.constant(
        sym_op.tran, dtype=tf.float32)/sym_op.DEN for sym_op in operations], axis=0)

    atom_pos_frac = tf.tensordot(
        atom_pos_orth, tf.transpose(orth2frac_tensor), 1)
    sym_oped_pos_frac = tf.transpose(tf.tensordot(R_G_tensor_stack, tf.transpose(
        atom_pos_frac), 1), perm=[2, 0, 1]) + T_G_tensor_stack

    if incell:
        sym_oped_pos_frac = sym_oped_pos_frac - \
            tf.math.floor(sym_oped_pos_frac)

    if fractional:
        return sym_oped_pos_frac
    else:
        sym_oped_pos_orth = tf.tensordot(
            sym_oped_pos_frac, tf.transpose(frac2orth_tensor), 1)
        return sym_oped_pos_orth
