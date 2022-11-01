import numpy as np
import jax
from jax import numpy as jnp

from .symmetry import asu2p1_jax

def voxelvalue_jax_asu(unitcell_grid_center_orth, atom_pos_orth,
                      frac2orth_tensor, orth2frac_tensor,
                      vdw_rad_tensor,
                      s=10., binary=True, cutoff=0.1):
    '''
    Differentiably render atom coordinates into real space grid map value
    Va(d, ra) = 1 / (1 + exp(s*(d-ra)))

    Without applying symmetry operations to the model

    Reference:
    Orlando, Gabriele, et al. "PyUUL provides an interface between biological 
    structures and deep learning algorithms." Nature communications 13.1 (2022)

    Parameters
    ----------
    unitcell_grid_center_orth: tensor, [N_grid, 3]
        Cartesian coordinates of real space unitcell grid center

    atom_pos_orth: tensor, [N_atom, 3]
        cartesian coordinates of model atoms you want to render, in an asu

    frac2orth_tensor: tensor, [3,3]
        fractionalize matrix

    vdw_rad_tensor: tensor, [N_atom, ]
        van de waals radius of atoms in the model accordingly

    s: float, default 10.0
        steepness parameter in the sigmoid function

    binary: binary, default True
        whether convert the map value to 0/1

    cutoff: float, default 0.1
        cutoff value to convert to binary. 1 if > cutoff, 0 otherwise.

    Returns
    -------
    voxel_value, tensor [N_grid,]
    '''
    atom_pos_frac = jnp.tensordot(atom_pos_orth, orth2frac_tensor.T, 1)
    atom_pos_frac_incell = atom_pos_frac - jnp.floor(atom_pos_frac)
    atom_pos_orth_incell = jnp.tensordot(atom_pos_frac_incell, frac2orth_tensor.T, 1)
    voxel2atom_dist = jnp.sqrt(jnp.sum(jnp.square(unitcell_grid_center_orth[:, None, :] - atom_pos_orth_incell[None, ...]),
                                            axis=-1))
    sigmoid_value = 1./(1.+jnp.exp(s*(voxel2atom_dist - vdw_rad_tensor)))
    voxel_value = jnp.sum(sigmoid_value, axis=-1)
    if binary:
        return jnp.where(voxel_value > cutoff, 1.0, 0.0)
    else:
        return voxel_value


def voxelvalue_jax_p1(unitcell_grid_center_orth, atom_pos_orth, unit_cell, space_group, vdw_rad_tensor,
                        s=10., binary=True, cutoff=0.1):
    '''
    Differentiably render atom coordinates into real space grid map value
    Va(d, ra) = 1 / (1 + exp(s*(d-ra)))

    Apply all symmetry operations to an asu model then render the whole unit_cell 

    Reference:
    Orlando, Gabriele, et al. "PyUUL provides an interface between biological 
    structures and deep learning algorithms." Nature communications 13.1 (2022)

    Parameters
    ----------
    unitcell_grid_center_orth: tensor, [N_grid, 3]
        Cartesian coordinates of real space unitcell grid center

    atom_pos_orth: tensor, [N_atom, N_ops, 3]
        cartesian coordinates of model atoms you want to render, in a single asu model
        Will apply all symmtery operations onto it and give a whole p1 unitcell

    unit_cell: gemmi.UnitCell

    space_group: gemmi.SpaceGroup

    vdw_rad_tensor: tensor, [N_atom, ]
        van de waals radius of atoms in the model accordingly

    s: float, default 10.0
        steepness parameter in the sigmoid function

    binary: binary, default True
        whether convert the map value to 0/1

    cutoff: float, default 0.1
        cutoff value to convert to binary. 1 if > cutoff, 0 otherwise.

    Returns
    -------
    voxel_value, tensor [N_grid,]
    '''
    sym_oped_atom_pos_orth_incell = asu2p1_jax(atom_pos_orth,
                                              unit_cell, space_group,
                                              incell=True, fractional=False)
    voxel2atom_dist = jnp.sqrt(jnp.sum(jnp.square(unitcell_grid_center_orth[:, None, None, :] - sym_oped_atom_pos_orth_incell[None, ...]),
                                            axis=-1))
    sigmoid_value = 1. / \
        (1.+jnp.exp(s*(voxel2atom_dist - vdw_rad_tensor[:, None])))

    voxel_value = jnp.sum(sigmoid_value, axis=1)
    if binary:
        return jnp.sum(jnp.where(voxel_value > cutoff, 1.0, 0.0), axis=-1)
    else:
        return jnp.sum(voxel_value, axis=-1)


def voxelvalue_jax_p1_savememory(unitcell_grid_center_orth, atom_pos_orth, unit_cell, space_group, vdw_rad_tensor,
                                s=10., binary=True, cutoff=0.1):
    '''
    Differentiably render atom coordinates into real space grid map value
    Va(d, ra) = 1 / (1 + exp(s*(d-ra)))

    Apply all symmetry operations to an asu model then render the whole unit_cell

    Save-memory version, do some loops instead of fully vectorization
    Useful when N_grid is super large 

    Reference:
    Orlando, Gabriele, et al. "PyUUL provides an interface between biological 
    structures and deep learning algorithms." Nature communications 13.1 (2022)

    Parameters
    ----------
    unitcell_grid_center_orth: tensor, [N_grid, 3]
        Cartesian coordinates of real space unitcell grid center

    atom_pos_orth: tensor, [N_atom, N_ops, 3]
        cartesian coordinates of model atoms you want to render, in a single asu model
        Will apply all symmtery operations onto it and give a whole p1 unitcell

    unit_cell: gemmi.UnitCell

    space_group: gemmi.SpaceGroup

    vdw_rad_tensor: tensor, [N_atom, ]
        van de waals radius of atoms in the model accordingly

    s: float, default 10.0
        steepness parameter in the sigmoid function

    binary: binary, default True
        whether convert the map value to 0/1

    cutoff: float, default 0.1
        cutoff value to convert to binary. 1 if > cutoff, 0 otherwise.

    Returns
    -------
    1D voxel_value, tensor [N_grid,]
    '''
    sym_oped_atom_pos_orth_incell = sym_oped_atom_pos_orth_incell = asu2p1_jax(atom_pos_orth,
                                                                                 unit_cell, space_group,
                                                                                 incell=True, fractional=False)
    N_ops = len(sym_oped_atom_pos_orth_incell[0])
    voxel_map = jnp.array(0.)
    for i in range(N_ops):
        model_i = sym_oped_atom_pos_orth_incell[:, i, :]
        voxel2atom_dist = jnp.sqrt(jnp.sum(jnp.square(unitcell_grid_center_orth[:, None, :] - model_i[None, ...]),
                                                axis=-1))
        sigmoid_value = 1./(1.+jnp.exp(s*(voxel2atom_dist - vdw_rad_tensor)))
        voxel_value = jnp.sum(sigmoid_value, axis=-1)
        if binary:
            map_i = jnp.where(voxel_value > cutoff, 1.0, 0.0)
        else:
            map_i = voxel_value
        voxel_map += map_i

    return voxel_map


def voxel_1dto3d_np(voxel_value_1d, na, nb, nc):
    '''
    Convert the 1D voxel value array into 3D array and keep the order

    Parameters
    ----------
    voxel_value_1d: array, [N_grid,]
        1D voxel value array

    na, nb, nc: int
        number of 
    '''
    temp_3d = voxel_value_1d.reshape(nc, na, nb)
    voxel_value_3d = np.transpose(temp_3d, [1, 2, 0])
    return voxel_value_3d


def voxel_1dto3d_jnp(voxel_value_1d, na, nb, nc):
    temp_3d = jnp.reshape(voxel_value_1d, [nc, na, nb])
    voxel_value_3d = jnp.transpose(temp_3d, axes=[1, 2, 0])
    return voxel_value_3d
