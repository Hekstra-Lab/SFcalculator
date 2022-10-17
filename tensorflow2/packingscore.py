import tensorflow as tf
from .voxel import voxelvalue_tf_p1

def packingscore_clashatom_tf(atom_pos_orth, vdw_min_dist, sfcalculator):
    '''
    Calculate the number of clash atoms as packing score

    Parameters
    ----------
    atom_pos_orth: tf.tensor, [N_atom, 3]
        asu model in cartesian coordinates, better remove all hydrogen atoms at the beginning

    vdw_min_dist: tf.tensor, [N_atom, N_atom]
        min intermolecular atom distances defined by vdw radius.
        use vdw_distance_matrix_tf to calculate this

    sfcalculator: dr.xtal.SFcalculator
        An instance with necessary attributes containing the spacegroup-related information

    Returns
    -------
    Number of clash atoms between all symmetrical pairs. The higher, the worse packing score
    '''
    if sfcalculator.space_group.hm == 'P 1':
        print("No packing score for P1 symmetry")
        return None
    atom_pos_frac = tf.tensordot(
        atom_pos_orth, tf.transpose(sfcalculator.orth2frac_tensor), 1)
    sym_oped_pos_frac = tf.transpose(tf.tensordot(sfcalculator.R_G_tensor_stack, tf.transpose(
        atom_pos_frac), 1), perm=[2, 0, 1]) + sfcalculator.T_G_tensor_stack
    sym_oped_pos_orth = tf.tensordot(sym_oped_pos_frac,
                                     tf.transpose(sfcalculator.frac2orth_tensor), 1)

    clash_count = 0
    num_ops = len(sym_oped_pos_orth[0])
    for i in range(num_ops-1):
        for j in range(i+1, num_ops):
            dist_matrixij = tf.sqrt(tf.reduce_sum(tf.square(sym_oped_pos_orth[:, i, :][None, ...] -
                                                            sym_oped_pos_orth[:, j, :][:, None, :]),
                                                  axis=-1))
            clash_count += tf.math.count_nonzero(dist_matrixij < vdw_min_dist)
    return clash_count


def packingscore_voxelgrid_tf(atom_pos_orth, unit_cell, space_group, vdw_rad, unitcell_grid_center_orth_tensor, CUTOFF=0.0001):
    '''
    Calculate the grid occupancy and clash percentage as packing score

    Parameters
    ----------
    atom_pos_orth: tf.tensor, [N_atom, 3]
        asu model in cartesian coordinates, better remove all hydrogen atoms at the beginning

    unit_cell: gemmi.UnitCell

    space_group: gemmi.SpaceGroup

    vdw_rad: tf.tensor, [N_atom, ]
        vdw radius of atoms, use vdw_rad_tensor to calculate

    unitcell_grid_center_orth_tensor: tf.float32 tensor, [N_grid, 3]
        center positions of all grids in carteisian space, use unitcell_grid_center to calculate

    CUTOFF: float, default 0.0001, must < 0.5
        cutoff to convert into binary map; Larger cutoff means slower decay further

    Returns
    -------
    Percentage of the occupancy of all unitcell grids, and percentage
    of clash grids between all symmetrical pairs.
    '''
    N_grid = len(unitcell_grid_center_orth_tensor[:, 0])
    spacing = tf.reduce_max(
        unitcell_grid_center_orth_tensor[1] - unitcell_grid_center_orth_tensor[0])
    # s ~ log(1/c -1) / (d/2 - r)
    spacing = tf.maximum(spacing, 3.5)
    steepness = tf.math.log(1.0/CUTOFF - 1.0)/(spacing/2.0 - 1.5)

    voxel_map_p1 = voxelvalue_tf_p1(unitcell_grid_center_orth_tensor, atom_pos_orth,
                                    unit_cell, space_group, vdw_rad,
                                    s=steepness, binary=True, cutoff=CUTOFF)

    occupancy = tf.math.count_nonzero(
        voxel_map_p1, dtype=tf.float32) / N_grid
    clash = tf.math.count_nonzero(
        voxel_map_p1 > 1, dtype=tf.float32) / N_grid

    return occupancy, clash
