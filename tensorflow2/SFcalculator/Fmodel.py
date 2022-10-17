'''
Calculate Structural Factor from an atomic model: F_model = k_total * (F_calc + k_mask * F_mask)

Note:
1. We use direct summation for the F_calc
2. Now we only include f_0, no f' or f'', so no anomalous scattering

Written in Tensorflow 2.0

May, 2021
'''

__author__ = "Minhuan Li"
__email__ = "minhuanli@g.harvard.edu"

import gemmi
import numpy as np
import tensorflow as tf
import reciprocalspaceship as rs

from .symmetry import *
from .mask import reciprocal_grid, rsgrid2realmask, realmask2Fmask
from .utils import *
from .utils import vdw_rad_tensor, unitcell_grid_center
from .packingscore import packingscore_voxelgrid_tf


def F_protein(HKL_array, dr2_array, fullsf_tensor, reciprocal_cell_paras,
              R_G_tensor_stack,
              T_G_tensor_stack,
              atom_pos_frac,
              atom_b_iso,
              atom_b_aniso,
              atom_occ,
              NO_Bfactor=False):
    '''
    Calculate Protein Structural Factor from an atomic model

    atom_pos_frac: 2D tensor, [N_atom, N_dim=3]
    '''
    # F_calc = sum_Gsum_j{ [f0_sj*DWF*exp(2*pi*i*(h,k,l)*(R_G*(x1,x2,x3)+T_G))]} fractional postion, Rupp's Book P279
    # G is symmetry operations of the spacegroup and j is the atoms
    # DWF is the Debye-Waller Factor, has isotropic and anisotropic version, based on the PDB file input, Rupp's Book P641
    HKL_tensor = tf.constant(HKL_array, dtype=tf.float32)
    F_calc = 0

    if NO_Bfactor:
        magnitude = fullsf_tensor * atom_occ[..., None]  # [N_atom, N_HKLs]
    else:
        # DWF calculator
        dwf_iso = DWF_iso(atom_b_iso, dr2_array)
        dwf_aniso = DWF_aniso(atom_b_aniso, reciprocal_cell_paras, HKL_array)
        # Some atoms do not have Anisotropic U
        mask_vec = tf.reduce_all(
            atom_b_aniso == tf.convert_to_tensor([0.]*6), axis=-1)
        mask_indice = tf.reshape(tf.where(mask_vec), [-1])
        dwf_all = tf.tensor_scatter_nd_update(
            dwf_aniso, mask_indice[..., None], dwf_iso[mask_vec])  # [N_atoms, N_HKLs]

        # Apply Atomic Structure Factor and Occupancy for magnitude
        magnitude = dwf_all * fullsf_tensor * \
            atom_occ[..., None]  # [N_atoms, N_HKLs]

    # Vectorized phase calculation
    sym_oped_pos_frac = tf.transpose(tf.tensordot(R_G_tensor_stack, tf.transpose(
        atom_pos_frac), 1), perm=[2, 0, 1]) + T_G_tensor_stack  # Shape [N_atom, N_op, N_dim=3]
    cos_phase = 0.
    sin_phase = 0.
    # Loop through symmetry operations instead of fully vectorization, to reduce the memory cost
    for i in range(len(sym_oped_pos_frac[0])):
        phase_G = 2*tf.constant(np.pi)*tf.tensordot(HKL_tensor,
                                                    tf.transpose(sym_oped_pos_frac[:, i, :]), 1)
        cos_phase += tf.cos(phase_G)
        sin_phase += tf.sin(phase_G)  # Shape [N_HKLs, N_atoms]

    # Calcualte the complex structural factor
    magnitude_T = tf.transpose(magnitude)
    F_calc = tf.complex(tf.reduce_sum(cos_phase*magnitude_T, axis=-1),
                        tf.reduce_sum(sin_phase*magnitude_T, axis=-1))

    return F_calc


def F_protein_batch(HKL_array, dr2_array, fullsf_tensor, reciprocal_cell_paras,
                    R_G_tensor_stack,
                    T_G_tensor_stack,
                    atom_pos_frac_batch,
                    atom_b_iso,
                    atom_b_aniso,
                    atom_occ,
                    NO_Bfactor=False,
                    PARTITION=20):
    '''
    Calculate Protein Structural Factor from a batch of atomic models

    atom_pos_frac_batch: 3D tensor, [N_batch, N_atoms, N_dim=3]

    TODO: Support batched B factors
    '''
    # F_calc = sum_Gsum_j{ [f0_sj*DWF*exp(2*pi*i*(h,k,l)*(R_G*(x1,x2,x3)+T_G))]} fractional postion, Rupp's Book P279
    # G is symmetry operations of the spacegroup and j is the atoms
    # DWF is the Debye-Waller Factor, has isotropic and anisotropic version, based on the PDB file input, Rupp's Book P641

    HKL_tensor = tf.constant(HKL_array, dtype=tf.float32)
    batchsize = tf.shape(atom_pos_frac_batch)[0]

    if NO_Bfactor:
        magnitude = fullsf_tensor * atom_occ[..., None]  # [N_atom, N_HKLs]
    else:
        # DWF calculator
        dwf_iso = DWF_iso(atom_b_iso, dr2_array)
        dwf_aniso = DWF_aniso(atom_b_aniso, reciprocal_cell_paras, HKL_array)
        # Some atoms do not have Anisotropic U
        mask_vec = tf.reduce_all(
            atom_b_aniso == tf.convert_to_tensor([0.]*6), axis=-1)
        mask_indice = tf.reshape(tf.where(mask_vec), [-1])
        dwf_all = tf.tensor_scatter_nd_update(
            dwf_aniso, mask_indice[..., None], dwf_iso[mask_vec])  # [N_atoms, N_HKLs]

        # Apply Atomic Structure Factor and Occupancy for magnitude
        magnitude = dwf_all * fullsf_tensor * \
            atom_occ[..., None]  # [N_atoms, N_HKLs]

    # Vectorized phase calculation
    sym_oped_pos_frac = tf.tensordot(atom_pos_frac_batch, tf.transpose(R_G_tensor_stack, [
                                     2, 1, 0]), 1) + tf.transpose(T_G_tensor_stack)  # Shape [N_batch, N_atom, N_dim=3, N_ops]

    N_ops = tf.shape(R_G_tensor_stack)[0]
    N_partition = batchsize // PARTITION + 1
    F_calc = 0.
    for j in range(N_partition):
        Fcalc_j = 0.
        if j*PARTITION >= batchsize:
            continue
        start = j*PARTITION
        end = min((j+1)*PARTITION, batchsize)
        for i in range(N_ops):  # Loop through symmetry operations to reduce memory cost
            phase_ij = 2 * tf.constant(np.pi) * tf.tensordot(sym_oped_pos_frac[start:end, :, :, i], tf.transpose(
                HKL_tensor), 1)  # Shape [PARTITION, N_atoms, N_HKLs]
            Fcalc_ij = tf.complex(tf.reduce_sum(tf.cos(phase_ij)*magnitude, axis=1),
                                  tf.reduce_sum(tf.sin(phase_ij)*magnitude, axis=1))  # Shape [PARTITION, N_HKLs], sum over atoms
            # Shape [PARTITION, N_HKLs], sum over symmetry operations
            Fcalc_j += Fcalc_ij
        if j == 0:
            F_calc = Fcalc_j
        else:
            # Shape [N_batches, N_HKLs]
            F_calc = tf.concat([F_calc, Fcalc_j], 0)

    return F_calc

# TODO: currently you can't change the atom name, number and order given by the reference PDB


class SFcalculator(object):
    '''
    A class to formalize the structural factor calculation.
    '''

    def __init__(self, PDBfile_dir,
                 mtzfile_dir=None,
                 dmin=None,
                 set_experiment=True,
                 nansubset=['FP', 'SIGFP']):
        '''
        Initialize with necessary reusable information, like spacegroup, unit cell info, HKL_list, et.c.

        Parameters:
        -----------
        model_dir: path, str
            path to the PDB model file, will use its unit cell info, space group info, atom name info,
            atom position info, atoms B-factor info and atoms occupancy info to initialize the instance.

        mtz_file_dir: path, str, default None
            path to the mtz_file_dir, will use the HKL list in the mtz instead, override dmin with an inference

        dmin: float, default None
            highest resolution in the map in Angstrom, to generate Miller indices in recirpocal ASU

        set_experiment: Boolean, Default True
            Whether or not to set Fo and SigF, r_free, r_work from the experimental mtz file. It only works when
            the mtzfile_dir is not None

        nansubset: list of str, default ['FP', 'SIGFP']
            list of column names to examine the nan values
        '''
        structure = gemmi.read_pdb(PDBfile_dir)  # gemmi.Structure object
        self.unit_cell = structure.cell  # gemmi.UnitCell object
        self.space_group = gemmi.SpaceGroup(
            structure.spacegroup_hm)  # gemmi.SpaceGroup object
        self.operations = self.space_group.operations()  # gemmi.GroupOps object
        self.R_G_tensor_stack = tf.stack([tf.constant(
            sym_op.rot, dtype=tf.float32)/sym_op.DEN for sym_op in self.operations], axis=0)
        self.T_G_tensor_stack = tf.stack([tf.constant(
            sym_op.tran, dtype=tf.float32)/sym_op.DEN for sym_op in self.operations], axis=0)

        self.reciprocal_cell = self.unit_cell.reciprocal()  # gemmi.UnitCell object
        # [ar, br, cr, cos(alpha_r), cos(beta_r), cos(gamma_r)]
        self.reciprocal_cell_paras = [self.reciprocal_cell.a,
                                      self.reciprocal_cell.b,
                                      self.reciprocal_cell.c,
                                      np.cos(np.deg2rad(
                                          self.reciprocal_cell.alpha)),
                                      np.cos(np.deg2rad(
                                          self.reciprocal_cell.beta)),
                                      np.cos(np.deg2rad(
                                          self.reciprocal_cell.gamma))
                                      ]
        # Generate ASU HKL array and Corresponding d*^2 array
        if mtzfile_dir:
            mtz_reference = rs.read_mtz(mtzfile_dir)
            try:
                mtz_reference.dropna(axis=0, subset=nansubset, inplace=True)
            except:
                raise ValueError(
                    f"{nansubset} columns not included in the mtz file!")
            # HKL array from the reference mtz file, [N,3]
            self.HKL_array = mtz_reference.get_hkls()
            self.dHKL = self.unit_cell.calculate_d_array(self.HKL_array).astype("float32")
            self.dmin = self.dHKL.min()
            assert mtz_reference.cell == self.unit_cell, "Unit cell from mtz file does not match that in PDB file!"
            assert mtz_reference.spacegroup.hm == self.space_group.hm, "Space group from mtz file does not match that in PDB file!"
            self.Hasu_array = generate_reciprocal_asu(
                self.unit_cell, self.space_group, self.dmin)
            assert diff_array(self.HKL_array, self.Hasu_array) == set(
            ), "HKL_array should be equal or subset of the Hasu_array!"
            self.asu2HKL_index = tf.constant(
                asu2HKL(self.Hasu_array, self.HKL_array), dtype=tf.int32)
            # d*^2 array according to the HKL list, [N]
            self.dr2asu_array = self.unit_cell.calculate_1_d2_array(
                self.Hasu_array)
            self.dr2HKL_array = self.unit_cell.calculate_1_d2_array(
                self.HKL_array)

            if set_experiment:
                self.set_experiment(mtz_reference)
        else:
            if not dmin:
                raise ValueError(
                    "high_resolution dmin OR a reference mtz file should be provided!")
            else:
                self.dmin = dmin
                self.Hasu_array = generate_reciprocal_asu(
                    self.unit_cell, self.space_group, self.dmin)
                self.dHasu = self.unit_cell.calculate_d_array(self.Hasu_array).astype("float32")
                self.dr2asu_array = self.unit_cell.calculate_1_d2_array(
                    self.Hasu_array)
                self.HKL_array = None

        self.atom_name = []
        self.atom_pos_orth = []
        self.atom_pos_frac = []
        self.atom_b_aniso = []
        self.atom_b_iso = []
        self.atom_occ = []
        model = structure[0]  # gemmi.Model object
        for chain in model:
            for res in chain:
                for atom in res:
                    # A list of atom name like ['O','C','N','C', ...], [Nc]
                    self.atom_name.append(atom.element.name)
                    # A list of atom's Positions in orthogonal space, [Nc,3]
                    self.atom_pos_orth.append(atom.pos.tolist())
                    # A list of atom's Positions in fractional space, [Nc,3]
                    self.atom_pos_frac.append(
                        self.unit_cell.fractionalize(atom.pos).tolist())
                    # A list of anisotropic B Factor [[U11,U22,U33,U12,U13,U23],..], [Nc,6]
                    self.atom_b_aniso.append(atom.aniso.elements_pdb())
                    # A list of isotropic B Factor [B1,B2,...], [Nc]
                    self.atom_b_iso.append(atom.b_iso)
                    # A list of occupancy [P1,P2,....], [Nc]
                    self.atom_occ.append(atom.occ)

        self.atom_pos_orth = tf.convert_to_tensor(self.atom_pos_orth)
        self.atom_pos_frac = tf.convert_to_tensor(self.atom_pos_frac)
        self.atom_b_aniso = tf.convert_to_tensor(self.atom_b_aniso)
        self.atom_b_iso = tf.convert_to_tensor(self.atom_b_iso)
        self.atom_occ = tf.convert_to_tensor(self.atom_occ)
        self.n_atoms = len(self.atom_name)

        self.unique_atom = list(set(self.atom_name))
        self.orth2frac_tensor = tf.constant(
            self.unit_cell.fractionalization_matrix.tolist())
        self.frac2orth_tensor = tf.constant(
            self.unit_cell.orthogonalization_matrix.tolist())

        # A dictionary of atomic structural factor f0_sj of different atom types at different HKL Rupp's Book P280
        # f0_sj = [sum_{i=1}^4 {a_ij*exp(-b_ij* d*^2/4)} ] + c_j
        self.full_atomic_sf_asu = {}
        for atom_type in self.unique_atom:
            element = gemmi.Element(atom_type)
            self.full_atomic_sf_asu[atom_type] = tf.constant([
                element.it92.calculate_sf(dr2/4.) for dr2 in self.dr2asu_array], dtype=tf.float32)
        self.fullsf_tensor = tf.convert_to_tensor(
            [self.full_atomic_sf_asu[atom] for atom in self.atom_name])

        self.inspected = False

    def set_experiment(self, exp_mtz):
        '''
        Set experimental data in the refinement

        exp_mtz, rs.Dataset, mtzfile read by reciprocalspaceship
        '''
        try:
            self.Fo = tf.constant(exp_mtz["FP"].to_numpy())
            self.SigF = tf.constant(exp_mtz["SIGFP"].to_numpy())
        except:
            print("MTZ file doesn't contain 'FP' or 'SIGFP'! Check your data!")
        try:
            self.rfree_id = np.argwhere(
                exp_mtz["FreeR_flag"].values == 0).reshape(-1)
            self.rwork_id = np.argwhere(
                exp_mtz["FreeR_flag"].values != 0).reshape(-1)
        except:
            print("No Free Flag! Check your data!")

    def inspect_data(self):
        '''
        Do an inspection of data, for hints about 
        1. solvent percentage for mask calculation
        2. suitable grid size 
        '''
        # solvent percentage
        vdw_rad = vdw_rad_tensor(self.atom_name)
        uc_grid_orth_tensor = unitcell_grid_center(self.unit_cell,
                                                   spacing=4.5,
                                                   return_tensor=True)
        occupancy, _ = packingscore_voxelgrid_tf(
            self.atom_pos_orth, self.unit_cell, self.space_group, vdw_rad, uc_grid_orth_tensor)
        self.solventpct = np.round(100 - occupancy.numpy()*100, 0)

        # grid size
        mtz = gemmi.Mtz(with_base=True)
        mtz.cell = self.unit_cell
        mtz.spacegroup = self.space_group
        if not self.HKL_array is None:
            mtz.set_data(self.HKL_array)
        else:
            mtz.set_data(self.Hasu_array)
        self.gridsize = mtz.get_size_for_hkl(sample_rate=3.0)

        print("Solvent Percentage:", self.solventpct)
        print("Grid size:", self.gridsize)
        self.inspected = True

    def Calc_Fprotein(self, atoms_position_tensor=None,
                      atoms_biso_tensor=None,
                      atoms_baniso_tensor=None,
                      atoms_occ_tensor=None,
                      NO_Bfactor=False,
                      Print=False):
        '''
        Calculate the structural factor from a single atomic model, without solvent masking

        Parameters
        ----------
        atoms_positions_tensor: 2D [N_atoms, 3] tensor or default None
            Positions of atoms in the model, in unit of angstrom; If not given, the model stored in attribute `atom_pos_frac` will be used

        atoms_biso_tensor: 1D [N_atoms,] tensor or default None
            Isotropic B factors of each atoms in the model; If not given, the info stored in attribute `atoms_b_iso` will be used

        atoms_baniso_tensor: 2D [N_atoms, 6] tensor or default None
            Anisotropic B factors of each atoms in the model; If not given, the info stored in attribute `atoms_b_aniso` will be used

        atoms_occ_tensor: 1D [N_atoms,] tensor or default None
            Occupancy of each atoms in the model; If not given, the info stored in attribute `atom_occ` will be used

        NO_Bfactor: Boolean, default False
            If True, the calculation will not use Bfactor parameterization; Useful when we are parameterizing the ensemble with a true distribution

        Print: Boolean, default False
            If True, it will return the Fprotein as the function output; Or It will just be saved in the `Fprotein_asu` and `Fprotein_HKL` attributes

        Returns
        -------
        None (Print=False) or Fprotein (Print=True)
        '''
        # Read and tensorfy necessary inforamtion
        if not atoms_position_tensor is None:
            assert len(
                atoms_position_tensor) == self.n_atoms, "Atoms in atoms_positions_tensor should be consistent with atom names in PDB model!"
            # TODO Test the following line with non-orthogonal unit cell, check if we need a transpose at the transform matrix
            self.atom_pos_frac = tf.tensordot(
                atoms_position_tensor, tf.transpose(self.orth2frac_tensor), 1)

        if not atoms_baniso_tensor is None:
            assert len(atoms_baniso_tensor) == len(
                self.atom_name), "Atoms in atoms_baniso_tensor should be consistent with atom names in PDB model!"
            self.atom_b_aniso = atoms_baniso_tensor

        if not atoms_biso_tensor is None:
            assert len(atoms_biso_tensor) == len(
                self.atom_name), "Atoms in atoms_biso_tensor should be consistent with atom names in PDB model!"
            self.atom_b_iso = atoms_biso_tensor

        if not atoms_occ_tensor is None:
            assert len(atoms_occ_tensor) == len(
                self.atom_name), "Atoms in atoms_occ_tensor should be consistent with atom names in PDB model!"
            self.atom_occ = atoms_occ_tensor

        self.Fprotein_asu = F_protein(self.Hasu_array, self.dr2asu_array,
                                      self.fullsf_tensor,
                                      self.reciprocal_cell_paras,
                                      self.R_G_tensor_stack, self.T_G_tensor_stack,
                                      self.atom_pos_frac,
                                      self.atom_b_iso, self.atom_b_aniso, self.atom_occ,
                                      NO_Bfactor=NO_Bfactor)
        if not self.HKL_array is None:
            self.Fprotein_HKL = tf.gather(
                self.Fprotein_asu, self.asu2HKL_index)
            if Print:
                return self.Fprotein_HKL
        else:
            if Print:
                return self.Fprotein_asu

    def Calc_Fsolvent(self, solventpct=None, gridsize=None, dmin_mask=6.0, Print=False, dmin_nonzero=3.0):
        '''
        Calculate the structure factor of solvent mask in a differentiable way

        Parameters
        ----------
        solventpct: Int or Float, default None
            An approximate value of volume percentage of solvent in the unitcell. 
            run `inspect_data` before to use a suggested value

        gridsize: [Int, Int, Int], default None
            The size of grid to construct mask map.
            run `inspect_data` before to use a suggected value

        dmin_mask: np.float32, Default 6 angstroms.
            Minimum resolution cutoff, in angstroms, for creating the solvent mask

        Print: Boolean, default False
            If True, it will return the Fmask as the function output; Or It will just be saved in the `Fmask_asu` and `Fmask_HKL` attributes
        '''

        if solventpct is None:
            assert self.inspected, "Run inspect_data first or give a valid solvent percentage!"
            solventpct = self.solventpct

        if gridsize is None:
            assert self.inspected, "Run inspect_data first or give a valid grid size!"
            gridsize = self.gridsize

        # Shape [N_HKL_p1, 3], [N_HKL_p1,]
        Hp1_array, Fp1_tensor = expand_to_p1(
            self.space_group, self.Hasu_array, self.Fprotein_asu,
            dmin_mask=dmin_mask, unitcell=self.unit_cell)
        rs_grid = reciprocal_grid(Hp1_array, Fp1_tensor, gridsize)
        self.real_grid_mask = rsgrid2realmask(
            rs_grid, solvent_percent=solventpct)
        if not self.HKL_array is None:
            self.Fmask_HKL = realmask2Fmask(
                self.real_grid_mask, self.HKL_array)
            zero_hkl_inds = np.argwhere(self.dHKL <= dmin_nonzero)
            self.Fmask_HKL = tf.tensor_scatter_nd_update(self.Fmask_HKL, zero_hkl_inds, tf.zeros(len(zero_hkl_inds), dtype=tf.complex64))
            if Print:
                return self.Fmask_HKL
        else:
            self.Fmask_asu = realmask2Fmask(
                self.real_grid_mask, self.Hasu_array)
            zero_hkl_inds = np.argwhere(self.dHasu <= dmin_nonzero)
            self.Fmask_asu = tf.tensor_scatter_nd_update(self.Fmask_asu, zero_hkl_inds, tf.zeros(len(zero_hkl_inds), dtype=tf.complex64))
            if Print:
                return self.Fmask_asu

    def Calc_Ftotal(self, kall=None, kaniso=None, ksol=None, bsol=None):

        if kall is None:
            kall = tf.constant(1.0)
        if kaniso is None:
            kaniso = tf.random.normal([6])
        if ksol is None:
            ksol = tf.constant(0.35)
        if bsol is None:
            bsol = tf.constant(50.0)

        if not self.HKL_array is None:
            dr2_complex_tensor = tf.constant(
                self.dr2HKL_array, dtype=tf.complex64)
            scaled_Fmask = tf.complex(
                ksol, 0.0)*self.Fmask_HKL*tf.exp(-tf.complex(bsol, 0.0)*dr2_complex_tensor/4.)
            self.Ftotal_HKL = tf.complex(
                kall, 0.0)*tf.complex(DWF_aniso(kaniso[None, ...], self.reciprocal_cell_paras, self.HKL_array)[0], 0.0)*(self.Fprotein_HKL+scaled_Fmask)
            return self.Ftotal_HKL
        else:
            dr2_complex_tensor = tf.constant(
                self.dr2asu_array, dtype=tf.complex64)
            scaled_Fmask = tf.complex(
                ksol, 0.0)*self.Fmask_asu*tf.exp(-tf.complex(bsol, 0.0)*dr2_complex_tensor/4.)
            self.Ftotal_asu = tf.complex(
                kall, 0.0)*tf.complex(DWF_aniso(kaniso[None, ...], self.reciprocal_cell_paras, self.Hasu_array)[0], 0.0)*(self.Fprotein_asu+scaled_Fmask)
            return self.Ftotal_asu

    def Calc_Fprotein_batch(self, atoms_position_batch, NO_Bfactor=False, Print=False, PARTITION=20):
        '''
        Calculate the Fprotein with batched models. Most parameters are similar to `Calc_Fprotein`

        atoms_positions_batch: tf.float32 tensor, [N_batch, N_atoms, 3]

        PARTITION: Int, default 20
            To reduce the memory cost during the computation, we divide the batch into several partitions and loops through them.
            Larger PARTITION will require larger GPU memory. Default 20 will take around 4GB, if N_atoms~1600 and N_HKLs~13000.
            But larger PARTITION will give a smaller wall time, so this is a trade-off.
        '''
        # Read and tensor-fy necessary information
        # TODO Test the following line with non-orthogonal unit cell, check if we need a transpose at the transform matrix
        atom_pos_frac_batch = tf.tensordot(atoms_position_batch, tf.transpose(
            self.orth2frac_tensor), 1)  # [N_batch, N_atoms, N_dim=3]

        self.Fprotein_asu_batch = F_protein_batch(self.Hasu_array, self.dr2asu_array,
                                                  self.fullsf_tensor,
                                                  self.reciprocal_cell_paras,
                                                  self.R_G_tensor_stack, self.T_G_tensor_stack,
                                                  atom_pos_frac_batch,
                                                  self.atom_b_iso, self.atom_b_aniso, self.atom_occ,
                                                  NO_Bfactor=NO_Bfactor,
                                                  PARTITION=PARTITION)  # [N_batch, N_Hasus]

        if not self.HKL_array is None:
            self.Fprotein_HKL_batch = tf.gather(
                self.Fprotein_asu_batch, self.asu2HKL_index, axis=-1)
            if Print:
                return self.Fprotein_HKL_batch
        else:
            if Print:
                return self.Fprotein_asu_batch

    def Calc_Fsolvent_batch(self, solventpct=None, gridsize=None, dmin_mask=6, Print=False, PARTITION=100):
        '''
        Should run after Calc_Fprotein_batch, calculate the solvent mask structure factors in batched manner
        most parameters are similar to `Calc_Fmask`

        PARTITION: Int, default 100
            To reduce the memory cost during the computation, we divide the batch into several partitions and loops through them.
            Larger PARTITION will require larger GPU memory. Default 100 will take around 15GB, if gridsize=[160,160,160].
            But larger PARTITION will give a smaller wall time, so this is a trade-off. 
        '''

        if solventpct is None:
            assert self.inspected, "Run inspect_data first or give a valid solvent percentage!"
            solventpct = self.solventpct

        if gridsize is None:
            assert self.inspected, "Run inspect_data first or give a valid grid size!"
            gridsize = self.gridsize

        Hp1_array, Fp1_tensor_batch = expand_to_p1(
            self.space_group, self.Hasu_array, self.Fprotein_asu_batch,
            dmin_mask=dmin_mask, Batch=True, unitcell=self.unit_cell)

        batchsize = tf.shape(self.Fprotein_asu_batch)[0]
        N_partition = batchsize // PARTITION + 1
        Fmask_batch = 0.

        if not self.HKL_array is None:
            HKL_array = self.HKL_array
        else:
            HKL_array = self.Hasu_array

        for j in range(N_partition):
            if j*PARTITION >= batchsize:
                continue
            start = j*PARTITION
            end = min((j+1)*PARTITION, batchsize)
            # Shape [N_batch, *gridsize]
            rs_grid = reciprocal_grid(
                Hp1_array, Fp1_tensor_batch[start:end], gridsize, end-start)
            real_grid_mask = rsgrid2realmask(
                rs_grid, solvent_percent=solventpct, Batch=True)
            Fmask_batch_j = realmask2Fmask(
                real_grid_mask, HKL_array, end-start)
            if j == 0:
                Fmask_batch = Fmask_batch_j
            else:
                # Shape [N_batches, N_HKLs]
                Fmask_batch = tf.concat([Fmask_batch, Fmask_batch_j], 0)

        if not self.HKL_array is None:
            self.Fmask_HKL_batch = Fmask_batch
            # TODO force high resoltuion HKL rows to be zero
            if Print:
                return self.Fmask_HKL_batch
        else:
            self.Fmask_asu_batch = Fmask_batch
            if Print:
                return self.Fmask_asu_batch

    def Calc_Ftotal_batch(self, kall=None, kaniso=None, ksol=None, bsol=None):

        if kall is None:
            kall = tf.constant(1.0)
        if kaniso is None:
            kaniso = tf.random.normal([6])
        if ksol is None:
            ksol = tf.constant(0.35)
        if bsol is None:
            bsol = tf.constant(50.0)

        if not self.HKL_array is None:
            dr2_complex_tensor = tf.constant(
                self.dr2HKL_array, dtype=tf.complex64)
            scaled_Fmask = tf.complex(
                ksol, 0.0)*self.Fmask_HKL_batch*tf.exp(-tf.complex(bsol, 0.0)*dr2_complex_tensor/4.)
            self.Ftotal_HKL_batch = tf.complex(kall, 0.0)*tf.complex(DWF_aniso(
                kaniso[None, ...], self.reciprocal_cell_paras, self.HKL_array)[0], 0.0)*(self.Fprotein_HKL_batch+scaled_Fmask)
            return self.Ftotal_HKL_batch
        else:
            dr2_complex_tensor = tf.constant(
                self.dr2asu_array, dtype=tf.complex64)
            scaled_Fmask = tf.complex(
                ksol, 0.0)*self.Fmask_asu_batch*tf.exp(-tf.complex(bsol, 0.0)*dr2_complex_tensor/4.)
            self.Ftotal_asu_batch = tf.complex(kall, 0.0)*tf.complex(DWF_aniso(
                kaniso[None, ...], self.reciprocal_cell_paras, self.Hasu_array)[0], 0.0)*(self.Fprotein_asu_batch+scaled_Fmask)
            return self.Ftotal_asu_batch

    def prepare_DataSet(self, HKL_attr, F_attr):
        F_out = getattr(self, F_attr)
        HKL_out = getattr(self, HKL_attr)
        assert len(F_out) == len(
            HKL_out), "HKL and structural factor should have same length!"
        F_out_mag = tf.abs(F_out)
        PI_on_180 = 0.017453292519943295
        F_out_phase = tf.math.angle(F_out) / PI_on_180
        dataset = rs.DataSet(
            spacegroup=self.space_group, cell=self.unit_cell)
        dataset["H"] = HKL_out[:, 0]
        dataset["K"] = HKL_out[:, 1]
        dataset["L"] = HKL_out[:, 2]
        dataset["FMODEL"] = F_out_mag.numpy()
        dataset["PHIFMODEL"] = F_out_phase.numpy()
        dataset["FMODEL_COMPLEX"] = F_out.numpy()
        dataset.set_index(["H", "K", "L"], inplace=True)
        return dataset
