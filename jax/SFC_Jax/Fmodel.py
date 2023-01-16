'''
Calculate Structural Factor from an atomic model: F_model = k_total * (F_calc + k_mask * F_mask)

Note:
1. We use direct summation for the F_calc
2. Now we only include f_0, no f' or f'', so no anomalous scattering

Written in Jax
'''

__author__ = "Minhuan Li"
__email__ = "minhuanli@g.harvard.edu"

import gemmi
import numpy as np
import jax
from jax import numpy as jnp
import reciprocalspaceship as rs

from .symmetry import generate_reciprocal_asu, expand_to_p1, get_p1_idx
from .mask import reciprocal_grid, rsgrid2realmask, realmask2Fmask
from .utils import DWF_aniso, DWF_iso, diff_array, asu2HKL
from .utils import vdw_rad_tensor, unitcell_grid_center
from .packingscore import packingscore_voxelgrid_jax


class SFcalculator(object):
    '''
    A class to formalize the structural factor calculation
    '''

    def __init__(self, PDBfile_dir,
                 mtzfile_dir=None,
                 dmin=None,
                 set_experiment=True,
                 nansubset=['FP', 'SIGFP'],
                 freeflag='FreeR_flag',
                 testset_value=0):
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

        self.R_G_tensor_stack = jnp.array(np.array([
            np.array(sym_op.rot)/sym_op.DEN for sym_op in self.operations])).astype(jnp.float32)
        self.T_G_tensor_stack = jnp.array(np.array([
            np.array(sym_op.tran)/sym_op.DEN for sym_op in self.operations])).astype(jnp.float32)

        self.reciprocal_cell = self.unit_cell.reciprocal()  # gemmi.UnitCell object
        # [ar, br, cr, cos(alpha_r), cos(beta_r), cos(gamma_r)]
        self.reciprocal_cell_paras = jnp.array([self.reciprocal_cell.a,
                                                self.reciprocal_cell.b,
                                                self.reciprocal_cell.c,
                                                np.cos(np.deg2rad(
                                                    self.reciprocal_cell.alpha)),
                                                np.cos(np.deg2rad(
                                                    self.reciprocal_cell.beta)),
                                                np.cos(np.deg2rad(
                                                    self.reciprocal_cell.gamma))
                                                ]).astype(jnp.float32)
        if mtzfile_dir:
            mtz_reference = rs.read_mtz(mtzfile_dir)
            try:
                mtz_reference.dropna(axis=0, subset=nansubset, inplace=True)
            except:
                raise ValueError(
                    f"{nansubset} columns not included in the mtz file!")
            # HKL array from the reference mtz file, [N,3]
            self.HKL_array = mtz_reference.get_hkls()
            self.dHKL = self.unit_cell.calculate_d_array(
                self.HKL_array).astype("float32")
            self.dmin = self.dHKL.min()
            assert mtz_reference.cell == self.unit_cell, "Unit cell from mtz file does not match that in PDB file!"
            assert mtz_reference.spacegroup.hm == self.space_group.hm, "Space group from mtz file does not match that in PDB file!"  # type: ignore
            self.Hasu_array = generate_reciprocal_asu(
                self.unit_cell, self.space_group, self.dmin)
            assert diff_array(self.HKL_array, self.Hasu_array) == set(
            ), "HKL_array should be equal or subset of the Hasu_array!"
            # TODO: See if need to change to tensor
            self.asu2HKL_index = asu2HKL(self.Hasu_array, self.HKL_array)
            # d*^2 array according to the HKL list, [N]
            self.dr2asu_array = self.unit_cell.calculate_1_d2_array(
                self.Hasu_array)
            self.dr2HKL_array = self.unit_cell.calculate_1_d2_array(
                self.HKL_array)
            if set_experiment:
                self.set_experiment(mtz_reference, freeflag, testset_value)
        else:
            if not dmin:
                raise ValueError(
                    "high_resolution dmin OR a reference mtz file should be provided!")
            else:
                self.dmin = dmin
                self.Hasu_array = generate_reciprocal_asu(
                    self.unit_cell, self.space_group, self.dmin)
                self.dHasu = self.unit_cell.calculate_d_array(
                    self.Hasu_array).astype("float32")
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

        self.atom_pos_orth = jnp.array(self.atom_pos_orth).astype(jnp.float32)
        self.atom_pos_frac = jnp.array(self.atom_pos_frac).astype(jnp.float32)
        self.atom_b_aniso = jnp.array(self.atom_b_aniso).astype(jnp.float32)
        self.atom_b_iso = jnp.array(self.atom_b_iso).astype(jnp.float32)
        self.atom_occ = jnp.array(self.atom_occ).astype(jnp.float32)
        self.n_atoms = len(self.atom_name)
        self.unique_atom = list(set(self.atom_name))

        self.orth2frac_tensor = jnp.array(
            self.unit_cell.fractionalization_matrix.tolist()).astype(jnp.float32)
        self.frac2orth_tensor = jnp.array(
            self.unit_cell.orthogonalization_matrix.tolist()).astype(jnp.float32)

        # A dictionary of atomic structural factor f0_sj of different atom types at different HKL Rupp's Book P280
        # f0_sj = [sum_{i=1}^4 {a_ij*exp(-b_ij* d*^2/4)} ] + c_j
        self.full_atomic_sf_asu = {}
        for atom_type in self.unique_atom:
            element = gemmi.Element(atom_type)
            self.full_atomic_sf_asu[atom_type] = np.array([
                element.it92.calculate_sf(dr2/4.) for dr2 in self.dr2asu_array])
        self.fullsf_tensor = jnp.array(np.array([
            self.full_atomic_sf_asu[atom] for atom in self.atom_name])).astype(jnp.float32)
        self.inspected = False

    def set_experiment(self, exp_mtz, freeflag='FreeR_flag', testset_value=0):
        '''
        Set experimental data in the refinement

        exp_mtz, rs.Dataset, mtzfile read by reciprocalspaceship
        '''
        try:
            self.Fo = jnp.array(exp_mtz["FP"].to_numpy()).astype(jnp.float32)
            self.SigF = jnp.array(
                exp_mtz["SIGFP"].to_numpy()).astype(jnp.float32)
        except:
            print("MTZ file doesn't contain 'FP' or 'SIGFP'! Check your data!")
        try:
            self.rfree_id = np.argwhere(
                exp_mtz[freeflag].values == testset_value).reshape(-1)
            self.rwork_id = np.argwhere(
                exp_mtz[freeflag].values != testset_value).reshape(-1)
        except:
            print("No Free Flag! Check your data!")

    def inspect_data(self, dmin_mask=6.0):
        '''
        Do an inspection of data, for hints about 
        1. solvent percentage for mask calculation
        2. suitable grid size 
        3. pre-calculate index used in expand to p1
        '''
        # solvent percentage
        vdw_rad = vdw_rad_tensor(self.atom_name)
        uc_grid_orth_tensor = unitcell_grid_center(self.unit_cell,
                                                   spacing=4.5,
                                                   return_tensor=True)
        occupancy, _ = packingscore_voxelgrid_jax(
            self.atom_pos_orth, self.unit_cell, self.space_group, vdw_rad, uc_grid_orth_tensor)
        self.solventpct = 1 - occupancy

        # grid size
        mtz = gemmi.Mtz(with_base=True)
        mtz.cell = self.unit_cell
        mtz.spacegroup = self.space_group
        if not self.HKL_array is None:
            mtz.set_data(self.HKL_array)
        else:
            mtz.set_data(self.Hasu_array)
        self.gridsize = mtz.get_size_for_hkl(sample_rate=3.0)

        # index used in expand to p1
        self.dmin_mask = dmin_mask
        self.Hp1_array_filtered, self.idx_1, self.idx_2 = get_p1_idx(
            self.space_group, self.Hasu_array, self.dmin_mask, self.unit_cell)

        print(f"Solvent Percentage: {self.solventpct:.3f}")
        print("Grid size:", self.gridsize)
        print("Filtered P1 HKL length: ", len(self.Hp1_array_filtered))
        self.inspected = True

    def Calc_Fprotein(self, atoms_position_tensor=None,
                      atoms_biso_tensor=None,
                      atoms_baniso_tensor=None,
                      atoms_occ_tensor=None,
                      NO_Bfactor=False,
                      Return=False):
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

        Return: Boolean, default False
            If True, it will return the Fprotein as the function output; Or It will just be saved in the `Fprotein_asu` and `Fprotein_HKL` attributes

        Returns
        -------
        None (Return=False) or Fprotein (Return=True)
        '''
        # Read and tensor-fy necessary inforamtion
        if not atoms_position_tensor is None:
            assert len(
                atoms_position_tensor) == self.n_atoms, "Atoms in atoms_positions_tensor should be consistent with atom names in PDB model!"
            self.atom_pos_frac = jnp.tensordot(
                atoms_position_tensor, self.orth2frac_tensor.T, 1)

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
            self.Fprotein_HKL = self.Fprotein_asu[self.asu2HKL_index]
            if Return:
                return self.Fprotein_HKL
        else:
            if Return:
                return self.Fprotein_asu

    def Calc_Fsolvent(self, solventpct=None, gridsize=None, Return=False, dmin_nonzero=3.0):
        '''
        Calculate the structure factor of solvent mask in a differentiable way

        Parameters
        ----------
        solventpct: 0 - 1 Float, default None
            An approximate value of volume percentage of solvent in the unitcell. 
            run `inspect_data` before to use a suggested value

        gridsize: [Int, Int, Int], default None
            The size of grid to construct mask map.
            run `inspect_data` before to use a suggected value

        dmin_mask: np.float32, Default 6 angstroms.
            Minimum resolution cutoff, in angstroms, for creating the solvent mask

        Return: Boolean, default False
            If True, it will return the Fmask as the function output; Or It will just be saved in the `Fmask_asu` and `Fmask_HKL` attributes
        '''

        if solventpct is None:
            assert self.inspected, "Run inspect_data first or give a valid solvent percentage!"
            solventpct = self.solventpct

        if gridsize is None:
            assert self.inspected, "Run inspect_data first or give a valid grid size!"
            gridsize = self.gridsize

        # Shape [N_HKL_p1, 3], [N_HKL_p1,]
        Fp1_tensor = expand_to_p1(
            self.space_group, self.Hasu_array, self.Fprotein_asu, 
            self.idx_1, self.idx_2,
            dmin_mask=self.dmin_mask, unitcell=self.unit_cell)
        rs_grid = reciprocal_grid(self.Hp1_array_filtered, Fp1_tensor, gridsize)
        self.real_grid_mask = rsgrid2realmask(
            rs_grid, solvent_percent=solventpct)  # type: ignore
        if not self.HKL_array is None:
            Fmask_HKL = realmask2Fmask(
                self.real_grid_mask, self.HKL_array)
            zero_hkl_bool = jnp.array(self.dHKL <= dmin_nonzero)
            self.Fmask_HKL = jnp.where(zero_hkl_bool, jnp.array(
                0., dtype=jnp.complex64), Fmask_HKL)
            if Return:
                return self.Fmask_HKL
        else:
            Fmask_asu = realmask2Fmask(
                self.real_grid_mask, self.Hasu_array)
            zero_hkl_bool = jnp.array(self.dHasu <= dmin_nonzero)
            self.Fmask_asu = jnp.where(zero_hkl_bool, jnp.array(
                0., dtype=jnp.complex64), Fmask_asu)
            if Return:
                return self.Fmask_asu

    def Calc_Ftotal(self, kall=None, kaniso=None, ksol=None, bsol=None, key=jax.random.PRNGKey(42)):
        if kall is None:
            kall = jnp.array(1.0)
        if kaniso is None:
            kaniso = jax.random.normal(key, shape=[6])
        if ksol is None:
            ksol = jnp.array(0.35)
        if bsol is None:
            bsol = jnp.array(50.0)

        if not self.HKL_array is None:
            dr2_tensor = jnp.array(self.dr2HKL_array)
            scaled_Fmask = ksol * self.Fmask_HKL * \
                jnp.exp(-bsol * dr2_tensor/4.0)
            self.Ftotal_HKL = kall * DWF_aniso(kaniso[None, ...], self.reciprocal_cell_paras, self.HKL_array)[
                0] * (self.Fprotein_HKL+scaled_Fmask)
            return self.Ftotal_HKL
        else:
            dr2_tensor = jnp.array(self.dr2asu_array)
            scaled_Fmask = ksol * self.Fmask_asu * \
                jnp.exp(-bsol * dr2_tensor/4.0)
            self.Ftotal_asu = kall * DWF_aniso(kaniso[None, ...], self.reciprocal_cell_paras, self.Hasu_array)[
                0] * (self.Fprotein_asu+scaled_Fmask)
            return self.Ftotal_asu

    def Calc_Fprotein_batch(self, atoms_position_batch, NO_Bfactor=False, Return=False, PARTITION=20):
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
        atom_pos_frac_batch = jnp.tensordot(
            atoms_position_batch, self.orth2frac_tensor.T, 1)  # [N_batch, N_atoms, N_dim=3]

        self.Fprotein_asu_batch = F_protein_batch(self.Hasu_array, self.dr2asu_array,
                                                  self.fullsf_tensor,
                                                  self.reciprocal_cell_paras,
                                                  self.R_G_tensor_stack, self.T_G_tensor_stack,
                                                  atom_pos_frac_batch,
                                                  self.atom_b_iso, self.atom_b_aniso, self.atom_occ,
                                                  NO_Bfactor=NO_Bfactor,
                                                  PARTITION=PARTITION)  # [N_batch, N_Hasus]

        if not self.HKL_array is None:
            # type: ignore
            self.Fprotein_HKL_batch = self.Fprotein_asu_batch[:,
                                                              self.asu2HKL_index]
            if Return:
                return self.Fprotein_HKL_batch
        else:
            if Return:
                return self.Fprotein_asu_batch

    def Calc_Fsolvent_batch(self, solventpct=None, gridsize=None, dmin_mask=6, Return=False, PARTITION=100, dmin_nonzero=3.0):
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

        Fp1_tensor_batch = expand_to_p1(
            self.space_group, self.Hasu_array, self.Fprotein_asu_batch, self.idx_1, self.idx_2,
            dmin_mask=dmin_mask, Batch=True, unitcell=self.unit_cell)

        batchsize = self.Fprotein_asu_batch.shape[0]  # type: ignore
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
                self.Hp1_array_filtered, Fp1_tensor_batch[start:end], gridsize, end-start)
            real_grid_mask = rsgrid2realmask(
                rs_grid, solvent_percent=solventpct, Batch=True)  # type: ignore
            Fmask_batch_j = realmask2Fmask(
                real_grid_mask, HKL_array, end-start)
            if j == 0:
                Fmask_batch = Fmask_batch_j
            else:
                # Shape [N_batches, N_HKLs]
                Fmask_batch = jnp.concatenate(
                    (Fmask_batch, Fmask_batch_j), dim=0)  # type: ignore
        zero_hkl_bool = jnp.array(self.dHKL <= dmin_nonzero)
        Fmask_batch = jnp.where(zero_hkl_bool, jnp.array(
            0., dtype=jnp.complex64), Fmask_batch)
        if not self.HKL_array is None:
            self.Fmask_HKL_batch = Fmask_batch
            if Return:
                return self.Fmask_HKL_batch
        else:
            self.Fmask_asu_batch = Fmask_batch
            if Return:
                return self.Fmask_asu_batch

    def Calc_Ftotal_batch(self, kall=None, kaniso=None, ksol=None, bsol=None, key=jax.random.PRNGKey(42)):

        if kall is None:
            kall = jnp.array(1.0)
        if kaniso is None:
            kaniso = jax.random.normal(key, shape=[6])
        if ksol is None:
            ksol = jnp.array(0.35)
        if bsol is None:
            bsol = jnp.array(50.0)

        if not self.HKL_array is None:
            dr2_tensor = jnp.array(self.dr2HKL_array)
            scaled_Fmask = ksol * self.Fmask_HKL_batch * \
                jnp.exp(-bsol * dr2_tensor/4.0)
            self.Ftotal_HKL_batch = kall * DWF_aniso(kaniso[None, ...], self.reciprocal_cell_paras, self.HKL_array)[
                0] * (self.Fprotein_HKL_batch+scaled_Fmask)
            return self.Ftotal_HKL_batch
        else:
            dr2_tensor = jnp.array(self.dr2asu_array)
            scaled_Fmask = ksol * self.Fmask_asu_batch * \
                jnp.exp(-bsol * dr2_tensor/4.0)
            self.Ftotal_asu_batch = kall * DWF_aniso(kaniso[None, ...], self.reciprocal_cell_paras, self.Hasu_array)[
                0] * (self.Fprotein_asu_batch+scaled_Fmask)
            return self.Ftotal_asu_batch

    def prepare_DataSet(self, HKL_attr, F_attr):
        F_out = getattr(self, F_attr)
        HKL_out = getattr(self, HKL_attr)
        assert len(F_out) == len(
            HKL_out), "HKL and structural factor should have same length!"
        F_out_mag = jnp.abs(F_out)
        PI_on_180 = 0.017453292519943295
        F_out_phase = jnp.angle(F_out) / PI_on_180
        dataset = rs.DataSet(spacegroup=self.space_group,
                             cell=self.unit_cell)  # type: ignore
        dataset["H"] = HKL_out[:, 0]
        dataset["K"] = HKL_out[:, 1]
        dataset["L"] = HKL_out[:, 2]
        dataset["FMODEL"] = np.array(F_out_mag)
        dataset["PHIFMODEL"] = np.array(F_out_phase)
        dataset["FMODEL_COMPLEX"] = np.array(F_out)
        dataset.set_index(["H", "K", "L"], inplace=True)
        return dataset


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
    HKL_tensor = jnp.array(HKL_array, dtype=jnp.float32)

    if NO_Bfactor:
        magnitude = fullsf_tensor * atom_occ[..., None]  # [N_atom, N_HKLs]
    else:
        # DWF calculator
        dwf_iso = DWF_iso(atom_b_iso, dr2_array)
        dwf_aniso = DWF_aniso(atom_b_aniso, reciprocal_cell_paras, HKL_array)
        # Some atoms do not have Anisotropic U
        mask_vec = jnp.all(atom_b_aniso == jnp.array([0.]*6), axis=-1)
        dwf_all = jnp.where(mask_vec[:, None], dwf_iso, dwf_aniso)

        # Apply Atomic Structure Factor and Occupancy for magnitude
        magnitude = dwf_all * fullsf_tensor * \
            atom_occ[..., None]  # [N_atoms, N_HKLs]

    # Vectorized phase calculation
    sym_oped_pos_frac = jnp.transpose(jnp.tensordot(R_G_tensor_stack,
                                                    atom_pos_frac.T, 1), [2, 0, 1]) + T_G_tensor_stack  # Shape [N_atom, N_op, N_dim=3]
    cos_phase = 0.
    sin_phase = 0.
    # Loop through symmetry operations instead of fully vectorization, to reduce the memory cost
    for i in range(sym_oped_pos_frac.shape[1]):
        phase_G = 2*np.pi * \
            jnp.tensordot(HKL_tensor, sym_oped_pos_frac[:, i, :].T, 1)
        cos_phase += jnp.cos(phase_G)
        sin_phase += jnp.sin(phase_G)  # Shape [N_HKLs, N_atoms]
    # Calcualte the complex structural factor
    F_calc = jax.lax.complex(jnp.sum(cos_phase*magnitude.T, axis=-1),
                             jnp.sum(sin_phase*magnitude.T, axis=-1))
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
    HKL_tensor = jnp.array(HKL_array).astype(jnp.float32)
    batchsize = atom_pos_frac_batch.shape[0]

    if NO_Bfactor:
        magnitude = fullsf_tensor * atom_occ[..., None]  # [N_atom, N_HKLs]
    else:
        # DWF calculator
        dwf_iso = DWF_iso(atom_b_iso, dr2_array)
        dwf_aniso = DWF_aniso(atom_b_aniso, reciprocal_cell_paras, HKL_array)
        # Some atoms do not have Anisotropic U
        mask_vec = jnp.all(atom_b_aniso == jnp.array([0.]*6), axis=-1)
        dwf_all = jnp.where(mask_vec[:, None], dwf_iso, dwf_aniso)
        # Apply Atomic Structure Factor and Occupancy for magnitude
        magnitude = dwf_all * fullsf_tensor * \
            atom_occ[..., None]  # [N_atoms, N_HKLs]

    # Vectorized phase calculation
    sym_oped_pos_frac = jnp.tensordot(atom_pos_frac_batch, jnp.transpose(R_G_tensor_stack, [
        2, 1, 0]), 1) + T_G_tensor_stack.T  # Shape [N_batch, N_atom, N_dim=3, N_ops]
    N_ops = R_G_tensor_stack.shape[0]
    N_partition = batchsize // PARTITION + 1
    F_calc = 0.
    for j in range(N_partition):
        Fcalc_j = 0.
        if j*PARTITION >= batchsize:
            continue
        start = j*PARTITION
        end = min((j+1)*PARTITION, batchsize)
        for i in range(N_ops):  # Loop through symmetry operations to reduce memory cost
            # Shape [PARTITION, N_atoms, N_HKLs]
            phase_ij = 2 * jnp.pi * \
                jnp.tensordot(
                    sym_oped_pos_frac[start:end, :, :, i], HKL_tensor.T, 1)
            Fcalc_ij = jax.lax.complex(jnp.sum(jnp.cos(phase_ij)*magnitude, axis=1),
                                       jnp.sum(jnp.sin(phase_ij)*magnitude, axis=1))  # Shape [PARTITION, N_HKLs], sum over atoms
            # Shape [PARTITION, N_HKLs], sum over symmetry operations
            Fcalc_j += Fcalc_ij
        if j == 0:
            F_calc = Fcalc_j
        else:
            # Shape [N_batches, N_HKLs]
            F_calc = jnp.concatenate((F_calc, Fcalc_j), dim=0)  # type: ignore
    return F_calc
