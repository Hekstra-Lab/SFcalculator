import torch
from .utils import try_gpu

def reciprocal_grid(Hp1_array, Fp1_tensor, gridsize, batchsize=None):
    '''
    Construct a reciprocal grid in reciprocal unit cell with HKL array and Structural Factor tensor
    Fully differentiable with torch

    Parameters
    ----------
    Hp1_array: np.int32 array
        The HKL list in the p1 unit cell. Usually the output of expand_to_p1

    Fp1_tensor: torch.complex64 tensor
        Corresponding structural factor tensor. Usually the output of expand_to_p1

    gridsize: array-like, int
        The size of the grid you want to create, the only requirement is "big enough"

    Return:
    Reciprocal space unit cell grid, as a torch.complex64 tensor
    '''
    grid = torch.zeros(gridsize, device=try_gpu(), dtype=torch.complex64)
    tuple_index = tuple(torch.tensor(Hp1_array.T, device=try_gpu(), dtype=int)) #type: ignore
    if batchsize is not None:
        for i in range(batchsize):
            Fp1_tensor_i = Fp1_tensor[i]
            grid_i = grid.clone()  
            grid_i[tuple_index] = Fp1_tensor_i # Reciprocal Grid of model i
            if i == 0:
                grid_batch = grid_i[None, ...]
            else:
                grid_batch = torch.concat((grid_batch, grid_i[None, ...]), dim=0) #type: ignore
        return grid_batch #type: ignore
    else:
        grid[tuple_index] = Fp1_tensor
        return grid


def rsgrid2realmask(rs_grid, solvent_percent=50.0, scale=50, Batch=False):
    '''
    Convert reciprocal space grid to real space solvent mask grid, in a
    fully differentiable way with tensorflow.

    Parameters:
    -----------
    rs_grid: tf.complex64 tensor
        Reciprocal space unit cell grid. Usually the output of reciprocal_grid

    solvent_percent: float
        The approximate volume percentage of solvent in the system, to generate the cutoff

    scale: int/float
        The scale used in sigmoid function, to make the distribution binary

    Return:
    -------
    tf.float32 tensor
    The solvent mask grid in real space, solvent voxels have value close to 1, while protein voxels have value close to 0.
    '''
    real_grid = tf.math.real(tf.signal.fft3d(rs_grid))
    real_grid_norm = (real_grid - tf.reduce_mean(real_grid)) / \
        tf.math.reduce_std(real_grid)
    if Batch:
        CUTOFF = tfp.stats.percentile(real_grid_norm[0], solvent_percent)
    else:
        CUTOFF = tfp.stats.percentile(
            real_grid_norm, solvent_percent)  # This is a little slow

    real_grid_mask = tf.sigmoid((CUTOFF-real_grid_norm)*scale)
    return real_grid_mask


def realmask2Fmask(real_grid_mask, H_array, batchsize=None):
    '''
    Convert real space solvent mask grid to mask structural factor, in a fully differentiable
    manner, with tensorflow.

    Parameters:
    -----------
    real_grid_mask: tf.float32 tensor
        The solvent massk grid in real space unit cell. Usually the output of rsgrid2realmask

    H_array: array-like, int
        The HKL list we are interested in to assign structural factors

    Return:
    -------
    tf.complex64 tensor
    Solvent mask structural factor corresponding to the HKL list in H_array
    '''
    Fmask_grid = tf.math.conj(tf.signal.fft3d(tf.complex(real_grid_mask, 0.)))
    if batchsize is not None:
        positive_index = to_pos_idx(H_array, Fmask_grid[0])
        Fmask = tf.gather_nd(Fmask_grid, tf.repeat(
            positive_index[None, ...], batchsize, axis=0), batch_dims=1)
    else:
        positive_index = to_pos_idx(H_array, Fmask_grid)
        Fmask = tf.gather_nd(Fmask_grid, positive_index)

    return Fmask