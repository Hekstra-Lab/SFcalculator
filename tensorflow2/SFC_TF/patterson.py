import tensorflow as tf
import numpy as np

def uvw_array_orth(dmin, dmax, step=0.3):
    '''
    create distance vector array within dmin to dmax range in orthogonal space
    usually choose dmin=3 and dmax=10, in unit of Angstrom
    '''
    u_list = np.arange(0.0, dmax+0.1, step)
    v_list = np.arange(0.0, dmax+0.1, step)
    w_list = np.arange(0.0, dmax+0.1, step)
    uvw_arr_orth = np.array(np.meshgrid(
        u_list, v_list, w_list)).T.reshape(-1, 3)
    d_array = np.sqrt(np.sum(uvw_arr_orth**2, axis=1))
    w = np.argwhere((d_array >= dmin) & (d_array <= dmax)).reshape(-1)
    return uvw_arr_orth[w]


def uvw_array_frac(unitcell, dmin, dmax, step=0.3):
    '''
    create distance vector array within dmin to dmax range in unitcell fractional space
    usually choose dmin=3 and dmax=10, in unit of Angstrom

    unitcell: Gemmi.UnitCell instance
    '''
    uvw_arr_orth = uvw_array_orth(dmin, dmax, step=step)
    o2f_matrix = np.array(unitcell.fractionalization_matrix.tolist())
    return np.dot(uvw_arr_orth, o2f_matrix.T)


def P_uvw_np(uvw, Fh2, HKL_array, volume):
    '''
    Compute the Patterson map for a given distance list uvw with numpy
    This is a purely vectorized function so will be memory-demanding
    so is designed for a small partition of full uvw list
    Typical length can be 10, but depends on how large your memory is.

    volume: float, volume of real space unitcell, Gemmi.UnitCell.volume attribute
    '''
    return np.sum(Fh2*np.cos(2*np.pi*np.sum(HKL_array[None, ...]*uvw[:, None, :], axis=-1)), axis=-1)*2/volume


def P_uvw_tf(uvw, Fh2, HKL_array, volume):
    '''
    Compute the Patterson map for a given distance list uvw with tensorflow
    This is a purely vectorized function so will be memory-demanding
    so is designed for a small partition of full uvw list
    Typical length can be 10000, but can be much larger if you have a larger GPU
    60000 HKL with 10000 uvw will take ~0.9GB GPU memory 

    Feed the function with tensors instead of arrays for a better performance
    '''
    return tf.reduce_sum(Fh2*tf.cos(2*np.pi*tf.reduce_sum(HKL_array[None, ...]*uvw[:, None, :], axis=-1)), axis=-1)*2/volume

def P_uvw_tf_batch(uvw, Fh2_batch, HKL_array, volume):
    '''
    Similar to `P_uvw_tf` but now the Fh2_batch is a batch of models
    '''
    return tf.reduce_sum(Fh2_batch[:,None,:]*tf.cos(2*np.pi*tf.reduce_sum(HKL_array[None, ...]*uvw[:, None, :], 
                                                          axis=-1)), axis=-1)*2/volume


def Patterson_tf(uvw_arr, Fh, HKL_array, volume, sharpen=False, remove_origin=False, PARTITION=10000):
    '''
    Differentiablly calculate the patterson map with tensorflow
    
    Parameters
    ----------
    uvw_arr: array-like, with shape [N_uvw, 3]
        The distance vector array in fractional coordinates
    
    Fh: tf.tensor, with shape [N_hkl,]
        Structure factor amplitude

    HKL_array: array-like, with shape [N_hkl, 3]
        miller index array

    volume: float 
        real space unitcell volume, Gemmi.UnitCell.volume attribute

    sharpen: bool, default False
        Whether normalize the SF magnitude to sharpen the patterson space.
        If True, do Eh = Fh/<Fh^2>^(1/2), Fh2_normed = (Eh^3 * Fh)^(1/2)
        Check Rupp's textbook P468 for details.

    remove_origin: bool, default False
        Whether remove the origin of the patterson map by subtract the Fh2.
        If True, do Fh2 = Fh^2 - <Fh^2>
        Check Rupp's textbook P468 for details.

    PARTITION: int, default 10000
        Size of partition for each vecotrized operation. 
        Large partition size gives faster performance but more demanding to your memory. 
        60000 HKL with 10000 parition requires about 0.9GB memory on GPU.
    '''
    N_uvw = len(uvw_arr)
    N_partition = N_uvw // PARTITION + 1
    uvw_tensor = tf.constant(uvw_arr, dtype=tf.float32)
    HKL_tensor = tf.constant(HKL_array, dtype=tf.float32)
    Pu = 0.

    if sharpen:
        Eh = Fh / tf.sqrt(tf.reduce_mean(Fh**2))
        Fh2 = tf.sqrt(Eh**3 * Fh)
    else:
        Fh2 = Fh**2

    if remove_origin:
        Fh2 = Fh2 - tf.reduce_mean(Fh2)

    for j in range(N_partition):
        Pu_j = 0.
        if j*PARTITION >= N_uvw:
            continue
        start = j*PARTITION
        end = min((j+1)*PARTITION, N_uvw)
        Pu_j = P_uvw_tf(uvw_tensor[start:end], Fh2, HKL_tensor, volume)
        
        if j == 0:
            Pu = Pu_j
        else:
            Pu = tf.concat([Pu, Pu_j],0)
    return Pu

def Patterson_tf_batch(uvw_arr, Fh_batch, HKL_array, volume, sharpen=False, remove_origin=False, PARTITION_uvw=10000, PARTITION_batch=20):
    '''
    Similar to `Patterson_tf` but for a batch of Fh

    Fh_batch: tf.tensor, [N_batch, N_hkl]
    
    PARTITION_uvw: int, default 10000
        Size of partition for each vecotrized operation. 
        Large partition size gives faster performance but more demanding to your memory. 
        60000 HKL with 10000 parition requires about 0.9GB memory on GPU.

    PARTITION_batch: int, default 20
        Size of partition on the batched model.
        60000 HKL with 10000 partition and 20 models requires about 17GB memory on GPU

    '''
    N_uvw = len(uvw_arr)
    N_partition_uvw = N_uvw // PARTITION_uvw + 1
    
    N_batch = tf.shape(Fh_batch)[0]
    N_partition_batch = N_batch // PARTITION_batch + 1

    uvw_tensor = tf.constant(uvw_arr, dtype=tf.float32)
    HKL_tensor = tf.constant(HKL_array, dtype=tf.float32)
    Pu = 0.

    if sharpen:
        Eh_batch = Fh_batch / tf.sqrt(tf.reduce_mean(Fh_batch**2, axis=-1, keepdims=True))
        Fh2_batch = tf.sqrt(Eh_batch**3 * Fh_batch)
    else:
        Fh2_batch = Fh_batch**2

    if remove_origin:
        Fh2_batch = Fh2_batch - tf.reduce_mean(Fh2_batch, axis=-1, keepdims=True)

    for i in range(N_partition_batch):
        Pu_i = 0.
        
        if i*PARTITION_batch >= N_batch:
            continue
        start_i = i*PARTITION_batch
        end_i = min((i+1)*PARTITION_batch, N_batch)
        Fh2_i = Fh2_batch[start_i:end_i]

        for j in range(N_partition_uvw):
            Pu_ij = 0.
            if j*PARTITION_uvw >= N_uvw:
                continue
            start_j = j*PARTITION_uvw
            end_j = min((j+1)*PARTITION_uvw, N_uvw)
            Pu_ij = P_uvw_tf_batch(uvw_tensor[start_j:end_j], Fh2_i, HKL_tensor, volume)
            
            if j == 0:
                Pu_i = Pu_ij
            else:
                Pu_i = tf.concat([Pu_i, Pu_ij],1)
        
        if i == 0:
                Pu = Pu_i
        else:
            Pu = tf.concat([Pu, Pu_i],0) 

    return Pu

