import numpy as np
import time
import matplotlib.pyplot as plt
import math
import torch
import pdb
import random

def get_mask_pseudo_diagonal_numpy(mask_shape, sparsity, file_name=None, experimentType="randomWithZero", 
                                   layerNum=None, numDiag=None, diag_pos=None, currLayer=1, debug=0):
    """Creates a pseudo-diagonal mask with the specified sparsity.
    Args:
        mask_shape: list, used to obtain shape of the random mask.
        sparsity: float, between 0 and 1.
    Returns:
        numpy.ndarray
    """
    # Create an array of zeros with the specified shape
    mask = np.zeros(mask_shape)
    diag_length = max(mask_shape[0], mask_shape[1])
    
    # Ensure reproducibility
    np.random.seed(int(time.time()))
    
    start_row = int(diag_pos)
    
    # Vectorized operation to create the pseudo-diagonal mask
    rows = (np.arange(diag_length) + start_row) % mask_shape[0]
    cols = np.arange(diag_length) % mask_shape[1]
    
    mask[rows, cols] = 1

    return mask

def get_mask_pseudo_diagonal_torch(mask_shape, sparsity, diag_pos, experimentType="random", device='cuda'):
    num_rows, num_cols = mask_shape
    #print(mask_shape)
    if num_rows >= num_cols:
        # Case when there are more rows than columns
        diag_length = num_cols
        start_row = int(diag_pos)
        rows = (torch.arange(diag_length, device=device) + start_row) % num_rows
        cols = torch.arange(diag_length, device=device) % num_cols
    else:
        # Case when there are more columns than rows
        diag_length = num_rows
        start_col = int(diag_pos)
        rows = torch.arange(diag_length, device=device) % num_rows
        cols = (torch.arange(diag_length, device=device) + start_col) % num_cols

    # Create a sparse tensor using the computed rows and cols
    indices = torch.stack([rows, cols], dim=0)
    values = torch.ones(diag_length, device=device)

    # Create the sparse COO tensor
    sparse_mask = torch.sparse_coo_tensor(indices, values, size=mask_shape, device=device)

    return sparse_mask  


""" def get_mask_pseudo_diagonal_torch(mask_shape, sparsity, diag_pos, experimentType="random", device='cuda'):
    
    # Create an array of zeros with the specified shape
    mask = torch.zeros(mask_shape, device=device)
    num_rows, num_cols = mask_shape

    if num_rows >= num_cols:
        # Case when there are more rows than columns
        diag_length = num_cols
        start_row = int(diag_pos)
        rows = (torch.arange(diag_length, device=device) + start_row) % num_rows
        cols = torch.arange(diag_length, device=device) % num_cols
    else:
        # Case when there are more columns than rows
        diag_length = num_rows
        start_col = int(diag_pos)
        rows = torch.arange(diag_length, device=device) % num_rows
        cols = (torch.arange(diag_length, device=device) + start_col) % num_cols

    mask[rows, cols] = 1

    return mask """

""" def get_mask_pseudo_diagonal_torch(mask_shape, sparsity, diag_pos, experimentType="random", device='cuda'):
    num_rows, num_cols = mask_shape
    
    # Determine the diagonal length and positions
    diag_length = min(num_rows, num_cols)
    
    if num_rows >= num_cols:
        start_row = int(diag_pos)
        rows = (torch.arange(diag_length, device=device) + start_row) % num_rows
        cols = torch.arange(diag_length, device=device)
    else:
        start_col = int(diag_pos)
        rows = torch.arange(diag_length, device=device)
        cols = (torch.arange(diag_length, device=device) + start_col) % num_cols

    # Initialize the mask with zeros
    mask = torch.zeros(mask_shape, device=device)
    
    # Set diagonal positions directly
    mask[rows, cols] = 1

    return mask """