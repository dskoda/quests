from ase.io import read
from ase import Atoms
from numba.typed import List
from numba import types
from bayes_opt import BayesianOptimization
from quests.entropy import perfect_entropy, diversity
from quests.descriptor import get_descriptors
import numpy as np

from .minimum_set_coverage import minimum_set_coverage
from .farthest_point_sampling import farthest_point_sampling

DEFAULT_CUTOFF: float = 5.0
DEFAULT_K: int = 32
EPS: float = 1e-15
DEFAULT_H: float = 0.015
DEFAULT_BS: int = 10000


def get_frame_descriptors(dset: List[Atoms], k: int = DEFAULT_K, cutoff: int = DEFAULT_CUTOFF, h: int = DEFAULT_H, batch_size: int = DEFAULT_BS):
    
    """Gets descriptors for each frame
    
    Arguments: 
        dset (List[Atoms]): dataset for which entropies will be computed
        k (int): number of nearest neighbors to use when computing descriptors.
        h (int): 
        batch_size (int): 
        cutoff (float): cutoff radius for the weight function.
        
    Returns: 
        frames_orig (list): list of the descriptors of each frame (np.ndarray)
        
    """
    
    frames_orig = []
    initial_entropies = []
    for frame in dset:
        y = get_descriptors([frame], k = k, cutoff = cutoff)
        frames_orig.append(y)
        initial_entropies.append(perfect_entropy(y, h=h, batch_size = batch_size))
    return frames_orig, np.array(initial_entropies)

def compress_dataset(dset: List[Atoms], k: int = DEFAULT_K, cutoff: float = DEFAULT_CUTOFF, h: float = DEFAULT_H, batch_size: int = DEFAULT_BS,
             compression_value: float = None, c_type: str = 'msc', l: float = None):
    
    """Gets descriptors for each frame
    
    Arguments: 
        dset (List[Atoms]): dataset for which entropies will be computed
        k (int): number of nearest neighbors to use when computing descriptors.
        cutoff (float)
        h (float): h value 
        batch_size (int):  
        cutoff (float): cutoff radius for the weight function.
        
    Returns: 
        frames_orig (list): list of the descriptors of each frame (np.ndarray)
        
    """
    
    assert c_type in ['msc', 'msc2', 'fps']
    
    # descriptors and initial entropies for each frame in the dataset
    frames, initial_entropies = get_frame_descriptors(dset, k, cutoff, h, batch_size)
    
    # dictionary with index & descriptors 
    descriptor_dict = {}
    for i in range(len(frames)):
        descriptor_dict[i] = frames[i]
        
        
    if c_type == 'fps':
        
        # retrieve indexes 
        
        indexes = farthest_point_sampling(frames, initial_entropies, descriptor_dict)
        if compression_value != None:
            return dset[indexes[:int(len(dset)*compression_value)]]
        else:
            
            # finding optimal compression value 
            
            def optimization_function(x):
                final_data = [frames[i] for i in indexes[:int(len(dset)*x)]]
                final_data = np.concatenate(final_data, axis = 0)
                entropy_msc = perfect_entropy(final_data, h = h, batch_size = batch_size)
                diversity_msc = diversity(final_data, h=h, batch_size = batch_size)
                return entropy_msc*np.log(diversity_msc)
            
            bounds = {'x': (0.1, 1)}
            optimizer = BayesianOptimization(f=optimization_function, pbounds=bounds, random_state=1)
            optimizer.maximize(init_points=5, n_iter=20)
            
            return dset[indexes[:int(len(dset)*optimizer.max['params']['x'])]]
    elif c_type == 'msc':
        
        # retrieve indexes
        
        indexes = minimum_set_coverage(frames, initial_entropies, descriptor_dict, h, l = 0)
        
        if compression_value != None:
            return dset[indexes[:int(len(dset)*compression_value)]]
        else:
            # finding optimal compression value 
            
            def optimization_function(x):
                final_data = [frames[i] for i in indexes[:int(len(dset)*x)]]
                final_data = np.concatenate(final_data, axis = 0)
                entropy_msc = perfect_entropy(final_data, h = h, batch_size = batch_size)
                diversity_msc = diversity(final_data, h=h, batch_size = batch_size)
                return entropy_msc*np.log(diversity_msc)
            
            bounds = {'x': (0.1, 1)}
            optimizer = BayesianOptimization(f=optimization_function, pbounds=bounds, random_state=1)
            optimizer.maximize(init_points=5, n_iter=20)
            
            return dset[indexes[:int(len(dset)*optimizer.max['params']['x'])]]
    else:
        if compression_value != None and l != None:
            return dset[minimum_set_coverage(frames, initial_entropies, descriptor_dict, h, l)[:int(len(dset)*compression_value)]]
        else:
            # finding optimal compression value
            
            def optimization_function(x, l):
                indexes = minimum_set_coverage(frames, initial_entropies, descriptor_dict, h, l)
                final_data = [frames[i] for i in indexes[:int(len(dset)*x)]]
                final_data = np.concatenate(final_data, axis = 0)
                entropy_msc = perfect_entropy(final_data, h = h, batch_size = batch_size)
                diversity_msc = diversity(final_data, h=h, batch_size = batch_size)
                return entropy_msc*np.log(diversity_msc)
            
            bounds = {'x': (0.1, 1), 'l': (0, 10)}
            optimizer = BayesianOptimization(f=optimization_function, pbounds=bounds, random_state=1)
            optimizer.maximize(init_points=5, n_iter=20)
            
            return dset[minimum_set_coverage(frames, initial_entropies, descriptor_dict, h, optimizer.max['params']['l'])
                        [:int(len(dset)*optimizer.max['params']['x'])]]
    