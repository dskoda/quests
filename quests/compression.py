from ase.io import read
from ase import Atoms
from numba import types
from numba.typed import Dict, List
import numpy as np 
from quests.entropy import delta_entropy, perfect_entropy
from quests.descriptor import get_descriptors

DEFAULT_CUTOFF: float = 5.0
DEFAULT_K: int = 32
EPS: float = 1e-15
DEFAULT_H: int = 0.015
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

def find_key(input_dict: dict, target: np.ndarray):
    
    """Given a dictionary of descriptors, determines the index of the target descriptors
    
    Arguments: 
        input_dict (dictionary): dictionary containing descriptors
        target (np.ndarray): numpy array of descriptor
        
    Returns: key (int): original index 
        
        
    """
    
    for key in input_dict:
        if (target.shape != input_dict[key].shape):
            continue
        if (target == input_dict[key]).all():
            return key
    return None
        
def minimum_set_coverage(frames, initial_entropies, descriptor_dict, compression_value, h):
    
    indexes = []
    
    compressed_data = frames[initial_entropies.argmax()]
    indexes.append(initial_entropies.argmax())
    frames.pop(initial_entropies.argmax())
    
    # loop to find order of values 
    
    for i in range(len(frames)):
        entropy = np.zeros(len(frames))
        for a in range(len(frames)):
            entropy[a] = np.mean(delta_entropy(frames[a], compressed_data, h = h))
        compressed_data = np.concatenate((compressed_data, frames[entropy.argmax()]), axis = 0)
        indexes.append(find_key(descriptor_dict, frames[entropy.argmax()]))
        frames.pop(entropy.argmax())
    
    return indexes
    
    
    
def farthest_point_sampling(frames, initial_entropies, descriptor_dict, compression_value):
    
    indexes = []
    
    # generate first value 
    
    data = frames[initial_entropies.argmax()]
    indexes.append(initial_entropies.argmax())
    frames.pop(initial_entropies.argmax())
    
    # loop to find order of values 
    
    for i in range(int(len(frames)*compression_value)):
        
        # generate minimum distance matrix 
        
        min_distance = np.zeros(len(frames))
        
        # calculates distance between closest values in sets 
        
        for c in range(len(frames)):
            distance_matrix = np.zeros((len(frames), len(data)))
            for a in range(len(frames)):
                for b in range(len(data)):
                    distance_matrix[a, b] = np.linalg.norm(data[b] - frames[a])
            min_distance[c] = np.min(distance_matrix)
            
        # appends farthest set
        
        indexes.append(find_key(descriptor_dict, frames[min_distance.argmax()]))
        data = np.concatenate((data, frames[min_distance.argmax()]), axis = 0)
        frames.pop(min_distance.argmax())
    
    return indexes
        
    
    
def compress(dset: List[Atoms], k: int = DEFAULT_K, cutoff: int = DEFAULT_CUTOFF, h: int = DEFAULT_H, batch_size: int = DEFAULT_BS,
             compression_value: float = None, c_type: str = 'msc'):
    
    
    
    assert compression_value > 0 and compression_value <= 1, "Compression value must be between 0 (non-inclusive) and 1"
    assert c_type in ['msc', 'msc2', 'fps']
    
    # descriptors and initial entropies for each frame in the dataset
    frames, initial_entropies = get_frame_descriptors(dset, k, cutoff, h, batch_size)
    
    # dictionary with index & descriptors 
    descriptor_dict = {}
    for i in range(len(frames)):
        descriptor_dict[i] = frames[i]
        
    if c_type == 'fps':
        return dset[farthest_point_sampling(frames, initial_entropies, descriptor_dict, compression_value)]
    elif c_type == 'msc':
        return dset[minimum_set_coverage(frames, initial_entropies, descriptor_dict, compression_value, h)]
    