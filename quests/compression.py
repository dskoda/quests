from ase.io import read
from ase import Atoms
from numba import types
from numba.typed import Dict, List
import numpy as np 
from quests.entropy import delta_entropy, perfect_entropy, diversity
from quests.descriptor import get_descriptors
from bayes_opt import BayesianOptimization

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
        
def minimum_set_coverage(frames: list, initial_entropies: np.ndarray, descriptor_dict: dict, h: float, l: float):
    
    """Given the frames and initial entropies, determine the most diverse set of atoms in the set
    
    Arguments: 
        frames (list): descriptors of each of the frames
        initial_entropies (np.ndarray): array with initial entropies of each of the frames
        descriptor_dict (dict): dictionary containing descriptors
        h (float): h value
        l (float): lambda value 
        
    Returns: indexes (list): list of indexes of the most diverse frames in order 
        
        
    """
    
    indexes = []
    
    compressed_data = frames[initial_entropies.argmax()]
    indexes.append(initial_entropies.argmax())
    frames.pop(initial_entropies.argmax())
    
    # loop to find order of values 
    
    for i in range(len(frames)):
        entropy = np.zeros(len(frames))
        for a in range(len(frames)):
            entropy[a] = np.mean(delta_entropy(frames[a], compressed_data, h = h)) + l*initial_entropies[find_key(descriptor_dict, frames[a])]
        compressed_data = np.concatenate((compressed_data, frames[entropy.argmax()]), axis = 0)
        indexes.append(find_key(descriptor_dict, frames[entropy.argmax()]))
        frames.pop(entropy.argmax())
    
    return indexes
    
    
    
def farthest_point_sampling(frames, initial_entropies, descriptor_dict):
    
    """Given the frames and initial entropies, determine the most diverse set of atoms in the set
    
    Arguments: 
        frames (list): descriptors of each of the frames
        initial_entropies (np.ndarray): array with initial entropies of each of the frames
        descriptor_dict (dict): dictionary containing descriptors
        
    Returns: indexes (list): list of indexes of the most diverse frames in order 
        
        
    """
    
    indexes = []
    
    # generate first value 
    
    data = frames[initial_entropies.argmax()]
    indexes.append(initial_entropies.argmax())
    frames.pop(initial_entropies.argmax())
    
    # loop to find order of values 
    
    for i in range(int(len(frames))):
        
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
             compression_value: float = None, c_type: str = 'msc', l: float = None):
    
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
    
    assert compression_value > 0 and compression_value <= 1, "Compression value must be between 0 (non-inclusive) and 1"
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
    