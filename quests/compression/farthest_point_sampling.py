import numpy as np 

DEFAULT_CUTOFF: float = 5.0
DEFAULT_K: int = 32
EPS: float = 1e-15
DEFAULT_H: float = 0.015
DEFAULT_BS: int = 10000


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
        
