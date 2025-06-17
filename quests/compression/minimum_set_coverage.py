import numpy as np
from quests.entropy import delta_entropy

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
        if target.shape != input_dict[key].shape:
            continue
        if (target == input_dict[key]).all():
            return key
    return None


def minimum_set_coverage(
    frames: list,
    initial_entropies: np.ndarray,
    h: float,
    entropy_weight: float,
    value: int = None,
):
    """Given the frames and initial entropies, determine the most diverse set of atoms in the set

    Arguments:
        frames (list): descriptors of each of the frames
        initial_entropies (np.ndarray): array with initial entropies of each of the frames
        descriptor_dict (dict): dictionary containing descriptors
        h (float): h value
        entropy_weight (float): weight that considers the "novelty" of a new sample based on
            the values of dH and the entropy of the sample itself. Higher weights favor samples
            with higher initial entropy.

    Returns: indexes (list): list of indexes of the most diverse frames in order


    """

    indexes = []

    # dictionary with index & descriptors
    descriptor_dict = {}
    for i in range(len(frames)):
        descriptor_dict[i] = frames[i]

    compressed_data = frames[initial_entropies.argmax()]
    indexes.append(initial_entropies.argmax())
    frames.pop(initial_entropies.argmax())

    # loop to find order of values

    num = value if value != None else len(frames)
    num = len(frames) if len(frames) <= num else num

    for i in range(num):
        entropy = np.zeros(len(frames))
        for a in range(len(frames)):
            entropy[a] = (
                np.mean(delta_entropy(frames[a], compressed_data, h=h))
                + entropy_weight
                * initial_entropies[find_key(descriptor_dict, frames[a])]
            )
        compressed_data = np.concatenate(
            (compressed_data, frames[entropy.argmax()]), axis=0
        )
        indexes.append(find_key(descriptor_dict, frames[entropy.argmax()]))
        frames.pop(entropy.argmax())

    return indexes
