import numpy as np

def cosine_distance(x: np.ndarray, y: np.ndarray) -> np.array:
    """
    Aim: calculates the cosine distance between a single sample x and multiple samples y
    Parameters:
        x: np.ndarray -> single sample
        y: np.ndarray -> multiple samples
    Returns:
        an array containing the distances between X and the various samples in Y.
    """
    # compute the dot product between x and each sample in y using np.dot(y, x)
    dot_product = np.dot(y, x)

    # compute the magnitude (euclidean norm) of x and of each sample in y
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y, axis=1)

    # As cosine distance = 1 - cosine similarity, first the cosine similarity is calculated and then the cosine distance
    cosine_similarity = dot_product / (norm_x * norm_y)
    cosine_distance = 1 - cosine_similarity

    return cosine_distance

if __name__ == '__main__':
    x = np.array([1, 2, 3])
    y = np.array([[1, 2, 3], [4, 5, 6]])
    our_distance = cosine_distance(x, y)
    from sklearn.metrics.pairwise import cosine_distances
    scipy_distance = cosine_distances(x, y)
    assert np.allclose(our_distance, scipy_distance)
    print(our_distance, scipy_distance)