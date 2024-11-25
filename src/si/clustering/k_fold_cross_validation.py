import numpy as np

def k_fold_cross_validation(model, X, y, k=5):
    """
    Perform k-fold cross-validation on a given model.

    Parameters:
    model: The machine learning model to be evaluated.
    X: Features dataset.
    y: Target dataset.
    k: Number of folds (default is 5).

    Returns:
    scores: List of scores for each fold.
    """
    fold_size = len(X) // k
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    scores = []
    
    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size
        test_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    return scores