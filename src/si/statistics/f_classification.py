

from si.data.dataset import Dataset
import scipy


def f_classification(dataset: Dataset) -> tuple:
    """
    Aim: Analysis of Variance of the dataset
    Input: dataset
    Output: tuple with F values + tuple with p values
    """
    classes = dataset.get_classes() # get the classes of the dataset

    #Group the samples by class in a list
    groups = []
    for class_ in classes:
        mask = dataset.y == class_
        group = dataset.X[ mask , :]
        groups.append(group)

    #Perform the ANOVA
    return scipy.stats.f_oneway(*groups)
