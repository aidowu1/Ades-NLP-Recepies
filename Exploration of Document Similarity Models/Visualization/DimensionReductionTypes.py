import enum


class DataReductionType(enum.Enum):
    """
    Dimensional reduction enum types
    """
    mds = 0
    pca = 1
    tsne = 3
    umap = 4