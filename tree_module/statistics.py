import numpy as np
from scipy.spatial import distance

def compute_spacing(predictions):

    centers = []

    for _, row in predictions.iterrows():

        cx = (row.xmin + row.xmax) / 2
        cy = (row.ymin + row.ymax) / 2

        centers.append([cx, cy])

    centers = np.array(centers)

    if len(centers) < 2:
        return 0

    dist_matrix = distance.cdist(centers, centers)

    nearest = np.min(dist_matrix + np.eye(len(centers))*1e9, axis=1)

    return np.mean(nearest)