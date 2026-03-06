import numpy as np


def compute_forest_health(ndvi, tree_density):

    vegetation_score = np.mean(ndvi) / 255

    density_score = min(tree_density / 500, 1)

    health = float((0.6 * vegetation_score) + (0.4 * density_score))

    health_index = health * 100

    return round(health_index, 2)