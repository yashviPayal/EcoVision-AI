import cv2

from .detector import TreeDetector
from .density import calculate_density
from .statistics import compute_spacing


detector = TreeDetector()


def analyze_forest(image_path, resolution=10):

    predictions = detector.detect(image_path)

    tree_count = len(predictions)

    image = cv2.imread(image_path)
    height, width, _ = image.shape

    area_km2, density = calculate_density(
        tree_count,
        width,
        height,
        resolution
    )

    avg_spacing = compute_spacing(predictions)

    result = {

        "tree_count": tree_count,
        "area_km2": area_km2,
        "tree_density": density,
        "avg_tree_spacing": avg_spacing

    }

    return result