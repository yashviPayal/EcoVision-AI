import cv2

from .detector import TreeDetector
from .density import calculate_density
from .statistics import compute_spacing
from .heatmap import generate_density_heatmap
from .ndvi import compute_ndvi_proxy
from .visualization import draw_detections
from .health import compute_forest_health


detector = TreeDetector()


def analyze_forest(image_path, resolution=0.5):

    image = cv2.imread(image_path)

    predictions = detector.detect(image_path)

    tree_count = len(predictions)

    height, width, _ = image.shape


    area_km2, density = calculate_density(
        tree_count,
        width,
        height,
        resolution
    )

    spacing = compute_spacing(predictions)


    heatmap = generate_density_heatmap(image, predictions)

    ndvi_map, ndvi_raw = compute_ndvi_proxy(image)

    annotated = draw_detections(image, predictions)


    health_score = compute_forest_health(ndvi_raw, density)


    cv2.imwrite("outputs/tree_density_heatmap.png", heatmap)
    cv2.imwrite("outputs/vegetation_health_map.png", ndvi_map)
    cv2.imwrite("outputs/tree_detections.png", annotated)


    result = {

        "tree_count": tree_count,

        "area_km2": area_km2,

        "tree_density": density,

        "avg_tree_spacing": float(spacing),

        "forest_health_score": health_score,

        "outputs": {

            "detections": "outputs/tree_detections.png",

            "density_heatmap": "outputs/tree_density_heatmap.png",

            "ndvi_map": "outputs/vegetation_health_map.png"
        }

    }

    return result