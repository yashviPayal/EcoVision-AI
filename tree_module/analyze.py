# analyze.py
import cv2
import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt

from .detector import TreeDetector
from .density import calculate_density
from .statistics import compute_spacing
from .heatmap import generate_density_heatmap
from .ndvi import compute_ndvi_proxy
from .visualization import draw_detections
from .health import compute_forest_health

# Initialize YOLO detector
detector = TreeDetector()

# Setup CNN model for embeddings
cnn_model = models.resnet50(pretrained=True)
cnn_model = torch.nn.Sequential(*list(cnn_model.children())[:-1])  # remove classifier
cnn_model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


def crop_and_save_trees(image, predictions, output_dir="tree_crops"):
    os.makedirs(output_dir, exist_ok=True)
    crop_paths = []
    for i, row in predictions.iterrows():
        xmin, ymin, xmax, ymax = map(int, [row.xmin, row.ymin, row.xmax, row.ymax])
        crop = image[ymin:ymax, xmin:xmax]
        crop_path = os.path.join(output_dir, f"tree_{i}.png")
        cv2.imwrite(crop_path, crop)
        crop_paths.append(crop_path)
    return crop_paths


def get_embeddings(crop_paths):
    embeddings = []
    for path in crop_paths:
        img = Image.open(path).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = cnn_model(tensor).squeeze().numpy()
        embeddings.append(feat)
    return np.array(embeddings)


def cluster_trees(embeddings, n_clusters=5, visualize=True, save_path="tree_clusters.png"):
    # Dimensionality reduction for clustering/visualization
    reducer = umap.UMAP(n_components=2)
    emb_2d = reducer.fit_transform(embeddings)

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(emb_2d)

    # Optional visualization
    if visualize:
        plt.figure(figsize=(8,6))
        plt.scatter(emb_2d[:,0], emb_2d[:,1], c=labels, cmap='tab10', s=20)
        plt.title("Tree Clusters (Species-like groups)")
        plt.savefig(save_path)
        plt.close()

    return labels


def analyze_forest(image_path, resolution=0.5, n_clusters=5):
    # Read image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Detect trees
    predictions = detector.detect(image_path)
    tree_count = len(predictions)

    # Compute area and density
    area_km2, density = calculate_density(tree_count, width, height, resolution)

    # Compute average spacing
    spacing = compute_spacing(predictions)

    # Generate density heatmap
    heatmap = generate_density_heatmap(image, predictions)

    # NDVI proxy
    ndvi_map, ndvi_raw = compute_ndvi_proxy(image)

    # Annotated image with detections
    annotated = draw_detections(image, predictions)

    # Forest health score
    health_score = compute_forest_health(ndvi_raw, density)

    # Crop trees and save individual images
    crop_paths = crop_and_save_trees(image, predictions)

    # Extract embeddings
    embeddings = get_embeddings(crop_paths)

    # Cluster trees into species-like groups
    labels = cluster_trees(embeddings, n_clusters=n_clusters)

    # Save annotated outputs
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/tree_density_heatmap.png", heatmap)
    cv2.imwrite("outputs/vegetation_health_map.png", ndvi_map)
    cv2.imwrite("outputs/tree_detections.png", annotated)

    # Save cluster labels mapping
    with open("outputs/tree_clusters.txt", "w") as f:
        for path, label in zip(crop_paths, labels):
            f.write(f"{path} -> Cluster {label}\n")

    # Return results
    result = {
        "tree_count": tree_count,
        "area_km2": area_km2,
        "tree_density": density,
        "avg_tree_spacing": float(spacing),
        "forest_health_score": health_score,
        "tree_clusters": list(labels),
        "outputs": {
            "detections": "outputs/tree_detections.png",
            "density_heatmap": "outputs/tree_density_heatmap.png",
            "ndvi_map": "outputs/vegetation_health_map.png",
            "cluster_visualization": "tree_clusters.png",
            "cluster_mapping": "outputs/tree_clusters.txt"
        }
    }

    return result
