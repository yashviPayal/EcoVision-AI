def calculate_density(tree_count, width_pixels, height_pixels, resolution):

    area_m2 = width_pixels * height_pixels * (resolution ** 2)

    area_km2 = area_m2 / 1e6

    density = tree_count / area_km2

    return area_km2, density