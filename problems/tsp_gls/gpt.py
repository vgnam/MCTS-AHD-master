import numpy as np

def heuristics(distance_matrix):
    row_median = np.median(distance_matrix, axis=1, keepdims=True)
    col_median = np.median(distance_matrix, axis=0, keepdims=True)
    global_10th = np.percentile(distance_matrix, 10)
    global_90th = np.percentile(distance_matrix, 90)
    avg_distance = np.mean(distance_matrix)

    # Enhanced scaling: adaptive weighting based on quartile and median deviation
    heuristics_matrix = np.where(
        (distance_matrix < global_10th) & (distance_matrix < row_median) & (distance_matrix < col_median),
        (distance_matrix / (row_median + col_median)) * 0.25 * (global_10th / avg_distance),
        np.where(
            (distance_matrix > global_90th) | (distance_matrix > row_median * 1.5) | (distance_matrix > col_median * 1.5),
            (distance_matrix / (row_median + col_median)) * 3.0 * (avg_distance / global_90th),
            (distance_matrix / (row_median + col_median)) * 1.0 * (avg_distance / np.median(distance_matrix))
        )
    )
    return heuristics_matrix


