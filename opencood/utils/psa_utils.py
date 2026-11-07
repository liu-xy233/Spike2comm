"""Training-free PSA, following DerrickXuNu/model_anostic main.py.

Reference revision: 3812bd9671517d55090e47661124929658b28043.
The official row-neighborhood grouping is deliberately NOT connected components.
Original scores/coordinates are never averaged or replaced. Exact ties may select
nothing. Invalid-input filtering and optional class/batch isolation are explicit
engineering adaptations; the official implementation handles one vehicle class.
"""

import numpy as np

from opencood.utils import common_utils


PSA_DISTANCE_THRESHOLD = 10.0
PSA_TEMPERATURE = 1e-6
PSA_SELECTION_THRESHOLD = 0.5


def _boxes_array(boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 3 or boxes.shape[1:] not in ((4, 2), (8, 3)):
        raise ValueError("PSA boxes must be [N,4,2] or [N,8,3] corners")
    return boxes


def _ids_array(ids, count, name):
    if ids is None:
        return np.zeros(count, dtype=np.int64)
    ids = np.asarray(ids)
    if ids.shape != (count,) or ids.dtype.kind not in 'iu':
        raise ValueError("{} must be an integer vector of length N".format(name))
    return ids


def _polygons_and_validity(boxes):
    """Reuse OpenCOOD's rotated-corner Polygon backend, rejecting bad geometry.

    common_utils.compute_iou casts to float32. Here the same Shapely operations
    retain float64, as in the reference, because the temperature is only 1e-6.
    """
    valid = np.isfinite(boxes).all(axis=(1, 2))
    polygons = [None] * len(boxes)
    for index in np.flatnonzero(valid):
        polygon = common_utils.convert_format(boxes[index:index + 1])[0]
        valid[index] = (polygon.is_valid and not polygon.is_empty
                        and np.isfinite(polygon.area) and polygon.area > 0)
        if valid[index]:
            polygons[index] = polygon
    return polygons, valid


def compute_bev_iou_matrix(boxes, dist_threshold=PSA_DISTANCE_THRESHOLD,
                           class_ids=None, batch_ids=None):
    """Pairwise rotated BEV IoU; invalid rows/columns (including diagonal) are 0.

    Match the reference's strict distance < 10m and mean over ALL corner
    coordinate dimensions (XYZ for 3D corners, XY for 2D corners).
    Optional class/batch IDs prohibit interactions across those boundaries.
    """
    boxes = _boxes_array(boxes)
    if not np.isfinite(dist_threshold) or dist_threshold <= 0:
        raise ValueError("dist_threshold must be finite and positive")
    classes = _ids_array(class_ids, len(boxes), 'class_ids')
    batches = _ids_array(batch_ids, len(boxes), 'batch_ids')
    polygons, valid = _polygons_and_validity(boxes)
    matrix = np.zeros((len(boxes), len(boxes)), dtype=np.float64)
    indices = np.flatnonzero(valid)
    matrix[indices, indices] = 1.0
    centers = np.zeros((len(boxes), boxes.shape[-1]), dtype=np.float64)
    centers[valid] = boxes[valid].mean(axis=1)
    for i in indices:
        # Only one distance row is allocated; the IoU matrix remains O(N^2).
        candidates = indices[indices > i]
        with np.errstate(over='ignore', invalid='ignore'):
            distances = np.linalg.norm(centers[candidates] - centers[i], axis=1)
        neighbors = candidates[(distances < dist_threshold)
                               & (classes[candidates] == classes[i])
                               & (batches[candidates] == batches[i])]
        for j in neighbors:
            value = (polygons[i].intersection(polygons[j]).area
                     / polygons[i].union(polygons[j]).area)
            if np.isfinite(value):
                matrix[i, j] = matrix[j, i] = np.clip(value, 0.0, 1.0)
    return matrix


def compute_self_iou_mat(boxes, dist_threshold=PSA_DISTANCE_THRESHOLD,
                         class_ids=None, batch_ids=None):
    """Reference-compatible name for the self IoU matrix."""
    return compute_bev_iou_matrix(boxes, dist_threshold, class_ids, batch_ids)


def _validate_matrix(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or not np.isfinite(matrix).all()
            or np.any(matrix < 0) or np.any(matrix > 1)):
        raise ValueError("IoU matrix must be finite, square and in [0,1]")
    return matrix


def find_psa_clusters(iou_matrix):
    """Official ordered row neighborhoods, including overlapping groups.

    Do not recursively expand neighborhoods or remove previously visited nodes
    from a new group. Both would change the reference algorithm.
    """
    matrix = _validate_matrix(iou_matrix)
    visited = np.zeros(len(matrix), dtype=bool)
    clusters = []
    for index, row in enumerate(matrix):
        if visited[index]:
            continue
        neighbors = np.flatnonzero(row)
        if neighbors.size:  # engineering guard for invalid/empty rows
            clusters.append(neighbors)
            visited[neighbors] = True
    return clusters


def promote_scores(iou_matrix, scores):
    matrix = _validate_matrix(iou_matrix)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(matrix),) or not np.isfinite(scores).all():
        raise ValueError("Promotion needs one finite score per matrix row")
    return matrix @ scores


def suppress_scores(updated_scores, temperature=PSA_TEMPERATURE):
    """Numerically stable softmax, with the official temperature by default."""
    values = np.asarray(updated_scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Suppression needs a finite score vector")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if values.size == 0:
        return values.copy()
    # Equivalent to softmax(values / temperature); subtract first to avoid
    # overflow for unusually small (explicitly supplied) temperatures.
    with np.errstate(over='ignore', under='ignore'):
        weights = np.exp((values - values.max()) / temperature)
    return weights / weights.sum()


def promote_suppress_aggregation(boxes, scores, class_ids=None, batch_ids=None,
                                 dist_threshold=PSA_DISTANCE_THRESHOLD,
                                 temperature=PSA_TEMPERATURE,
                                 selection_threshold=PSA_SELECTION_THRESHOLD,
                                 return_details=False):
    """Return selected ORIGINAL input indices, retaining official order/ties.

    No NMS, coordinate voting, score reweighting or argmax fallback is applied.
    Nonfinite/out-of-range probabilities and invalid boxes are discarded before
    promotion. Optional IDs isolate classes/scenes, without changing one-class
    behavior. The reference can select an index in multiple overlapping groups;
    this low-level function deliberately preserves that behavior.
    """
    boxes = _boxes_array(boxes)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(boxes),):
        raise ValueError("scores must have shape [N]")
    if (not np.isfinite(selection_threshold)
            or not 0 <= selection_threshold <= 1):
        raise ValueError("selection_threshold must be in [0,1]")
    # Validate even for an empty input.
    suppress_scores(np.empty(0), temperature)
    classes = _ids_array(class_ids, len(boxes), 'class_ids')
    batches = _ids_array(batch_ids, len(boxes), 'batch_ids')
    _, valid = _polygons_and_validity(boxes)
    valid &= np.isfinite(scores) & (scores >= 0) & (scores <= 1)
    original_indices = np.flatnonzero(valid)
    matrix = compute_self_iou_mat(boxes[valid], dist_threshold,
                                  classes[valid], batches[valid])
    clusters = find_psa_clusters(matrix)
    selected, promoted, soft = [], [], []
    for cluster in clusters:
        values = promote_scores(matrix[np.ix_(cluster, cluster)], scores[valid][cluster])
        probabilities = suppress_scores(values, temperature)
        selected.extend(original_indices[cluster[probabilities > selection_threshold]])
        promoted.append(values)
        soft.append(probabilities)
    selected = np.asarray(selected, dtype=np.int64)
    if return_details:
        return selected, {
            'valid_indices': original_indices,
            'clusters': [original_indices[cluster] for cluster in clusters],
            'promoted_scores': promoted,
            'soft_scores': soft,
        }
    return selected
