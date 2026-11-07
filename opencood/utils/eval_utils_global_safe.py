import os
import numpy as np

from opencood.utils import common_utils
from opencood.hypes_yaml import yaml_utils


def voc_ap(rec, prec):
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    return ap, mrec, mpre



def _cross_2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _signed_polygon_area(poly):
    poly = np.asarray(poly, dtype=np.float64)
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * np.sum(
        x * np.roll(y, -1) -
        y * np.roll(x, -1)
    )


def _polygon_area(poly):
    return abs(_signed_polygon_area(poly))


def _line_intersection(p1, p2, q1, q2):
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    r = p2 - p1
    s = q2 - q1
    denom = _cross_2d(r, s)

    if abs(denom) < 1e-12:
        return p2.copy()

    t = _cross_2d(q1 - p1, s) / denom
    return p1 + t * r


def _clip_convex_polygon(subject, clip):
    subject = np.asarray(subject, dtype=np.float64)
    clip = np.asarray(clip, dtype=np.float64)

    if subject.shape[0] < 3 or clip.shape[0] < 3:
        return np.empty((0, 2), dtype=np.float64)

    orientation = 1.0 if _signed_polygon_area(clip) >= 0 else -1.0

    output = [p.copy() for p in subject]

    for i in range(clip.shape[0]):
        cp1 = clip[i]
        cp2 = clip[(i + 1) % clip.shape[0]]

        input_list = output
        output = []

        if not input_list:
            break

        def inside(p):
            return (
                orientation *
                _cross_2d(cp2 - cp1, p - cp1)
                >= -1e-9
            )

        s = input_list[-1]

        for e in input_list:
            e_inside = inside(e)
            s_inside = inside(s)

            if e_inside:
                if not s_inside:
                    output.append(
                        _line_intersection(s, e, cp1, cp2)
                    )
                output.append(e)
            elif s_inside:
                output.append(
                    _line_intersection(s, e, cp1, cp2)
                )

            s = e

    if not output:
        return np.empty((0, 2), dtype=np.float64)

    return np.asarray(output, dtype=np.float64)


def _box_to_polygon_xy(box):
    box = np.asarray(box, dtype=np.float64)

    if (
        box.ndim != 2 or
        box.shape[0] < 4 or
        box.shape[1] < 2
    ):
        return np.empty((0, 2), dtype=np.float64)

    poly = box[:4, :2]

    if not np.all(np.isfinite(poly)):
        return np.empty((0, 2), dtype=np.float64)

    return poly


def _convex_polygon_iou(poly_a, poly_b):
    if poly_a.shape[0] < 3 or poly_b.shape[0] < 3:
        return 0.0

    area_a = _polygon_area(poly_a)
    area_b = _polygon_area(poly_b)

    if area_a <= 1e-12 or area_b <= 1e-12:
        return 0.0

    intersection = _clip_convex_polygon(poly_a, poly_b)
    inter_area = _polygon_area(intersection)

    union = area_a + area_b - inter_area

    if union <= 1e-12:
        return 0.0

    return float(inter_area / union)


def caluclate_tp_fp(det_boxes,
                    det_score,
                    gt_boxes,
                    result_stat,
                    iou_thresh):

    fp = []
    tp = []
    gt = gt_boxes.shape[0]

    if det_boxes is not None:
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        score_order_descend = np.argsort(-det_score)

        sorted_scores = det_score[score_order_descend]

        det_polygon_list = [
            _box_to_polygon_xy(det_boxes[idx])
            for idx in score_order_descend
        ]

        gt_polygon_list = [
            _box_to_polygon_xy(box)
            for box in gt_boxes
        ]

        for det_polygon in det_polygon_list:

            if len(gt_polygon_list) == 0:
                fp.append(1)
                tp.append(0)
                continue

            ious = np.asarray(
                [
                    _convex_polygon_iou(
                        det_polygon,
                        gt_polygon
                    )
                    for gt_polygon in gt_polygon_list
                ],
                dtype=np.float32,
            )

            if ious.size == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = int(np.argmax(ious))
            gt_polygon_list.pop(gt_index)

        result_stat[iou_thresh]['score'] += \
            sorted_scores.tolist()

    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt


def calculate_ap(result_stat, iou):
    stat = result_stat[iou]

    fp = np.array(stat['fp'])
    tp = np.array(stat['tp'])
    score = np.array(stat['score'])

    assert len(fp) == len(tp)
    assert len(tp) == len(score)

    # Critical CoSDH/CoAlign correction:
    # globally sort ALL validation predictions by confidence.
    sorted_index = np.argsort(-score)

    fp = fp[sorted_index].tolist()
    tp = tp[sorted_index].tolist()

    gt_total = stat['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (
            fp[idx] + tp[idx]
        )

    ap, mrec, mprec = voc_ap(
        rec[:],
        prec[:]
    )

    return ap, mrec, mprec


def eval_final_results(result_stat,
                       save_path,
                       noise_level=None):

    dump_dict = {}

    ap_30, mrec_30, mpre_30 = \
        calculate_ap(result_stat, 0.30)

    ap_50, mrec_50, mpre_50 = \
        calculate_ap(result_stat, 0.50)

    ap_70, mrec_70, mpre_70 = \
        calculate_ap(result_stat, 0.70)

    dump_dict.update({
        'ap_30': ap_30,
        'ap_50': ap_50,
        'ap_70': ap_70,
        'mpre_50': mpre_50,
        'mrec_50': mrec_50,
        'mpre_70': mpre_70,
        'mrec_70': mrec_70,
    })

    yaml_utils.save_yaml(
        dump_dict,
        os.path.join(save_path, 'eval_global.yaml')
    )

    print(
        'GLOBAL-SORT AP | '
        'AP@0.3: %.4f | '
        'AP@0.5: %.4f | '
        'AP@0.7: %.4f'
        % (ap_30, ap_50, ap_70)
    )

    return ap_30, ap_50, ap_70
