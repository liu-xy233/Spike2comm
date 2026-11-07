"""Small, dependency-light CLI helpers for opt-in PSA inference."""

import json
import os


POSTPROCESS_METHODS = ('baseline', 'psa_strict', 'psa_hybrid')


def add_psa_arguments(parser):
    parser.add_argument('--postprocess_method', choices=POSTPROCESS_METHODS,
                        default='baseline', help='Inference aggregation; default preserves legacy behavior')
    parser.add_argument('--eval_output_dir', default=None,
                        help='New, non-existing result directory; required for PSA')
    parser.add_argument('--count_empty_predictions', action='store_true',
                        help='Count GT on empty-prediction frames for ALL compared runs; required for PSA')


def configure_psa_inference(hypes, opt):
    """Only override the runtime postprocessor; never edit a saved config."""
    mode = opt.postprocess_method
    if mode not in POSTPROCESS_METHODS:
        raise ValueError("Unknown postprocess_method: {}".format(mode))
    if mode != 'baseline':
        if opt.fusion_method not in ('intermediate', 'intermediate_with_comm'):
            raise ValueError("PSA requires intermediate or intermediate_with_comm fusion")
        if hypes['postprocess'].get('core_method') != 'VoxelPostprocessor':
            raise ValueError("PSA requires VoxelPostprocessor")
        if not opt.eval_output_dir:
            raise ValueError("PSA requires --eval_output_dir pointing to a NEW directory")
        if not opt.count_empty_predictions:
            raise ValueError("PSA can produce empty frames on exact ties. Use "
                             "--count_empty_predictions for PSA AND its baseline/control; "
                             "legacy baseline skips empty frames.")
    # Explicit baseline also disables a mode loaded from an optional YAML copy.
    hypes['postprocess']['postprocess_method'] = mode


def prepare_output_directory(opt):
    if opt.eval_output_dir is None:
        return opt.model_dir
    target = os.path.abspath(opt.eval_output_dir)
    if target == os.path.abspath(opt.model_dir):
        raise ValueError("Evaluation output must not overwrite the model directory")
    os.makedirs(target, exist_ok=False)
    return target


def write_psa_diagnostics(output_dir, frames, result_stat, comm_rate, opt):
    metrics = {}
    for threshold, values in result_stat.items():
        true_positives = int(sum(values['tp']))
        metrics[str(threshold)] = {
            'recall': true_positives / values['gt'] if values['gt'] else 0.0,
            'fp': int(sum(values['fp'])), 'gt': int(values['gt']),
        }
    measured_comm = opt.fusion_method == 'intermediate_with_comm'
    summary = {
        'postprocess_method': opt.postprocess_method,
        'empty_prediction_policy': 'count_gt',
        'metrics': metrics, 'frames': frames,
        'mean_psa_seconds': sum(f['psa_seconds'] for f in frames) / max(1, len(frames)),
        'communication_rate': float(comm_rate) if measured_comm else None,
        'communication_rate_measured': measured_comm,
        'communication_note': 'Existing feature-mask rate; additional box-message bytes are not included.',
        'score_definition': 'raw_sigmoid', 'quality_rectification': False,
        'late_fusion': False, 'nms': False, 'rbv': False,
    }
    with open(os.path.join(output_dir, 'psa_diagnostics.json'), 'w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)
