# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


import glob
import importlib
from tkinter.messagebox import NO
import yaml
import os
import re
from datetime import datetime
import shutil
import torch
import torch.optim as optim

def backup_script(full_path, folders_to_save=["models", "data_utils", "utils", "loss"]):
    target_folder = os.path.join(full_path, 'scripts')
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    
    current_path = os.path.dirname(__file__)  # __file__ refer to this file, then the dirname is "?/tools"

    for folder_name in folders_to_save:
        ttarget_folder = os.path.join(target_folder, folder_name)
        source_folder = os.path.join(current_path, f'../{folder_name}')
        shutil.copytree(source_folder, ttarget_folder)

def _find_last_checkpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*epoch(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        return max(epochs_exist)
    return 0


def _normalize_state_dict_keys(state_dict_):
    state_dict = {}
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    return state_dict


def _resolve_checkpoint_path(path, epoch=None):
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, 'net_latest.pth')):
            return os.path.join(path, 'net_latest.pth'), 100
        initial_epoch = _find_last_checkpoint(path) if epoch is None else int(epoch)
        if initial_epoch <= 0:
            raise FileNotFoundError('No net_epoch*.pth checkpoint found in {}'.format(path))
        return os.path.join(path, 'net_epoch%d.pth' % initial_epoch), initial_epoch
    if os.path.isfile(path):
        result = re.findall(r".*epoch(\d+)\.pth.*", path)
        checkpoint_epoch = int(result[0]) if result else 0
        return path, checkpoint_epoch
    raise FileNotFoundError('{} not found'.format(path))


def load_pretrained_model(pretrained_path, model, allowed_missing_prefixes=None, allow_unexpected=False):
    """Load model weights for initialization without resuming training state.

    This intentionally does not restore optimizer, scheduler, or epoch. It is
    meant for cases such as initializing a DFA model from a baseline checkpoint.
    """
    allowed_missing_prefixes = tuple(allowed_missing_prefixes or ())
    # ATTEN has no parameters. Initializing the new ESSA experiment therefore
    # legitimately introduces exactly the namespaces of its instantiated modules.
    from opencood.models.fuse_modules.sdtv3_attention_fusion import (
        sdtv3_fusion_state_prefixes,
    )
    allowed_missing_prefixes += sdtv3_fusion_state_prefixes(model)
    # IoU-quality heads are optional additions when initializing from legacy
    # checkpoints. Missing keys are still printed below for an explicit audit.
    allowed_missing_prefixes += ('iou_head.',)
    checkpoint_path, checkpoint_epoch = _resolve_checkpoint_path(pretrained_path)
    print('initializing model weights from {}'.format(checkpoint_path))
    if checkpoint_epoch:
        print('source checkpoint epoch {}'.format(checkpoint_epoch))

    loaded = torch.load(checkpoint_path, map_location='cpu')
    state_dict = _normalize_state_dict_keys(loaded)
    model_state_dict = model.state_dict()

    compatible_state_dict = {}
    skipped_shape = []
    unexpected_keys = []
    for k, v in state_dict.items():
        if k not in model_state_dict:
            unexpected_keys.append(k)
            continue
        if v.shape != model_state_dict[k].shape:
            skipped_shape.append((k, tuple(model_state_dict[k].shape), tuple(v.shape)))
            continue
        compatible_state_dict[k] = v

    incompatible = model.load_state_dict(compatible_state_dict, strict=False)
    missing_keys = list(incompatible.missing_keys)

    print('Pretrained load missing keys ({}):'.format(len(missing_keys)))
    for key in missing_keys:
        print('  {}'.format(key))
    print('Pretrained load unexpected keys ({}):'.format(len(unexpected_keys)))
    for key in unexpected_keys:
        print('  {}'.format(key))
    if skipped_shape:
        print('Pretrained load shape-mismatched keys ({}):'.format(len(skipped_shape)))
        for key, required, loaded_shape in skipped_shape:
            print('  {} required {} loaded {}'.format(key, required, loaded_shape))

    disallowed_missing = [
        key for key in missing_keys
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if disallowed_missing:
        raise RuntimeError(
            'Pretrained checkpoint is missing parameters outside allowed new modules: {}'.format(
                ', '.join(disallowed_missing[:20])
            )
        )
    if unexpected_keys and not allow_unexpected:
        raise RuntimeError(
            'Pretrained checkpoint has unexpected parameters: {}'.format(
                ', '.join(unexpected_keys[:20])
            )
        )
    if skipped_shape:
        raise RuntimeError(
            'Pretrained checkpoint has shape-mismatched parameters: {}'.format(
                ', '.join(item[0] for item in skipped_shape[:20])
            )
        )
    return model


def _validate_sdtv3_resume(state_dict, model):
    """Never silently replace a missing/incompatible ESSA branch on resume."""
    from opencood.models.fuse_modules.sdtv3_attention_fusion import (
        sdtv3_fusion_state_prefixes,
    )
    prefixes = sdtv3_fusion_state_prefixes(model)
    expected = model.state_dict()
    invalid = [
        key for key, value in expected.items()
        if key.startswith(prefixes)
        and (key not in state_dict or state_dict[key].shape != value.shape)
    ]
    unexpected = [
        key for key in state_dict
        if ('.essa.' in key or key.startswith('essa.')) and key not in expected
    ]
    if invalid or unexpected:
        raise RuntimeError(
            'ESSA checkpoint/config mismatch. Use --pretrained_path to initialize '
            'a new ESSA experiment from an ATTEN checkpoint; resume/inference '
            'requires the matching ESSA config and all ESSA weights. '
            'Missing/mismatched: {}; unexpected: {}'.format(invalid, unexpected)
        )


def load_saved_model(saved_path, model, epoch=None):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)
    
    if os.path.exists(os.path.join(saved_path, 'net_latest.pth')):
        state_dict = _normalize_state_dict_keys(torch.load(
            os.path.join(saved_path, 'net_latest.pth')))
        _validate_sdtv3_resume(state_dict, model)
        model_state_dict = model.state_dict()
        for k in list(state_dict.keys()):
            if k not in model_state_dict:
                print('Drop parameter {}.'.format(k))
                state_dict.pop(k)
            elif state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}.'.format(
                          k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        for k in model_state_dict:
            if k not in state_dict:
                print('No param {}.'.format(k))
                state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        return 100, model
    else:
        if epoch is None:
            initial_epoch = _find_last_checkpoint(saved_path)
        else:
            initial_epoch = int(epoch)
            
        if initial_epoch > 0:
            print('resuming by loading epoch %d' % initial_epoch)
        
        state_dict_ = torch.load(os.path.join(saved_path, 'net_epoch%d.pth' % initial_epoch))
        state_dict = _normalize_state_dict_keys(state_dict_)
        _validate_sdtv3_resume(state_dict, model)
        
        model_state_dict = model.state_dict()

        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, ' \
                        'loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k))
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k))
                state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        return initial_epoch, model



def setup_train(hypes):
    """
    Create folder for saved model based on current timestep and model name

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    model_name = hypes['name']
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../logs')

    full_path = os.path.join(current_path, folder_name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)
        # save the yaml file
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

    backup_script(full_path)

    return full_path


def create_model(hypes):
    """
    Import the module "models/[model_name].py

    Parameters
    __________
    hypes : dict
        Dictionary containing parameters.

    Returns
    -------
    model : opencood,object
        Model object.
    """
    backbone_name = hypes['model']['core_method']
    backbone_config = hypes['model']['args']

    model_filename = "opencood.models." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace('_', '')

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print('backbone not found in models folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (model_filename,
                                                       target_model_name))
        exit(0)
    instance = model(backbone_config)
    return instance


def create_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    loss_func_name = hypes['loss']['core_method']
    loss_func_config = hypes['loss']['args']

    loss_filename = "opencood.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion


def setup_optimizer(hypes, model):
    """
    Create optimizer corresponding to the yaml file

    Parameters
    ----------
    hypes : dict
        The training configurations.
    model : opencood model
        The pytorch model
    """
    method_dict = hypes['optimizer']
    optimizer_method = getattr(optim, method_dict['core_method'], None)
    if not optimizer_method:
        raise ValueError('{} is not supported'.format(method_dict['name']))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError('No trainable parameters were found for optimizer setup.')
    if 'args' in method_dict:
        return optimizer_method(trainable_params,
                                lr=method_dict['lr'],
                                **method_dict['args'])
    else:
        return optimizer_method(trainable_params,
                                lr=method_dict['lr'])


def setup_lr_schedular(hypes, optimizer, init_epoch=None):
    """
    Set up the learning rate schedular.

    Parameters
    ----------
    hypes : dict
        The training configurations.

    optimizer : torch.optimizer
    """
    lr_schedule_config = hypes['lr_scheduler']
    last_epoch = init_epoch if init_epoch is not None else 0
    

    if lr_schedule_config['core_method'] == 'step':
        from torch.optim.lr_scheduler import StepLR
        step_size = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config['core_method'] == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = MultiStepLR(optimizer,
                                milestones=milestones,
                                gamma=gamma)

    else:
        from torch.optim.lr_scheduler import ExponentialLR
        gamma = lr_schedule_config['gamma']
        scheduler = ExponentialLR(optimizer, gamma)

    for _ in range(last_epoch):
        scheduler.step()

    return scheduler


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str):
            return inputs
        return inputs.to(device)
