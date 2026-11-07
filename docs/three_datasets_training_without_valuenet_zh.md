# 三个数据集的完整检测器训练与评估运行文档（不含 ValueNet）

核对日期：2026-11-08。依据当前工作区源码，不以旧 README、历史迁移计划或配置文件名推断实际功能。

本文覆盖 DAIR-V2X、OPV2V、V2X-Sim 2.0 的数据准备、独立运行配置、启动前检查、训练、续训、验证集选权重及最终评估。三套模型分别在各自数据集上训练，默认从头初始化，不是三个数据集依次接力训练同一个模型。

本文采用的完整网络为：均值体素编码 → E3DSNN 稀疏 3D 主干 → 多尺度 BEV 增强及脉冲残差 neck → 本地检测头产生置信图 → 置信度阈值通信 → 坐标对齐 → SDTv3 ESSA 融合 → 分类、框回归和 IoU 质量头。评估还包含配置启用的 confidence-aware late fusion。上述检测模块在同一次检测器训练中联合优化，不需要先单独训练 neck、ESSA 或 IoU 头；late fusion 是后处理，没有单独的训练阶段。

排除范围包括 ValueNet/CSVN 的训练与推理，以及为其服务的 Oracle 标签生成、ego 特征缓存、标签对齐和 Top-K 回放。普通检测器训练入口不会调用 ValueNet；无需准备任何 ValueNet 权重。通信仍由本地分类置信图、高斯平滑和硬阈值选择完成。

**验证边界：本次完成了源码核对、配置派生及文档代码的静态检查；没有执行真实数据的 GPU 训练或全量 AP 评估。运行环境仍须通过下文检查。**

## 1. 先明确配置来源和必要调整

采用以下文件作为配置来源，随后生成独立文件，不覆盖原配置：

| 数据集 | 来源文件（相对于仓库根目录） | 源配置状态 | 本文运行配置的调整 |
|---|---|---|---|
| DAIR-V2X | `opencood/hypes_yaml/dair-v2x/dair_where2comm_e3dsnn_sdtv3_essa_snnresneck_20261024.yaml` | ESSA；默认 `legacy_dair` 监督 | 显式写出监督模式；修改数据路径 |
| OPV2V | `opencood/hypes_yaml/opv2v/opv2v_e3dsnn_where2comm_ddp_bs4_A_50epoch.yaml` | **ATTEN**；`cooperative_only`；使用 `load_point_pillar_params` | 改为 ESSA；改用 E3DSNN parser，将 IoU 配置同步到 loss、model 和 postprocess |
| V2X-Sim 2.0 | `opencood/hypes_yaml/v2xsim/v2xsim_e3dsnn_where2comm_sdtv3_essa.yaml` | ESSA；未写监督模式 | **必须显式设置 `cooperative_only`**；设置 PKL 和原始点云路径 |

这里的 OPV2V ESSA 配置是根据现有模块接口派生的新实验配置，不代表仓库已有 OPV2V ESSA 的完整训练结果。如果要复现原 OPV2V ATTEN 实验，应使用对应原配置及原权重，单独记录实验名称，不能把它称为本文统一的 ESSA 实验。

V2X-Sim 的调整不是可忽略的命名变化：默认 loss 模式是 `legacy_dair`，主模型会输出 `_single_v/_single_i` 别名，但 V2X-Sim reader 不提供对应单端标签，公共 loss 汇总函数会抛出 `Incomplete legacy DAIR auxiliary supervision`。设置 `cooperative_only` 才与当前 reader 的标签契约相符。OPV2V、V2X-Sim 均不能直接改成 `all_agents_aux`，因为当前 reader 不提供该模式要求的 `label_dict_all_agents`。

本文保留各数据集的空间范围和训练超参数，不把它们伪装成完全相同的实验协议：

| 项目 | DAIR-V2X | OPV2V | V2X-Sim 2.0 |
|---|---|---|---|
| Dataset 类 | `IntermediateFusionDatasetDAIRE3DSNN` | `IntermediateFusionDatasetOPV2VE3DSNN` | `IntermediateFusionDatasetV2XSimE3DSNN` |
| xyz 范围，单位 m | `[-100.8,-40,-3,100.8,40,1]` | `[-140.8,-38.4,-3,140.8,38.4,1]` | `[-32,-32,-3,32,32,2]` |
| xyz 体素大小，单位 m | `[0.2,0.2,0.1]` | 同左 | 同左 |
| 预期检测网格 H×W | `50×126` | `48×176` | `40×40` |
| max_cav，含 ego | 2 | 5 | 5 |
| ego 规则 | 车端 | 训练时按场景固定种子打乱车辆顺序；验证/测试按车辆 ID 排序取首车 | 固定 Agent 1 |
| 邻居距离筛选 | 当前 DAIR reader 没有启用距离排除 | 70 m | 50 m |
| 监督 | fused + 车端 local + 路端 local | 仅 fused | 仅 fused |
| batch_size | 4 | 4 | 4 |
| 总 epoch | 30 | 50 | 30 |
| Adam 初始 lr | 0.004 | 0.002 | 0.004 |
| MultiStepLR 配置 | milestones `[10,20]`，gamma `0.1` | 同左 | 同左 |
| 通信阈值 | 0.05 | 0.05 | 0.05 |

H×W 是按当前范围、体素和 stride=8 推导的期望值，须在真实前向中核对。OPV2V 的随机 ego 是构建数据集时按场景确定的顺序，不是每次取样或每个 epoch 重新随机。

每条受监督分支的目标为 `L_branch = 1.0 × L_cls + 2.0 × L_reg + 1.0 × L_iou`：分类使用 sigmoid focal loss，框回归使用 Smooth L1，IoU 头对正 anchor 的预测框与匹配目标框计算 detached 3D IoU，以 `2×IoU−1` 为质量监督目标。DAIR 将 fused、车端 local、路端 local 三条分支相加；另外两套仅计算 fused 分支。仅 fused 监督仍会更新共享主干、融合模块和检测头，但不等于每个 agent 都有独立本地标签监督。通信硬阈值本身不是可微的训练目标。

## 2. 环境准备与仓库路径

下文命令统一采用 **Linux/WSL 的 Bash 和 NVIDIA CUDA 环境**。当前 Windows 工作区对应 WSL 路径通常是 `/mnt/d/model/Where2comm-main-new`；在服务器上替换为实际仓库路径。多卡入口固定使用 NCCL，不要在原生 Windows PowerShell 中直接照抄 `torchrun` 示例。

```bash
cd /mnt/d/model/Where2comm-main-new
# 在这里激活已经具备 PyTorch/CUDA/spconv 的训练环境。
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p run_configs run_logs
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.device_count())'
```

仓库 `requirements.txt` 和 [INSTALL.md](INSTALL.md) 保留了早期依赖版本，例如 `numba==0.49.0`、旧 scipy/matplotlib，以及历史 PyTorch/spconv 安装方案；它们不是当前分支完整验证过的环境锁文件。不要在已有可用环境里直接全量重装这些旧版本。当前源码同时尝试 spconv 2.x 的 `spconv.pytorch` 和旧接口，但这不意味着任意 Python、PyTorch、CUDA、spconv/cumm 组合都兼容。

已有训练环境需要包含：PyTorch、匹配的 spconv/cumm、numpy、PyYAML、scipy、numba、OpenCV、matplotlib、open3d、Cython、tensorboardX、shapely、einops、easydict，以及源码额外引用的 `icecream`、`pyquaternion`。`train_utils.py` 还导入 `tkinter.messagebox`，精简 Linux Python 环境须能导入 tkinter。按下面检查确定缺失项，再补齐对应环境依赖。

在当前 Python/CUDA 环境中编译仓库扩展；CUDA 编译器和相关头文件需可用：

```bash
python opencood/utils/setup.py build_ext --inplace
python opencood/pcdet_utils/setup.py build_ext --inplace
python - <<'PY'
import torch
import numpy, yaml, scipy, numba, cv2, matplotlib, open3d
import tensorboardX, shapely, einops, easydict, icecream, pyquaternion, tkinter
from opencood.models.sub_modules.e3dsnn_spconv_backbone import E3DSNNSparseBEVBackbone
from opencood.utils.box_overlaps import bbox_overlaps
from opencood.pcdet_utils.iou3d_nms import iou3d_nms_cuda
from opencood.data_utils.datasets import build_dataset
assert torch.cuda.is_available(), '本流程需要可用 CUDA'
print('依赖和编译扩展可导入；下一步还需真实数据前向检查。')
PY
python -m pip freeze > run_logs/environment.txt
```

本流程通过 `PYTHONPATH` 从仓库导入代码，无需为了执行训练而调用会自动解析旧 `install_requires` 的安装命令。导入成功只能证明依赖可加载，不证明 CUDA 扩展运行、数据格式及显存均满足训练要求。

## 3. 准备三个数据集

### 3.1 DAIR-V2X

本 reader 使用车路协同点云、标定、协同 world 标签和双方本地标签。目录至少应包含：

```text
cooperative-vehicle-infrastructure/
├── train.json                         # 车辆帧 ID 的 JSON 列表
├── val.json                           # 同上，验证划分
├── cooperative/
│   ├── data_info.json
│   └── label_world/*.json
├── vehicle-side/
│   ├── velodyne/...
│   ├── calib/lidar_to_novatel/*.json
│   ├── calib/novatel_to_world/*.json
│   └── label/lidar/*.json
└── infrastructure-side/
    ├── velodyne/...
    ├── calib/virtuallidar_to_world/*.json
    └── label/virtuallidar/*.json
```

点云实际相对路径取自 `cooperative/data_info.json`，目录树中的 `velodyne` 仅示意；以索引指向的文件为准。split JSON 应是类似 `["000001", "000002"]` 的帧 ID 列表，不是包含 train/val/test 三个键的总划分字典。车辆帧 ID 必须能匹配协同索引。

协同标签目录使用 `cooperative_label_dir`，车辆本地标签使用 `vehicle_label_dir`；它们可以指向独立补充标注目录。路侧本地标签仍从 `data_dir/infrastructure-side/label/virtuallidar` 读取。不要把仅有 cooperative 标签的数据包误认为已满足三路监督。

原 DAIR 配置的 `test_dir` 实际指向 `val.json`。本文保留这一已有行为，并明确将结果称为 **DAIR 验证集结果**。只有具备独立、可评估且有匹配标注的测试划分时，才将其替换为测试 JSON 并报告测试集指标。

### 3.2 OPV2V

reader 接收解压后的场景目录，不接收 DAIR split JSON 或 V2X-Sim PKL：

```text
opv2v/
├── train/<scenario>/<agent_id>/<timestamp>.yaml
│                                 └──── <timestamp>.pcd
├── validate/<scenario>/<agent_id>/...
└── test/<scenario>/<agent_id>/...
```

每个时间戳的 YAML 提供位姿及目标信息，PCD 提供点云。`root_dir/validate_dir/test_dir` 分别指向 split 根目录。实际目录若叫 `val`，把配置改成实际名称即可。当前配置关闭异步和定位误差，使用本地坐标点云及后续特征对齐。

### 3.3 V2X-Sim 2.0

当前实现需要 **CoSDH 兼容格式的 infos PKL 加原始 LiDAR 文件**：

```text
v2xsim/
├── v2xsim2_info/
│   ├── v2xsim_infos_train.pkl
│   ├── v2xsim_infos_val.pkl
│   └── v2xsim_infos_test.pkl
└── V2X-Sim-2.0/
    └── sweeps/...
```

PKL 顶层为非空 list/tuple，每条记录必须有 `agent_num`、`sample_token`、`timestamp`，并对 `i=1..agent_num` 提供：

| 字段 | 要求 |
|---|---|
| `lidar_path_i` | 能通过当前路径映射找到的原始点云文件 |
| `lidar_pose_i` | 有限值的 `4×4` 全局位姿矩阵 |
| `labels_i.gt_boxes_global` | `N×10` world 框：`[x,y,z,l,w,h,qw,qx,qy,qz]` |
| `labels_i.gt_object_ids` | 与框一一对应、可用于跨 agent 去重的目标 ID |

原始点云按 float32、每点 5 列读取，只取前 4 列作为模型输入。仅有 occupancy、voxel_indices、其他模型的稀疏回归标签，或只有顶层 `gt_boxes_global` 的可视化 PKL，均不满足该 reader。

`v2xsim.raw_root` 指向原始数据根目录；旧绝对路径先按 `legacy_root` 去前缀，必要时按 `sweeps/` 片段映射。仅更改顶层 `root_dir` 不会替换 reader 实际使用的 PKL：应修改 `v2xsim.info_files`。

当前协议只支持编号 `1..agent_num`、最多 5 个 agent，固定 Agent 1 为 ego，不能据此宣称支持 Agent 0 RSU 或六智能体协议。GT 来自距离筛选后参与 agent 的 world 标签，按全局 ID 去重并投影到 ego；当前投框函数不做点命中可见性过滤。不要无说明地替换为另一种可见性或多 ego 评估协议。

## 4. 生成独立、可追溯的运行配置

先填写本机真实路径。变量名只是本节生成配置时使用，训练入口不会自动读取这些环境变量。

```bash
export DAIR_ROOT=/workspace/dair-v2x-coalign/cooperative-vehicle-infrastructure
export OPV2V_ROOT=/data/opv2v
export V2XSIM_RAW_ROOT=/data/v2xsim/V2X-Sim-2.0
export V2XSIM_INFO_ROOT=/data/v2xsim/v2xsim2_info
```

下面生成三个正式配置和三个 1-epoch 配置。正式训练仍保留来源文件的 batch、epoch、lr；统一开启 ESSA、IoU 质量监督/校正和现有 late fusion。为规避 DDP 当前 `num_workers=0` 时仍传 `prefetch_factor=2` 的兼容性问题，独立配置将训练和验证 worker 都设为 1；这是数据加载设置，不改变模型。

```bash
python - <<'PY'
import copy
import os
from pathlib import Path
import yaml
from opencood.hypes_yaml.yaml_utils import load_yaml

sources = {
    'dair': 'opencood/hypes_yaml/dair-v2x/dair_where2comm_e3dsnn_sdtv3_essa_snnresneck_20261024.yaml',
    'opv2v': 'opencood/hypes_yaml/opv2v/opv2v_e3dsnn_where2comm_ddp_bs4_A_50epoch.yaml',
    'v2xsim': 'opencood/hypes_yaml/v2xsim/v2xsim_e3dsnn_where2comm_sdtv3_essa.yaml',
}
out = Path('run_configs')
out.mkdir(exist_ok=True)
dair_root = Path(os.environ['DAIR_ROOT'])
opv2v_root = Path(os.environ['OPV2V_ROOT'])
info_root = Path(os.environ['V2XSIM_INFO_ROOT'])

def plain(value):
    # 去除来源配置中的 numpy 专用 YAML 标签，保留普通列表和字典。
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    return value

for dataset, source in sources.items():
    h = plain(load_yaml(source))
    h['name'] = dataset + '_e3dsnn_essa_no_valuenet'
    h['yaml_parser'] = 'load_e3dsnn_where2comm_params'
    h['iou_quality_head'] = dict(enabled=True, loss_weight=1.0,
                                 quality_power=1.0, debug=False)
    h['model']['args']['backbone_fix'] = False
    h['model']['args']['fusion_args']['agg_operator'] = dict(
        mode='SDTV3_ESSA', feature_dim=256, num_heads=8,
        v_expansion=4, spike_max=4, spike_norm=4)
    h['loss']['supervision_mode'] = (
        'legacy_dair' if dataset == 'dair' else 'cooperative_only')
    h['loss']['include_round_losses'] = False  # 当前三套配置均为单轮通信
    h['train_params'].update(
        num_workers=1, train_num_workers=1, val_num_workers=1,
        persistent_workers=False, train_persistent_workers=False,
        val_persistent_workers=False, prefetch_factor=1,
        eval_freq=1, save_freq=1)

    if dataset == 'dair':
        h['data_dir'] = str(dair_root)
        h['root_dir'] = str(dair_root / 'train.json')
        h['validate_dir'] = str(dair_root / 'val.json')
        h['test_dir'] = h['validate_dir']  # 默认只报告 DAIR val
        h['vehicle_label_dir'] = str(dair_root / 'vehicle-side/label/lidar')
        h['cooperative_label_dir'] = str(dair_root / 'cooperative/label_world')
    elif dataset == 'opv2v':
        h['root_dir'] = str(opv2v_root / 'train')
        h['validate_dir'] = str(opv2v_root / 'validate')
        h['test_dir'] = str(opv2v_root / 'test')
    else:
        files = {s: str(info_root / ('v2xsim_infos_' + s + '.pkl'))
                 for s in ('train', 'val', 'test')}
        h['v2xsim'].update(raw_root=os.environ['V2XSIM_RAW_ROOT'],
                           info_root=str(info_root), info_files=files,
                           split='train', eval_split='val')
        h['root_dir'], h['validate_dir'], h['test_dir'] = (
            files['train'], files['val'], files['test'])

    path = out / (dataset + '_essa_no_valuenet.yaml')
    path.write_text(yaml.safe_dump(h, sort_keys=False), encoding='utf-8')
    # 重走正式 parser，使 IoU 配置和 anchor 参数传到所有消费者。
    h = plain(load_yaml(str(path)))
    path.write_text(yaml.safe_dump(h, sort_keys=False), encoding='utf-8')
    smoke = copy.deepcopy(h)
    smoke['name'] += '_smoke'
    smoke['train_params'].update(batch_size=1, epoches=1)
    smoke_path = out / (dataset + '_essa_no_valuenet_smoke.yaml')
    smoke_path.write_text(yaml.safe_dump(smoke, sort_keys=False), encoding='utf-8')
    print(path, 'mode=', h['loss']['supervision_mode'],
          'grid=', h['model']['args']['grid_size'],
          'quality=', h['postprocess']['iou_quality_head'])
PY
```

若使用独立补充标注目录，在生成后修改 DAIR 文件中的两项 label 路径；smoke 和正式配置都要同步。运行上述片段会重写这六个派生文件，后续人工调整应另存或记录，避免再次生成时丢失。

三个正式配置加载后均应满足：`backbone_fix=false`、增强 neck `enabled=true`、`agg_operator.mode=SDTV3_ESSA`、三个 IoU 消费位置配置一致。OPV2V 原配置只在 `loss.args` 写入 IoU 参数，原 parser 不会自动把它同步到后处理，因此本文显式更换 parser；不能仅看 loss 中 `enabled=true` 就认定评估已使用 IoU 分数校正。

来源 YAML 中虽然写有翻转、旋转、缩放配置，但当前这三条 intermediate reader 路径没有调用场景几何增强流程；不能仅凭 `data_augment` 字段声称训练执行了这些增强。V2X-Sim 当前 reader 也没有接入配置中的定位噪声注入。本文不补写未实现的数据流程。

## 5. 正式训练前的最小检查

依次在三个配置上检查数据读取、模型输出、总 loss 和反向传播；下面使用真实首帧，batch=1，不保存随机模型：

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.loss_utils import get_loss_mode, compute_training_loss

assert torch.cuda.is_available()
for name, hw in [('dair', (50,126)), ('opv2v', (48,176)), ('v2xsim', (40,40))]:
    h = load_yaml('run_configs/' + name + '_essa_no_valuenet.yaml')
    model = train_utils.create_model(h).cuda()
    criterion = train_utils.create_loss(h)
    mode = get_loss_mode(h)
    for training in (True, False):
        ds = build_dataset(h, visualize=False, train=training)
        assert len(ds) > 0
        batch = ds.collate_batch_train([ds[0]])
        assert batch is not None
        batch = train_utils.to_device(batch, torch.device('cuda'))
        batch['ego']['epoch'] = 0
        model.train(training)
        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            pred = model(batch['ego'])
            agents = int(batch['ego']['record_len'].sum().item())
            assert tuple(pred['psm'].shape) == (1, 2, *hw)
            assert tuple(pred['rm'].shape) == (1, 14, *hw)
            assert tuple(pred['iou'].shape) == (1, 2, *hw)
            assert pred['psm_single'].shape[0] == agents
            assert tuple(batch['ego']['label_dict']['pos_equal_one'].shape[1:3]) == hw
            loss = compute_training_loss(pred, batch, h, criterion, mode)
            assert torch.isfinite(loss).item()
            if training:
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all().item() for g in grads)
        print(name, 'train' if training else 'val', 'samples=', len(ds),
              'agents=', agents, 'mode=', mode, 'loss=', loss.item())
        del pred, loss, batch, ds
    del model, criterion
    torch.cuda.empty_cache()
print('三个数据集的首帧前向/训练反向检查完成。')
PY
```

首帧检查通过后，可先执行 1 个完整 epoch 检查 DataLoader、权重保存和验证循环。下例是 DAIR；将 `dair` 替换为 `opv2v` 或 `v2xsim`，分别完成检查。**smoke 配置没有截断数据集，仍遍历整个训练 split，只是 batch=1、epoch=1；它可能耗时较长。**

```bash
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
  --hypes_yaml run_configs/dair_essa_no_valuenet_smoke.yaml
```

首帧无法代表全部数据；重点排查后续帧的文件缺失、GT 超出 `max_num=100`、空点云及多 agent 情况。正式训练不用 smoke 的权重，也不传 smoke 的 `--model_dir`。

## 6. 三个数据集的正式训练命令

### 6.1 单卡训练

从仓库根目录运行；下面三条可以依次执行，分别创建独立日志目录：

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
  --hypes_yaml run_configs/dair_essa_no_valuenet.yaml \
  2>&1 | tee run_logs/dair_train.log

CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
  --hypes_yaml run_configs/opv2v_essa_no_valuenet.yaml \
  2>&1 | tee run_logs/opv2v_train.log

CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
  --hypes_yaml run_configs/v2xsim_essa_no_valuenet.yaml \
  2>&1 | tee run_logs/v2xsim_train.log
```

输出目录由 `train_utils.setup_train()` 自动创建：

```text
opencood/logs/<name>_YYYY_MM_DD_HH_MM_SS/
├── config.yaml
├── net_epoch1.pth
├── net_epoch2.pth
├── ...
├── events.out.tfevents...
└── scripts/                          # 部分源码快照
```

请以结束日志中的 `checkpoints saved to ...` 为准记录目录。`scripts/` 仅备份 models/data_utils/utils/loss，不含完整 tools 和 hypes_yaml；还应保存代码版本、本文派生配置、启动命令及环境清单。

`train.py` 的 `--hypes_yaml` 必填，没有 `--batch_size`、`--epochs`、`--output_dir` 或 `--amp` 参数。修改总轮数必须写 `train_params.epoches`，拼写就是 `epoches`。本入口没有启用 AMP 或梯度累积；需要降低显存时，先降低配置中的 `batch_size`，并记录实际总 batch。不要随意改变体素和范围来掩盖显存问题，那会改变检测网格及实验协议。

### 6.2 多卡 DDP

例如每个数据集使用两张 GPU，命令如下；同样应分别启动和记录：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  opencood/tools/train_ddp.py --hypes_yaml run_configs/dair_essa_no_valuenet.yaml

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  opencood/tools/train_ddp.py --hypes_yaml run_configs/opv2v_essa_no_valuenet.yaml

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  opencood/tools/train_ddp.py --hypes_yaml run_configs/v2xsim_essa_no_valuenet.yaml
```

DDP 的 `batch_size` 是 **每张卡的场景数**，全局 batch = 每卡 batch × 进程数。例如配置 4、两张卡时总 batch 为 8；若要维持与单卡 batch=4 相同的总 batch，应派生 DDP 配置并将每卡 batch 改成 2。代码不会随卡数自动缩放学习率，也不应把不同总 batch 的运行当作严格等价复现。

单卡入口的训练和验证 DataLoader 均 `drop_last=True`，验证还会 shuffle；DDP 验证按 rank 分配互不重复的样本、保留末尾不满 batch，并汇总按场景数加权的 loss。因此两入口的验证 loss 统计并不完全相同。正式 AP 用第 8 节的 batch=1 全量推理获得，不能用训练日志中的验证 loss 代替。

DDP 当前把 `num_workers=0` 对应的 `prefetch_factor` 写成 2；较新的 PyTorch 会拒绝该组合。本文生成的训练/验证 worker 都为 1。若调试时必须使用 worker=0，可改用 `train.py` 并将其 `num_workers` 设为 0；若坚持使用 DDP，则须先另行修正该 DataLoader 参数，不能只改 YAML 后宣称问题已解决。

## 7. 续训与旧权重初始化

### 7.1 继续同一个实验

例如某次 DAIR 已训练到 `net_epoch12.pth`，计划总共训练 30 epoch：

```bash
RUN=opencood/logs/实际的_dair_实验目录
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
  --hypes_yaml "$RUN/config.yaml" --model_dir "$RUN"
```

多卡续训只需将入口替换为 `torchrun ... opencood/tools/train_ddp.py`，保留上述两个参数。行为须注意：

1. 传入 `--model_dir` 后，实际加载的是该目录的 `config.yaml`，外部 YAML 不再决定训练参数。增加总轮数要修改这份保存配置中的 `train_params.epoches`。
2. `epoches` 表示目标总轮数，不是额外追加轮数。已到 30、目标仍为 30 时没有新训练；想再做 10 轮应设为 40。
3. 正常目录自动选择编号最大的 `net_epochN.pth`；训练 CLI 没有选择指定 resume epoch 的参数。
4. checkpoint 只有 `model.state_dict()`。续训恢复模型及 epoch 起点，重新创建 Adam，并根据 epoch 推进新 scheduler，**不恢复 Adam 动量、完整 scheduler/RNG/DataLoader 状态**，不是严格无缝恢复。学习率还受源码 `scheduler.step(epoch)` 调用方式影响，应记录每轮实际打印值，不凭配置将下降时点写死。
5. 目录若含 `net_latest.pth`，加载器会优先使用它，并把返回 epoch 写为 100；即使推理传 `--eval_epoch` 也优先走该分支。普通训练本身不生成这个文件。不要把外部同名文件混入本实验目录；需要精确选权重时使用第 8 节的独立评估目录。

### 7.2 从旧检测器初始化一个新实验（可选）

从头训练不需要这一步。若已有同数据集、同几何配置的旧 ATTEN 检测器，可初始化新的 ESSA 实验：

```bash
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
  --hypes_yaml run_configs/dair_essa_no_valuenet.yaml \
  --pretrained_path /path/to/compatible_old_run/net_epoch30.pth
```

该模式使用新 YAML、从 epoch 0 开始，新建优化器和日志目录。允许缺失的新 ESSA、IoU 头等命名空间会打印出来；它不是任意 checkpoint 的宽松加载接口，已有参数的形状不匹配或不允许的缺失/多余键仍会报错。`--pretrained_path` 与 `--model_dir` 互斥。

不建议把 DAIR/OPV2V 权重直接作为 V2X-Sim 权重使用：前两者 z 网格为 40，后者为 50，稀疏输出高度及 BEV 投影相关权重形状会变化。本流程默认分别从头训练，以免引入未经验证的跨数据集迁移。

## 8. 验证集选权重与最终评估

训练器只计算验证 loss、按频率保存权重，**不会自动保存 best AP 模型**。应预先固定选择标准，例如验证集 AP@0.7，评估候选 checkpoint 后选择；测试集只用于最终结果。

统一使用 `opencood/tools/inference_global_ap.py`，参数必须包含 `--count_empty_predictions`。该开关使没有预测框的帧仍计入 GT，避免漏算困难帧。该入口使用跨帧分数全局排序，输出的是 **旋转 BEV IoU 下的 AP@0.3/0.5/0.7**，不是 3D AP。

### 8.1 为指定 checkpoint 创建独立评估目录

不直接修改训练目录的配置。下例选择某次 OPV2V 实验第 50 轮的验证集；将 `RUN`、`EPOCH`、`SPLIT` 修改为实际值。此片段适用于三个数据集。

```bash
export RUN=opencood/logs/实际的_opv2v_实验目录
export EPOCH=50
export SPLIT=val                       # val 或 test
export EVAL_DIR="${RUN}/eval_${SPLIT}_ep${EPOCH}"
python - <<'PY'
import os
import shutil
from pathlib import Path
import yaml
from opencood.hypes_yaml.yaml_utils import load_yaml

run = Path(os.environ['RUN'])
epoch = int(os.environ['EPOCH'])
split = os.environ['SPLIT']
assert split in ('val', 'test')
checkpoint = run / ('net_epoch%d.pth' % epoch)
assert checkpoint.is_file(), checkpoint
h = load_yaml(str(run / 'config.yaml'))
if 'v2xsim' in h:
    h['v2xsim']['eval_split'] = split
    selected = h['v2xsim']['info_files'][split]
else:
    selected = h['validate_dir'] if split == 'val' else h['test_dir']
# inference_global_ap.py 会执行 validate_dir = test_dir，故两者都设为目标 split。
h['validate_dir'] = selected
h['test_dir'] = selected
target = Path(os.environ['EVAL_DIR'])
target.mkdir(parents=True, exist_ok=False)
with (target / 'config.yaml').open('w', encoding='utf-8') as f:
    yaml.dump(h, f, sort_keys=False)
shutil.copy2(checkpoint, target / checkpoint.name)
print('评估目录:', target)
print('实际数据:', selected)
print('V2X-Sim eval_split:', h.get('v2xsim', {}).get('eval_split'))
PY

CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_global_ap.py \
  --model_dir "$EVAL_DIR" --eval_epoch "$EPOCH" \
  --fusion_method intermediate_with_comm \
  --save_vis_n 0 --count_empty_predictions
```

每次候选 epoch 使用不同目录；相同目录已存在时生成片段会停止，避免覆盖先前结果。权重仅复制指定文件，避免 `net_latest.pth` 干扰选择。V2X-Sim reader 按 `v2xsim.eval_split` 和 `info_files` 读取数据，所以只改 `test_dir` 不足以切换 val/test。

选择完成后，把 `EPOCH` 固定为验证集选出的编号，`SPLIT=test`，重新执行整段目录生成及推理命令。对于本文默认 DAIR 配置，test 路径仍是 val，不能将其改名为测试集结果；要使用独立测试数据，先明确有对应标注的测试 JSON，再在评估配置中指定。

### 8.2 理解本次评估包含的内容

`intermediate_with_comm` 表示调用当前中间特征融合模型，并记录 `comm_rate`，不会自动关闭后处理的 late fusion。本文三套派生配置均启用：

```yaml
iou_quality_head:
  enabled: true
  loss_weight: 1.0
  quality_power: 1.0
postprocess:
  confidence_aware_late_fusion:
    enabled: true
    collaborator_score_weight: 0.55
    score_threshold: 0.30
  target_args:
    score_threshold: 0.2
  nms_thresh: 0.15
```

这里是配置片段，不是完整 YAML；不要用它替换整个 `postprocess`。IoU 质量目标参与训练，质量分数参与推理排序；当前 power 是 1.0，不能照抄旧文档中的 4。协作者原始分数门限 0.30、权重 0.55、最终候选门限 0.2 与通信阈值 0.05 是不同参数。

如果研究只传中间特征的通信效果，应另建评估目录，将 `postprocess.confidence_aware_late_fusion.enabled` 设为 false，再报告对照。`--fusion_method late` 不是关闭这个开关的方式，也不是本文完整检测器的标准评估入口。

当前 `comm_rate` 统计非 ego agent 的选中网格位置比例；单 agent 场景为 0。此推理脚本在 batch=1 下对各帧比例取平均。该指标不是带宽、字节数、脉冲稀疏率或端到端通信时延，且没有计入 late fusion 所需的邻居检测框传输。

### 8.3 结果文件与通信阈值对照

每个评估目录中保存：

| 文件 | 内容 |
|---|---|
| `config.yaml`、指定的 `net_epochN.pth` | 本次评估配置和权重 |
| `eval_global.yaml` | AP 与部分 PR 曲线数组；重复推理会覆盖 |
| `result_global.txt` | epoch、AP、comm_rate；追加写入 |
| `vis_bev/`、`vis_3d/` | 仅在 `--save_vis_n` 大于 0 时生成 |
| `npy/` | 仅在 `--save_npy` 时保存预测、GT 和点云 |

需要图示时把 `--save_vis_n 0` 改为 10。可视化代码通过路径是否含大写 `OPV2V` 来判断左右手坐标，不能仅凭默认图形方向判定数值评估有误；定量评估可保持关闭可视化。

可用 `--comm_thre 0.1` 等参数评估训练后不同通信阈值，但应在验证集固定比较方案，并为每个阈值另建目录、保存完整命令。CLI 覆盖不会写回 `config.yaml`。训练阈值 0.05 与推理阈值变化应分别记录；不能将阈值对照称为 ValueNet Top-K 实验。

## 9. 运行失败时按实际入口定位

| 现象 | 应检查的内容 |
|---|---|
| `Incomplete legacy DAIR auxiliary supervision` | OPV2V/V2X-Sim 的生效配置是否为 `loss.supervision_mode: cooperative_only`；续训看日志目录配置 |
| `prefetch_factor ... num_workers=0` | DDP 的训练和验证 worker 都应大于 0，或另行修正源码后使用 0 |
| spconv/cumm、`box_overlaps`、`iou3d_nms_cuda` 导入失败 | 当前解释器、扩展编译环境、PyTorch/CUDA ABI 是否一致 |
| V2X-Sim PKL 存在但仍找不到文件 | `v2xsim.info_files`、`raw_root`、`legacy_root` 及索引中的点云路径 |
| V2X-Sim 仍在评估 val | `v2xsim.eval_split`；不能只改顶层目录 |
| BEV/anchor 尺寸不一致 | 范围、voxel、feature_stride、model/fusion 配置及 parser 是否一致 |
| ESSA 权重缺失或 mismatch | checkpoint 与融合模式是否配套；初始化用 `pretrained_path`，恢复用 `model_dir` |
| 一启动就结束 | 已保存 epoch 是否达到 `epoches`；是否混入返回 epoch=100 的 `net_latest.pth` |
| 显存不足 | 每卡 batch、实际 agent 数、输入范围；先减 batch，不擅自更改几何协议 |
| loss 日志看起来不是三路总和 | `criterion.logging()` 记录的是 criterion 内部最近一次调用；DAIR train.py 的逐步日志不能直接当作三路 loss 总和 |
| AP 异常偏高或不同脚本差距很大 | split、空预测帧是否计入、全局排序、late fusion、IoU 校正和阈值是否一致 |

不要使用 `train.py --run_test` 串接评估：该选项仍拼接历史硬编码绝对路径 `/GPFS/data/yhu/code/OpenCOOD/...`。训练结束后显式执行第 8 节命令。

## 10. 每次实验应留下的记录

至少保存数据集及 split 的实际路径/版本、标注来源、参与 agent/ego 规则、代码版本或源码归档、环境清单、配置、启动命令、GPU 数、每卡/总 batch、训练日志、选择 checkpoint 的验证指标、最终评估配置与结果。DAIR 的验证划分复用、OPV2V 的 ESSA 派生配置、V2X-Sim 的固定 Agent 1 协议都应在实验报告中明确。

完成标准是：三个数据集分别通过数据和梯度检查，产生各自训练 checkpoint，依据验证集选择权重，在明确划分及固定后处理设置下保存 AP 和通信率结果。ValueNet 相关工具不属于这条完成标准。

## 11. 关键源码依据

- 配置解析与 IoU 同步：[yaml_utils.py](../opencood/hypes_yaml/yaml_utils.py)。
- 单卡/DDP 参数、训练及保存行为：[train.py](../opencood/tools/train.py)、[train_ddp.py](../opencood/tools/train_ddp.py)。
- 权重初始化、恢复与日志目录：[train_utils.py](../opencood/tools/train_utils.py)。
- 三种监督模式及标签要求：[loss_utils.py](../opencood/tools/loss_utils.py)。
- DAIR 数据契约：[intermediate_fusion_dataset_dair.py](../opencood/data_utils/datasets/intermediate_fusion_dataset_dair.py)。
- OPV2V 数据及 ego 顺序：[intermediate_fusion_dataset_opv2v_e3dsnn.py](../opencood/data_utils/datasets/intermediate_fusion_dataset_opv2v_e3dsnn.py)、[opv2v/basedataset.py](../opencood/data_utils/datasets/opv2v/basedataset.py)。
- V2X-Sim schema、split 与几何：[v2xsim/basedataset.py](../opencood/data_utils/datasets/v2xsim/basedataset.py)、[intermediate_fusion_dataset_v2xsim_e3dsnn.py](../opencood/data_utils/datasets/intermediate_fusion_dataset_v2xsim_e3dsnn.py)。
- 检测器和脉冲 neck：[point_pillar_where2comm_e3dsnn.py](../opencood/models/point_pillar_where2comm_e3dsnn.py)、[e3dsnn_multiscale_enhancement_neck.py](../opencood/models/sub_modules/e3dsnn_multiscale_enhancement_neck.py)。
- 通信与融合：[where2comm.py](../opencood/models/comm_modules/where2comm.py)、[where2comm_attn.py](../opencood/models/fuse_modules/where2comm_attn.py)、[sdtv3_attention_fusion.py](../opencood/models/fuse_modules/sdtv3_attention_fusion.py)。
- 质量监督和后处理：[point_pillar_loss.py](../opencood/loss/point_pillar_loss.py)、[voxel_postprocessor.py](../opencood/data_utils/post_processor/voxel_postprocessor.py)。
- 评估 split、空预测帧和全局排序：[inference_global_ap.py](../opencood/tools/inference_global_ap.py)、[eval_utils_global.py](../opencood/utils/eval_utils_global.py)。
