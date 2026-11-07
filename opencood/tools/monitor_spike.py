import torch


def print_value_bins(name, outputs):
    if isinstance(outputs, torch.Tensor):
        outputs = [outputs]

    with torch.no_grad():
        for i, output in enumerate(outputs):
            # 1. 模拟 MultiSpike 内部的 Quant 离散化
            clamped = torch.clamp(output.detach(), min=0, max=4)
            rounded = torch.round(clamped)
            
            # 2. 模拟除以 4 的缩放操作，得到标准的脉冲比例值
            scaled = rounded / 4.0
            
            # 3. 设置极小的浮点数容差，精确匹配离散点
            eps = 0.01
            
            count_0    = ((scaled >= 0.00 - eps) & (scaled <= 0.00 + eps)).sum().item()
            count_025  = ((scaled >= 0.25 - eps) & (scaled <= 0.25 + eps)).sum().item()
            count_05   = ((scaled >= 0.50 - eps) & (scaled <= 0.50 + eps)).sum().item()
            count_075  = ((scaled >= 0.75 - eps) & (scaled <= 0.75 + eps)).sum().item()
            count_1    = ((scaled >= 1.00 - eps) & (scaled <= 1.00 + eps)).sum().item()
            avg_value = scaled.mean().item()

            print(
                f"{name}[{i}] shape={tuple(scaled.shape)} "
                f"0={count_0} 0.25={count_025} 0.5={count_05} "
                f"0.75={count_075} 1={count_1} avg={avg_value:.6f}"
            )
