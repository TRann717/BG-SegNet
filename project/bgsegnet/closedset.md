# BG-SegNet 语义分割方案说明（BG-SegNet Segmentation）


## Innovation

- 边界增强（数学定义可直接引用）
  - 软边界目标（类无关或类感知）：
    - 先做形态学梯度得到硬边界 `B`，再做距离变换 `d(x)` 得到像素到最近边界的距离。
    - 定义软目标：`s(x) = exp( - d(x)^2 / (2 * sigma^2) )`，当 `|d(x)| > band_pixels` 置 `s(x)=0`；类感知版本对每类分别计算。
  - 统一软边界损失（单通道示意）：
    - Focal: `L_f = E_band[ alpha * (1 - p)^gamma * ( - y * log(p) ) + (1 - alpha) * p^gamma * ( - (1 - y) * log(1 - p) ) ]`
    - BCE: `L_b = E_band[ BCE(logits, y) ]`
    - L1: `L_1 = E_band[ | p - y | ]`
    - 梯度对齐: `L_g = E_band[ 1 - cos(grad(p), grad(y)) ]`
    - 多尺度一致性: `L_ms = E_band[ | sigmoid(b4) - up(sigmoid(b8)) | ]`
    - 汇总: `L_bd = w_f * L_f + w_b * L_b + w_l1 * L_1 + w_g * L_g + w_ms * L_ms`
  - 边界门控精炼：`Delta = Refine(feats * g)`，不确定性 `u` 抑制；输出 `out = feats + alpha(x) * (1 - u(x)) * Delta`。 
    - 自适应 `alpha(x) = alpha_min + (alpha_max - alpha_min) * g(x)^gamma`
    - OABG：4 个深度可分离 3x3 基核（H/V/D1/D2），方向权 `w_k(x) = | dot(tangent(x), k) | / Z`，`y = sum_k w_k * DW_k(Delta)` 后经 `1x1` 混合。

- 语义主干与解码稳定性
  - 冻结 `SAM2.1` 骨干，统一预处理为像素域 0..255 标准化，输出规范尺度 `F4/F8/F16`。
  - `UPerNet`（PPM+FPN，GroupNorm）在 1/4 生成融合特征，适配小 batch。
  - `DecoderHead` 支持 `bilinear/CARAFE`，可选 GUM（以边界强度为 guide）改善 1× 边界细节。

- 训练日程（简明公式）
  - Warmup：`lr_t = base_lr * warmup_factor(t)`（`t < warmup_iters` 线性上升）
  - 边界升温：`w_bd(epoch) = w_bd_max * min(1, epoch / bd_warmup_epochs)`
  - 渐进式门控：early detach(g)，在固定 epoch 或性能停滞后释放梯度（可选）。

## Architecture

- 前向（最短路径视图）
  - `x` → 增广（不归一化）→ `SAM2FeatureExtractor`（0..255 标准化）→ `F4/F8/F16`
  - `F4/F8/F16` → `UPerNet` → `NeckOut@1/4`
  - 若开边界：`BoundaryHead` 得 `b4/b8/dir`，`g = sigmoid(b)`；`BoundaryGate`: `out = feats + alpha(x) * (1 - u) * Refine(feats * g)`
  - `DecoderHead` → 闭集 `logits@1x`；可选 GUM 用 `guide = up(g)` 精炼

- 损失（训练）
  - `L_seg = w_ce * CE + w_dice * Dice/Tversky + w_aux * CE_aux`
  - `L_bd =` 统一软边界 + 方向一致性 + 几何边带（Triplet/CurvAware，按需开启）
  - 总损失：`L = L_seg + w_bd(epoch) * L_bd`

- 指标
  - `mIoU/mAcc/OverallAcc/FWIoU`；若开启边界分支，额外 `BoundaryIoU`（bIoU）。

## Block detail

- 数据与增广（`bgsegnet/src/data/dataset_yolo.py`）
  - YOLO 分割标签（多边形）→ 语义掩码：标签首列 `class_id` 从 0 开始，本项目读取时将前景记为 `class_id+1`，背景为 0。
  - 训练增广：随机裁剪、翻转、色抖、轻模糊；不做 Normalize（预处理统一由特征提取器完成）。
  - Loader：`pin_memory/prefetch_factor/persistent_workers` 与 `worker_init_fn`（固定随机种子）。

- SAM2 特征（`bgsegnet/src/models/sam2_loader.py`）
  - `load_sam2_model(sam2_path)` 通过 Hydra 组装 SAM2.1 Hiera B+ 并从 `sam2.1_hiera_base_plus.pt` 读权重。
  - `SAM2FeatureExtractor`：标准化到 0..255，输出 `F4/F8/F16`；若缺尺度用插值补齐；用 `img_size` 自检通道数。

- 颈部与头（`bgsegnet/src/models/neck/ppm.py`, `heads/*`）
  - `UPerNet`：PPM(F16) + FPN（lat/smooth/fuse）在 1/4 输出；可返回金字塔以做多尺度边界监督。
  - `DecoderHead`：3x3 + GN + ReLU + Dropout → 上采样到 1× → 1x1 分类；`init_bias_with_priors()` 用像素先验稳定早期训练。
  - `BoundaryHead`：类无关/类感知边界与可选方向场；`generate_soft_boundary_*` 提供稳定软目标（距离裁剪+float64 指数）。
  - `BoundaryGate`：门控 `g`、自适应 `alpha(x)`、OABG、SE、U-aware 抑制与 2 类类混合（前/背景）。

- 损失（`bgsegnet/src/losses/seg_losses.py`）
  - CE（支持类权）+ Dice/Tversky（可忽略背景）+ Aux CE。
  - 统一软边界：`L_bd` 详见公式；类感知版本逐类计算并平均；方向一致性用边界梯度方向为 GT；多尺度一致性比较 b4 与上采样 b8。
  - 几何边带：Triplet（边带/内/外）与 CurvAware（切线去抖 + 法线锐化）。

- 训练器与日志（`bgsegnet/src/engine/trainer.py`, `src/utils/logger.py`）
  - 优化器：AdamW/SGD；边界参数组 `0.5x LR`；Cosine/Poly 调度；AMP + 梯度累积 + 裁剪。
  - 边界升温与门控渐进解冻；保存 `last/best`；TensorBoard 标量/样例、CSV 日志。

## 配置要点（`bgsegnet/config.yaml`）

- data：
  - `dataset_root`：YOLO 分割目录，需有 `{train,val}/images` 与 `{train,val}/labels`。示例：`/path/to/yolo_seg`。
  - `num_classes`：包含背景的类别数（示例：2）。
  - `img_size`、`aug.crop_size`：输入与训练裁剪尺寸；`class_freq` 可选，用于 bias 初始化。

- model：
  - `sam2_path`：目录，内必须存在 `sam2.1_hiera_base_plus.pt`（SAM2.1 Hiera B+ 权重）。示例：`/path/to/sam2_dir`。
  - `freeze_backbone`：建议 `true`；`neck/head` 参数与 `boundary.*`（启用、权重、类感知、方向、多尺度、OABG、自适应 α、门控策略）。

- train：
  - `epochs/batch_size/accum_steps/lr/weight_decay/scheduler/warmup_iters/amp/grad_clip/val_freq/seed`。
  - `bd_warmup_epochs`：边界损失升温到 `model.boundary.weight` 的 epoch 数。

- log：`outdir`、TensorBoard（间隔与样例图）与 CSV 写出开关。

必备路径汇总：
- 数据集：`data.dataset_root=/path/to/yolo_seg`
- SAM2 权重目录：`model.sam2_path=/path/to/sam2_dir` 且该目录内存在 `sam2.1_hiera_base_plus.pt`

## 训练与复现

- 训练（：
  ```bash
  python -m bgsegnet.train --config ./bgsegnet/config.yaml
  ```
  - 断点：`--resume runs/<expN>/weights/last.pth`
  - 仅验证：`python -m bgsegnet.train --config ./bgsegnet/config.yaml --eval`

- 快速步骤：
  - 准备数据为 YOLO 分割格式（见上）。
  - 下载 SAM2.1 Hiera B+ 权重为 `sam2.1_hiera_base_plus.pt`，放入 `model.sam2_path` 指定目录。
  - 修改 `bgsegnet/config.yaml` 中三处：`data.dataset_root`、`model.sam2_path`、`experiment` 名称。
  - 运行训练命令；日志与权重输出在 `runs/<experiment>`（自动递增 `exp2/exp3...`）。

---