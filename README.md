# Simple MMSeg：简单数据接口，保留算法配置

这是基于 MMSegmentation 后端的本地二分类分割项目，适合裂缝、缺陷等 `image + 黑白 mask` 任务。你只需维护图片、mask 和一个简短 JSON 实验配置，无需编写 Dataset 类或 OpenMMLab 数据 pipeline。

保留 MMSegmentation **v1.2.2** 的原始配置目录、继承关系和许可证；模型结构与算法专用优化器设置沿用上游。训练数据、数据增强、二分类输出、训练步数、学习率调度、评估和保存策略由本项目统一管理。它并非删除 MMCV 后重新实现所有算法。

## 快速开始

先安装与操作系统对应的 Miniconda/Anaconda，再在项目根目录运行对应入口。脚本自动创建项目独立 Python 3.10 环境，安装全部 15 个实验和 TensorBoard，检查编译算子和模型注册、获取官方配置，并准备 7:2:1 数据划分。

**Linux / WSL2（Bash）：**

```bash
bash scripts/setup.sh
conda activate ./.venv-mmseg
```

**Windows（Conda PowerShell）：**

```powershell
# 一条命令完成环境和数据配置；不会自动开始训练
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
conda activate ./.venv-mmseg
```

安装后，两种系统都可以检查并启动双卡独立模型队列：

```bash
python scripts/check_experiments.py --level config
python scripts/train_gpu_queue.py --gpu-ids 0,1
```

另开终端，激活同一环境后启动可视化：

```bash
python scripts/webui.py --logdir work_dirs --port 6006
```

执行后打开 **http://127.0.0.1:6006**。训练开始写入日志后，界面会显示 loss、学习率和验证指标曲线，并支持比较多个实验。

详细说明：[安装](#1-安装一次) · [数据准备](#2-数据就是两个目录) · [算法与增强](#3-选择算法) · [损失与优化器](#当前实验的损失函数与优化器) · [日志与 WebUI](#训练日志与-tensorboard-webui)。

## 1. 安装一次

推荐 Conda + Python 3.10。Linux/WSL2 使用 `setup.sh`，Windows 使用 `setup.ps1`；`powershell -ExecutionPolicy ...` 不适用于 Linux。两者均使用独立环境，已有目标环境时检查 Python 版本后继续安装，不修改其他环境。

Linux/WSL2：

```bash
bash scripts/setup.sh
# 无需提前 conda activate；脚本通过 conda run 在目标环境中执行所有安装步骤
conda activate ./.venv-mmseg
python -m simple_seg --help
```

WSL2 需要在 Linux 发行版内安装 Linux 版 Conda，环境不能与 Windows 共用；已有 Windows `.venv-mmseg` 时，为 Linux 指定另一个环境目录。

```bash
# 尚未放入数据，只配置环境
bash scripts/setup.sh --skip-data
# 仅预览，不下载、不创建文件
bash scripts/setup.sh --print-only
# 指定 Linux 环境目录，路径支持空格
bash scripts/setup.sh --env-path "$HOME/conda-envs/simple-mmseg"
conda activate "$HOME/conda-envs/simple-mmseg"
```

Windows Conda PowerShell：

```powershell
.\scripts\setup.ps1
& .\.venv-mmseg\python.exe -m simple_seg --help
```

固定组合：PyTorch 2.1.0、torchvision 0.16.0、CUDA 11.8 版 PyTorch、MMCV 2.1.0、MMSegmentation 1.2.2、NumPy 1.26.4。CUDA 版需要支持该运行时的 NVIDIA 驱动；无需单独编译 CUDA 工具链。脚本要求 MMCV 有可用 wheel，找不到会停止并显示错误。

脚本默认使用项目目录的 `.venv-mmseg`。安装日志保存在 `work_dirs/setup/`；命令失败立即停止，可修复后重新运行。已有完整 7:2:1 划分时复用，不重新随机划分；已有其他比例或不完整划分时会报错，避免无意覆盖。获取官方配置后还会核对 15 个实验的数据流程。依赖版本依据 [MMSegmentation 兼容检查](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/__init__.py) 和 [MMDetection 兼容检查](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/__init__.py)，环境通过 Conda 的 [create --prefix](https://docs.conda.io/projects/conda/en/stable/commands/create.html) 创建，安装通过 [conda run --prefix](https://docs.conda.io/projects/conda/en/stable/commands/run.html) 执行。

```powershell
# 尚未放入 dataset/Image 和 dataset/Mask，只配置环境
.\scripts\setup.ps1 -SkipData
# 仅预览步骤，不安装或创建文件
.\scripts\setup.ps1 -PrintOnly
# 指定独立环境位置
.\scripts\setup.ps1 -EnvPath F:\conda-envs\simple-mmseg
```

依赖文件按用途划分：

| 文件 | 包含内容 | 适用范围 |
| --- | --- | --- |
| [`requirements.txt`](requirements.txt) | CUDA 11.8 PyTorch、MMCV/MMEngine/MMSeg、MMPRETRAIN、MMDetection 和 TensorBoard | 全部 15 个实验、评估、预测和 WebUI |
| [`requirements-common.txt`](requirements-common.txt) | 共用模型及工具依赖，不选择 PyTorch CUDA/CPU 版本 | 安装脚本内部复用，不作为新环境的独立安装入口 |
| [`requirements-models.txt`](requirements-models.txt) | 转发到 requirements.txt | 保持旧安装命令兼容 |

`requirements.txt` 面向 CUDA 11.8 版 PyTorch。Linux/WSL2 的一键入口是 `bash scripts/setup.sh`，Windows 的入口是 `powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1`。默认一键安装已包括 ConvNeXt/Mask2Former 所需的额外模型包。

可选 CPU 环境：Linux/WSL2 用 `bash scripts/setup.sh --cpu`，Windows 用 `./scripts/setup.ps1 -Cpu`。不要将 CUDA requirements 直接用于 CPU 安装。两者同样需要官方提供对应 MMCV wheel；找不到时停止，不自动源码编译。安装模型依赖不保证所有模型或算子支持 CPU 训练。

手动安装（已激活 Python 3.10 环境）：

也可以使用项目根目录的 `requirements.txt` 一次安装 CUDA 11.8 版依赖，然后注册项目命令：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

下面是等价的分步安装方式：

```powershell
python -m pip install numpy==1.26.4
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install mmcv==2.1.0 --only-binary mmcv -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
python -m pip install -e '.[models]'
```

后续示例中的 `python` 指这个独立环境中的 Python。可以先 `conda activate ./.venv-mmseg`；不激活时，Linux 用 `./.venv-mmseg/bin/python`，Windows PowerShell 用 `& .\.venv-mmseg\python.exe` 替换。单模型多 GPU 的 NCCL DDP 需要 Linux/WSL2；一张卡一个模型的任务队列在 Windows 和 Linux 均可使用。

## 2. 数据就是两个目录

```text
项目目录/
  dataset/
    Image/
      sample_001.jpg
      sample_002.jpg
    Mask/
      sample_001.png
      sample_002.png
    splits/               # prepare 自动生成
```

- 按相对路径去掉扩展名配对，支持子目录；图片和 mask 扩展名可以不同。
- 黑色 `0` 为背景；白色 `255` 为前景，也支持 `0/1` mask。
- **源 mask 的 255 会转换为标签 1，不会被误当成忽略区域。** 训练 padding 才使用忽略标签 255。
- 图片和 mask 尺寸必须一致。彩色类别图、灰度软标签、重复 stem 和缺失配对会直接报错。
- 图片按 RGB/BGR 处理方式交给上游预处理器；保留各模型的 mean/std。

检查全部文件，并按 70%/20%/10% 划分，不复制或改写原始数据：

```powershell
python -m simple_seg prepare
# 自定义目录和比例
python -m simple_seg prepare --root dataset --images Image --masks Mask --val-ratio 0.2 --test-ratio 0.1
```

所有实验默认从 `dataset` 读取数据，共用 `dataset/splits/train.json`、`val.json`、`test.json`，划分摘要保存在同目录的 `summary.json`。验证集和测试集数量分别向下取整（各至少一张），剩余样本归训练集。固定 seed，划分可复现；已有划分会拒绝覆盖，需要显式 `--force`。当前实现按单张图片随机划分，如果同一原图裁出多个 patch 或同一视频有相邻帧，请自行按原图/序列分组编辑这三个 JSON，确保它们不跨集合。manifest 每项只含 `id`、相对数据根目录的 `image`、`mask`。

## 3. 选择算法

当前数据共 **6042 对**，训练/验证/测试分别为 **4230/1208/604**。全部 13 个实验的训练 `batch_size=12`（每卡），梯度累积为 1；val/test 继续每卡 1 张，以适应不同比例和尺寸的图片。

训练上限统一为 **30000 iter**，验证及保存间隔均为 **265 iter**；每次验证保留快照，按 mIoU 保存 best，继续使用 **5000 iter** 无有效提升早停。单卡每次完整遍历约 `4230/12=352.5 iter`，因此上限相当于约 **85 轮**，作为当前从头训练的初始预算，不代表已经测得最佳收敛轮数。官方 InfiniteSampler 持续采样，这里的“轮数”仅是累计样本量的换算。

双卡时全局 batch 为 24，30000 iter 相当于约 **170 轮**。如果希望双卡也约 100 轮，可给单次或批量训练加 `--max-iters 17625`；此参数同步用于生成学习率调度。验证/保存间隔仍为 265 iter，约对应双卡的两轮，早停仍按 5000 iter 计算。换算公式为 `等效轮数 = iter × 每卡 batch_size × GPU 数 / 训练样本数`。梯度累积增加每次优化更新所包含的样本数，不会增加每个 dataloader iter 读取的样本数。

```bash
# 双卡、全局 batch=24，约 100 轮上限
python scripts/train.py configs/experiments/segformer_b0.json --gpus 2 --max-iters 17625
# 所有模型采用相同的双卡训练预算
python scripts/batch_train.py configs/batch/extended.json --gpus 2 --max-iters 17625
```

每卡 16 是按当前实验要求统一设置，尚未验证其显存占用；Mask2Former、较大的 backbone 和 DeepLabV3+ 的官方大裁剪可能超出显存。上游学习率及 AMP 设置保持原样，没有随 batch 自动放大学习率。若出现 OOM，需要按实际显存减小每卡 batch；必要时可使用梯度累积，并重新按实际 batch 换算训练 iter。

上游配置已下载到 `third_party/mmsegmentation/configs/`，含 53 个算法目录、681 个 Python 配置（含部分基础变体和 TTA 配置，数量不等于算法数）。重新获取或在新机器初始化：

```powershell
python -m simple_seg fetch-configs
python -m simple_seg models segformer
python -m simple_seg models deeplabv3plus
# 离线：使用官方 v1.2.2 源码 ZIP
python -m simple_seg fetch-configs --archive path/to/mmsegmentation-1.2.2.zip
```

当前有 13 个实验模板：

| 文件 | 算法 |
| --- | --- |
| `configs/experiments/segformer_b0.json` | SegFormer MiT-B0 |
| `configs/experiments/deeplabv3plus_r50.json` | DeepLabV3+ ResNet-50 |
| `configs/experiments/unet.json` | U-Net + FCN head |
| `configs/experiments/convnext_tiny.json`、`convnext_small.json` | ConvNeXt Tiny/Small + UPerNet |
| `configs/experiments/mask2former_r50.json`、`mask2former_swin_tiny.json` | Mask2Former + ResNet-50 / Swin-Tiny |
| `configs/experiments/swin_tiny.json`、`swin_small.json` | Swin Tiny/Small + UPerNet |
| `configs/experiments/twins_svt_small.json`、`twins_pcpvt_small.json` | Twins SVT-S / PCPVT-S + UPerNet |
| `configs/experiments/segnext_tiny.json`、`segnext_small.json` | SegNeXt MSCAN Tiny/Small + LightHamHead |
| `configs/experiments/deeplabv3plus_mobilenetv2.json` | 轻量 DeepLabV3+ MobileNetV2 |
| `configs/experiments/twins_svt_small_fpn.json` | 轻量 Twins SVT-S + FPN |

复制一个 JSON 并修改 `model_config`，即可接入其他上游配置。路径相对项目根目录；所有模板默认继承官方训练裁剪尺寸、增强、验证缩放和推理窗口。可选 `size` 顺序为 **[高, 宽]**，仅显式覆盖训练 RandomCrop 和训练 padding，不改变验证缩放、推理窗口或步长。默认 `num_workers: 0` 便于 Windows 使用。`max_iters` 是训练迭代数，不是 epoch。mask 使用原生分割变换的最近邻插值，训练不足 crop 大小的部分由预处理器 padding，标签填充值为忽略值 255。

ConvNeXt 需要 `mmpretrain==1.2.0`；Mask2Former 需要 `mmdet==3.3.0` 和支持 Deformable Attention / point_sample 的完整 MMCV。安装额外模型依赖：

```bash
python -m pip install -r requirements-models.txt
# 若已经按 requirements.txt 配好基础环境，也可安装项目扩展依赖
python -m pip install -e '.[models]'
```

配置解析与 `--dry-run` 可以在尚未安装这些可选模型包时使用；真正构建 Runner 前会显式导入注册模块，缺少组件时会报错。Mask2Former 使用官方 MMSeg 语义分割头直接接收黑白语义 mask，不需要另建 COCO instance 标注。它的二分类适配保留 `out_channels=256` 的 mask 特征维度，并把分类损失权重改成 `[1, 1, 0.1]`（背景、前景、无目标）；不是把所有通道都改成 2。

### 原生数据增强与训练技巧

所有现有实验模板使用 `"augmentation": {"mode": "upstream"}`，省略 augmentation 也继承官方配置。训练、验证、测试各自读取其上游 dataloader 的 pipeline，仅将 `LoadAnnotations` 替换为黑白二分类 mask 加载器。原生空间变换、变换顺序及参数完整保留；独立预测使用官方 test pipeline，去除标签加载。

大多数 ADE20K 实验保留 `RandomResize → RandomCrop → RandomFlip → PhotoMetricDistortion`；Mask2Former 保留其官方 `RandomChoiceResize + ResizeShortestEdge`。ConvNeXt 的滑窗 stride 保留 `(341,341)`，不会自动改成窗口的一半。U-Net 模板继承 DRIVE 配置，训练 crop 为 **64×64**，验证 scale 为 `(584,565)`；DeepLabV3+ 继承 Cityscapes，训练 crop 为 `(512,1024)`，验证 scale 为 `(2048,1024)`。这些是所选官方配置的参数，不是针对当前裂缝数据重新调参的结果。

### 官方数据流程核对（MMSegmentation v1.2.2）

13 个实验的上游配置与当前配置逐项对照见 [`reports/data_pipeline_audit.json`](reports/data_pipeline_audit.json)，包含训练/验证 pipeline、推理设置和预处理器。此报告为当前配置快照，不代表 GPU 训练验证。

官方将三个尺寸独立配置：训练 `RandomCrop.crop_size` 为 `[高, 宽]`；验证 `Resize.scale` 为 `[宽, 高]`；滑窗模式另有 `model.test_cfg.crop_size` 和 `stride`。例如 ADE20K 基础配置训练随机缩放后裁 512×512，验证按 `(2048,512)` 保持宽高比缩放整图，而非强制所有图片变成 512×512。Mask2Former 的上游训练还使用 `RandomChoiceResize + ResizeShortestEdge`，并非同一套 `RandomResize`。

**当前实现已移除这些统一覆盖：** 验证/测试使用各自官方 Resize 参数和 `keep_ratio`；`model.test_cfg` 的 whole/slide、crop_size 和 stride 保持原样；预处理器的训练 padding 与测试 `test_cfg` 独立保留。对于正方形 896×896 输入，ADE20K 验证 `(2048,512)` 仍会将其缩小至 512×512，这是该官方配置的行为；Cityscapes 验证 `(2048,1024)` 则会得到 1024×1024。沿用官方配置不意味着验证一定使用原始分辨率，也不能保证缩小时保留全部细裂缝。

官方验证在图片缩放之后加载原始 mask。推理后的 logits 先去除 padding，再插值回 `ori_shape`，最后生成类别标签；`IoUMetric` 累计整个数据集的交并集等计数后计算指标。恢复到原尺寸不会恢复输入缩小时丢失的细节。参考 [后处理源码](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/models/segmentors/base.py)、[整图/滑窗源码](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/models/segmentors/encoder_decoder.py)、[指标源码](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/evaluation/metrics/iou_metric.py)。

训练预处理器的 `size` 用于 padding，并不负责把验证图片 resize 成训练尺寸；验证 padding 由预处理器独立的 `test_cfg` 控制。随机裁剪本身也不会缩放目标。训练与验证的空间大小可以不同；是否保留原始分辨率、采用何种缩放范围，需要作为明确的实验协议决定，不能仅凭官方使用随机缩放就断言不会丢失细裂缝，也不能断言原图验证一定更好。

以下为**可选的自定义增强示例**，显式选择它才会替换官方训练 pipeline；val/test/predict 仍沿用官方流程：

```json
{
  "size": [512, 512],
  "augmentation": {
    "mode": "random_crop",
    "resize_scale": [1024, 512],
    "ratio_range": [0.5, 2.0],
    "cat_max_ratio": 0.95,
    "horizontal_flip_prob": 0.5,
    "vertical_flip_prob": 0.0,
    "photometric_distortion": true,
    "extra_transforms": [
      {"type": "RandomRotate", "prob": 0.25, "degree": 10}
    ]
  },
  "scheduler": {
    "type": "poly",
    "warmup_iters": 1500,
    "warmup_ratio": 0.000001,
    "power": 1.0,
    "eta_min": 0.0
  },
  "accumulative_counts": 1
}
```

- `size` 是 **[高, 宽]**；`resize_scale` 沿用原生的 **[宽, 高]**。随机缩放保持原始长宽比，然后裁出目标 crop。随机裁剪只在训练时使用；验证/测试使用确定性 resize 和原尺寸 GT，预测不会随机裁剪。
- `cat_max_ratio` 限制 crop 中单一类别的占比。默认继承官方参数（这些模板为 0.75）；上面的自定义示例为 0.95，1.0 表示不做类别占比限制。这是官方有限次数重采样策略，**不保证每个 crop 都有裂缝**，也不改变训练/验证划分。
- 两个翻转概率独立；光照扰动可设 `false` 关闭，或设为含 `brightness_delta`、`contrast_range`、`saturation_range`、`hue_delta` 的原生参数对象。
- `extra_transforms` 可追加已注册的单图原生变换，例如随机旋转；旋转默认对 mask 的新区域填 255。为保持 crop 尺寸，禁止 `auto_bound=true`，且此入口不能替换图片/标签加载、resize 和打包阶段。Mosaic 等需要混合多张图片和数据集包装器的策略，不能只通过这个单图入口接入。
- 若要显式使用自定义固定缩放，设置训练 `size` 及 `"augmentation": {"mode": "resize", "horizontal_flip_prob": 0.5, "photometric_distortion": false}`。恢复官方增强则设为 `{"mode":"upstream"}` 或删除 augmentation。
- `scheduler.type` 支持 `poly` 或 `cosine`，均可叠加原生 `LinearLR` warmup。warmup 的 `start_factor` 对应 `warmup_ratio`。极短 `--max-iters` 检查会把 warmup 截断到 `max_iters-1`；warmup=0 表示不启用。
- 保留上游优化器种类、参数分组、layer/stage decay、无 weight decay 的 norm/position embedding 规则，以及 Mask2Former 的梯度裁剪。`amp` 控制混合精度；可添加 `"grad_clip": {"max_norm": 1.0, "norm_type": 2}` 覆写裁剪，`null` 关闭裁剪；也可用现有 `--cfg-options` 覆写原生优化器参数。
- `accumulative_counts` 使用 MMEngine 原生梯度累积。每卡 batch=2、双卡、累积=4 时，一个完整优化更新对应 16 张图；`max_iters`、保存间隔、验证间隔及早停仍按 dataloader 训练迭代数计算，不按 optimizer 更新次数计算。学习率仍由你显式设置。

模板的数据策略沿用各自官方配置；训练长度、验证/保存频率、早停和日志仍由本项目实验设置控制，因此不是原论文成绩的完整复现实验。默认仍 `no_pretrained=true`；要使用上游 backbone 预训练请显式改为 false。

### 当前实验的损失函数与优化器

损失函数衡量预测误差；SGD、Adam、AdamW 根据损失的梯度更新模型参数。本项目保留每个上游算法的损失配置，具体值可从展开配置核对：

| 实验 | 当前损失配置 | 优化器、初始基础 LR |
| --- | --- | --- |
| U-Net、DeepLabV3+ R50 | 主头 CE × 1.0 + 辅助头 CE × 0.4 | SGD，0.01，momentum=0.9 |
| SegFormer B0、SegNeXt Tiny/Small | 主头 CE × 1.0 | AdamW，0.00006 |
| ConvNeXt Tiny/Small | 主头 CE × 1.0 + 辅助头 CE × 0.4 | AdamW，0.0001 |
| Swin Tiny/Small、Twins SVT/PCPVT Small | 主头 CE × 1.0 + 辅助头 CE × 0.4 | AdamW，0.00006 |
| Twins SVT-S + FPN | 主头 CE × 1.0 | AdamW，0.0001 |
| Mask2Former R50/Swin-Tiny | 分类 CE × 2 + Mask BCE × 5 + Dice × 5，保留 decoder 中间层监督 | AdamW，0.0001 |

普通模型的 CE 是二类 softmax 交叉熵；辅助头是另一组预测上的交叉熵。Mask2Former 的分类包含背景、前景和无目标，class weights 为 `[1, 1, 0.1]`。它的 mask 损失为 sigmoid BCE，Dice 也来自原生实现；中间 decoder 层还会产生额外的监督损失，因此总 loss 并非只有最后一层的三个数。各分项日志值已经由原生损失函数应用权重。

当前普通模型使用 CE；`mDice` 是评估指标，不代表其训练自动使用了 Dice loss。实际所有损失类型、权重、学习率、weight decay 和参数分组可生成报告查看：

```bash
python scripts/describe_experiments.py
```

输出 `reports/experiment_training.json`；每次训练的 `run_manifest.json` 也会保存当前展开配置中的相同信息，包含 CLI 覆写后的结果。上表是模板默认值，warmup、参数分组和调度器会进一步改变每组实际学习率。

SGD（这里带 momentum）主要沿梯度及累计方向更新，每组使用设定的学习率；Adam 还估计梯度的一阶、二阶矩，为不同参数调整更新幅度，通常需要更多优化器状态。AdamW 在 Adam 的基础上把 weight decay 从梯度更新中解耦；本项目上述自适应优化器实际使用 **AdamW**。两类方法没有对所有任务都更好的结论，切换时需要重新设置学习率，不能直接照搬 SGD 的 0.01。实现依据：[SGD](https://github.com/pytorch/pytorch/blob/v2.1.0/torch/optim/sgd.py)、[Adam](https://github.com/pytorch/pytorch/blob/v2.1.0/torch/optim/adam.py)、[AdamW](https://github.com/pytorch/pytorch/blob/v2.1.0/torch/optim/adamw.py)。

默认 `no_pretrained: true` 从零训练且移除预训练下载；要使用上游 backbone 预训练权重，可设为 `false`，首次运行可能下载权重。改变前景类别后，不能直接加载上游多分类分割头权重。`amp: true` 可启用混合精度。可添加 `lr` 覆盖上游优化器学习率。

## 4. 训练、评估、预测

```powershell
# 只解析、展开并保存配置；不会构建模型或训练
python -m simple_seg train configs/experiments/segformer_b0.json --dry-run

# 正式训练
python -m simple_seg train configs/experiments/segformer_b0.json

# 短训练检查，可另存目录；会真实读取数据并执行模型
python -m simple_seg train configs/experiments/unet.json --max-iters 2 --work-dir work_dirs/smoke --cfg-options train_cfg.val_interval=1 default_hooks.checkpoint.interval=1

# 恢复同一实验最近一次训练，或指定 checkpoint
python -m simple_seg train configs/experiments/segformer_b0.json --resume
python -m simple_seg train configs/experiments/segformer_b0.json --resume work_dirs/segformer_b0/iter_265.pth

# 在独立 test 集评估
python -m simple_seg test configs/experiments/segformer_b0.json work_dirs/segformer_b0/iter_30000.pth

# 单张图片或整个文件夹，输出原图尺寸的黑白 PNG
python -m simple_seg predict work_dirs/segformer_b0/resolved_config.py work_dirs/segformer_b0/iter_30000.pth dataset/Image --output predictions
```

训练目录包含展开的 `resolved_config.py`、日志、每次验证的 checkpoint 和最佳 mIoU 权重。默认保留全部普通 checkpoint，以及一个当前最佳 mIoU 权重。上游 IoUMetric 输出每类 IoU、Dice、Precision、Recall、F-score 以及均值；**mIoU/mDice/mFscore 包含背景与前景两类**，裂缝任务请同时查看 `foreground` 行。评估在原始 mask 尺寸进行；预测输出为 `0/255`。

### 训练日志与 TensorBoard WebUI

沿用你指定的 [MMEngine 官方训练日志可视化方案](https://mmengine.readthedocs.io/en/latest/common_usage/visualize_training_log.html)，默认同时启用 `LocalVisBackend` 和 `TensorboardVisBackend`。无需 TensorFlow、云账号或上传数据。依赖已经加入 requirements；已有环境只需补装：

```bash
python -m pip install tensorboard==2.18.0
# 在另一个终端启动浏览器界面，训练可以继续运行
python scripts/webui.py --logdir work_dirs --port 6006
```

打开 `http://127.0.0.1:6006`。也可以使用等价的原生命令 `python -m tensorboard.main --logdir work_dirs --host 127.0.0.1 --port 6006`。`scripts/webui.py` 默认只监听本机；Ctrl+C 停止 WebUI，不影响另一终端的训练。SSH 服务器训练时，可在本机另开终端执行 `ssh -L 6006:127.0.0.1:6006 user@server`，再访问本机地址。

- **比较实验**：选择 `work_dirs` 递归读取所有实验和所有运行记录；选择 `work_dirs/batch/<批次ID>` 只比较该批次；也可把 `--logdir` 指向单模型目录。
- **训练曲线**：`train/loss`、主头/辅助头/Mask2Former decoder 各分项 loss、学习率、时间及框架记录的梯度信息等原生标量。
- **验证曲线**：`val/mIoU`、`val/mDice`、`val/Precision`、`val/Recall`、`val/F1-score` 等。默认汇总值为两类宏平均；每类指标仍保留在文字日志表中。验证阶段默认记录分割指标，训练 loss 来自训练阶段。
- **测试结果**：`test/...` 写入事件文件和 `test_metrics.jsonl`，单次 checkpoint 测试横坐标为 0；验证曲线横坐标是实际训练 iter。
- **及时落盘**：每个日志记录周期和每次验证后 flush TensorBoard；正常结束、早停和训练异常路径关闭 writer。异常会记入文字日志，运行状态标为 `failed` / `interrupted`。

所有 13 个 JSON 模板都包含：

```json
{
  "logging": {
    "interval": 10,
    "window_size": 10,
    "tensorboard": true
  }
}
```

`interval` 为训练标量记录周期，`window_size` 为 loss 等原生平滑统计窗口；若希望每一步记录未平滑 loss，两个值都设为 1。验证指标每次验证完整保存。`tensorboard=false` 可关闭事件文件写入，保留文字/JSON 日志。TensorBoard 自身还提供显示平滑，修改显示平滑不会改写原始事件数据。新训练会写事件文件；此前只有纯 JSON 的历史日志不会自动变成 TensorBoard 曲线。

保存目录沿用 MMEngine 时间戳结构，恢复训练继续使用原 checkpoint，同时留下新的运行日志目录：

```text
work_dirs/<实验>/
  resolved_config.py                 最近一次展开配置
  iter_N.pth / best_mIoU_iter_N.pth  权重
  <MMEngine 时间戳>/
    <时间戳>.log                     原生文字日志，含每类指标和错误
    run_manifest.json               实验参数、实际损失/优化器、版本、划分数量及 SHA256
    run_status.json                 initialized/running/completed/early_stopped/failed/interrupted/tested
    val_metrics.jsonl               每次验证的完整指标与 iter
    test_metrics.jsonl              执行测试时产生
    vis_data/
      config.py                     本次展开配置快照
      scalars.json                  原生标量 JSON Lines
      <时间戳>.json                 原生 LoggerHook 的标量记录
      events.out.tfevents.*         TensorBoard 事件
```

本地日志默认保留，不随 checkpoint 保留数量删除。双卡只由主进程写这些可视化与运行记录；批量训练每个模型的事件文件相互独立。`run_manifest.json` 中的原始实验参数与实际解析结果分别保存，CLI 覆写以实际解析结果为准。

### 保存间隔、评估指标和早停

全部实验 JSON 都显式提供以下设置：

```json
{
  "max_iters": 30000,
  "val_interval": 265,
  "checkpoint_interval": 265,
  "save_on_val": true,
  "max_keep_ckpts": -1,
  "metrics": ["mIoU", "mDice", "mFscore", "Precision", "Recall", "F1-score"],
  "early_stopping": {
    "patience_iters": 5000,
    "min_delta": 0.0
  }
}
```

| 参数 | 含义 |
| --- | --- |
| `max_iters` | 最大训练迭代数，早停可以提前结束训练 |
| `val_interval` | 每隔多少训练 iter 做一次验证；最大迭代结束时也会验证 |
| `checkpoint_interval` | 每隔多少 iter 保存普通权重，独立于验证间隔 |
| `save_on_val` | 默认 `true`：每次验证后都保存对应 `iter_N.pth`，即使不在定期保存点 |
| `max_keep_ckpts` | 默认 `-1` 保留所有普通权重；设正整数则只保留最近 N 个，包含验证快照 |
| `metrics` | 原生 MMSeg `IoUMetric` 的指标选择；必须包含用于最佳权重和早停的 `mIoU` |
| `early_stopping.patience_iters` | 距上次有效验证 mIoU 提升已过去的训练 iter 数，默认 5000 |
| `early_stopping.min_delta` | 有效提升要求：`新 mIoU > 上次有效最佳 mIoU + min_delta`；默认 0，持平不算提升 |

`checkpoint_interval=500`、`val_interval=1000` 时，定期权重在 500、1000、1500… iter 保存，并保证每次验证都有对应快照。若两个事件落在同一 iter，使用同一个 `iter_N.pth`，验证后刷新其中的最佳指标和早停状态，不会生成 `iter_N+1.pth`。普通快照包含优化器、调度器和 MessageHub，可用于 `--resume`。

最佳权重始终由官方 CheckpointHook 逻辑按验证 **mIoU 严格增大**保存为 `best_mIoU_iter_N.pth`，出现更好的 mIoU 时替换旧的 best 文件；普通历史快照不会因此删除。best 文件沿用官方行为，不带优化器/调度器状态，适合评估或预测，恢复完整训练请使用普通 `iter_N.pth`。

早停以第一次验证建立初始最佳值。如果最佳有效 mIoU 最后出现在 iter 3000，而之后验证一直未提升，`patience_iters=5000`、`val_interval=1000` 会在 iter 8000 验证后停止；如果 iter 7000 出现有效提升，则计数从 7000 重新开始。计数单位是训练迭代数，**不是 5000 次验证**。

早停只在验证结果产生后判断：当 patience 不是验证间隔的倍数时，在第一次满足迭代跨度的验证点停止，不会在没有新验证结果的两个验证点之间猜测 mIoU。例如最佳在 1000，之后验证点为 3500、6100，则在 6100 检查到已超过 5000 iter。`min_delta` 与原生返回值使用相同单位：MMSeg mIoU 为 0–100 的百分数且保留两位小数，`min_delta=0.1` 表示 0.1 个百分点。只有默认 `min_delta=0` 时，早停的“有效最佳”与 best checkpoint 的严格最佳完全一致。

早停计数会随普通 checkpoint 保存，`--resume` 后继续计数；早停触发时也保存当前可恢复权重。设置 `"early_stopping": null` 关闭早停。双卡各进程使用 MMEngine 汇总后的同一份验证指标作出一致决定；批量任务中正常早停视为该模型训练完成，之后继续下一个模型。

指标沿用官方 [IoUMetric](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/evaluation/metrics/iou_metric.py)：默认配置会返回 `aAcc`、`mAcc`、`mIoU`、`mDice`、`mFscore`、`mPrecision`、`mRecall`；F-score 默认 beta=1，即 F1。每类结果打印到原生日志。没有自行重写指标公式。

实验配置也显式列出 `Precision`、`Recall`、`F1-score`。它们映射到原生 `mFscore` 指标组，固定 `beta=1`，并在验证/测试日志和日志 JSON 中输出同名字段，分别等于 `mPrecision`、`mRecall`、`mFscore`。这些汇总值是**背景与前景两类的宏平均**，不是仅前景的指标；裂缝本身的 Precision/Recall/F1 请查看原生每类结果表中的 `foreground` 行（该表的 F1 列名为 `Fscore`）。原生指标字段同时保留，best 和早停仍按 `mIoU`。

权重保存扩展自官方 [CheckpointHook](https://github.com/open-mmlab/mmengine/blob/v0.10.7/mmengine/hooks/checkpoint_hook.py)。官方 [EarlyStoppingHook](https://github.com/open-mmlab/mmengine/blob/v0.10.7/mmengine/hooks/early_stopping_hook.py) 的 patience 按验证次数计数，本项目使用 `IterEarlyStoppingHook` 按实际 iter 跨度计算，并在 checkpoint 写入前更新可恢复状态。

测试包含真实 MMEngine Runner 的小型 CPU 模型训练、checkpoint 写入和断点恢复：第一次在 iter 2 达到最佳，iter 4 保存后恢复，patience=4 时在 iter 6 停止，验证计数未重置。此检查不代表完整版 MMSeg 分割模型或双卡训练已经实测。

`--cfg-options` 支持 MMEngine 的点路径覆写，例如 `optim_wrapper.optimizer.lr=0.0001`。显式传入的 `--work-dir` 优先于配置覆写。

## 5. 单卡、双卡与批量训练脚本

双卡沿用原生 MMSegmentation 的方式：每张 GPU 一个进程，`--launcher pytorch` 交给 MMEngine Runner 初始化分布式、包装 DDP、同步梯度、分配采样和汇总评估；不另写训练循环。保留原模型的 SyncBN 设置。训练入口同时接受原生的 `--local-rank` / `--local_rank`、`--amp`、`--resume`、`--cfg-options`；也接受 `slurm` / `mpi` launcher，但需由对应集群调度器提供运行环境。

**双卡 GPU 默认使用 NCCL，推荐 Linux/WSL2 + NVIDIA GPU。原生 Windows 的 PyTorch 不支持 NCCL，不能直接用这个默认后端执行双卡训练；单卡仍可使用 Windows。** 本项目不会默默改成未经验证的 Gloo GPU 路径。运行前可以用 `python -c "import torch; import torch.distributed as d; print(torch.cuda.device_count(), d.is_nccl_available())"` 检查当前环境。若在 WSL2 中运行，需在 WSL2 内重新安装 Linux 版环境，不能复用 Windows 的 `.venv-mmseg`。

### 原生风格入口

`tools/train.py` 使用本项目的 JSON 实验配置；`tools/dist_train.sh` 保留 MMSeg 的 `CONFIG GPUS` 参数顺序，内部使用推荐的 `torch.distributed.run`（torchrun），替代上游旧的 `torch.distributed.launch`：

```bash
# 单卡
python tools/train.py configs/experiments/segformer_b0.json --amp

# Linux / WSL2 双卡
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh configs/experiments/segformer_b0.json 2 --amp

# 从同一工作目录最近一次 checkpoint 恢复
bash tools/dist_train.sh configs/experiments/segformer_b0.json 2 --amp --resume

# 指定端口，防止同时运行的任务冲突
PORT=29601 bash tools/dist_train.sh configs/experiments/unet.json 2
```

脚本与原生一样接受环境变量 `NNODES`、`NODE_RANK`、`MASTER_ADDR`、`PORT`。多节点需要在每个节点执行相同命令并设置不同 `NODE_RANK`，保持相同数据路径、配置、GPU 数和主节点地址。

### Python 训练启动脚本

`scripts/train.py` 为单卡或双卡生成上述原生启动命令，默认使用当前 Python 环境；双卡示例同样应在 Linux/WSL2 中执行：

```bash
python scripts/train.py configs/experiments/segformer_b0.json --gpus 1 --amp
python scripts/train.py configs/experiments/segformer_b0.json --gpus 2 --gpu-ids 0,1 --amp
python scripts/train.py configs/experiments/segformer_b0.json --gpus 2 --port 29601 --resume

# 只打印启动命令，不启动进程、不训练、不写输出
python scripts/train.py configs/experiments/segformer_b0.json --gpus 2 --print-only

# 执行配置检查，仍不构建模型或训练
python scripts/train.py configs/experiments/unet.json --dry-run
```

JSON 中的 `batch_size` 与原生 MMSeg 一样是**每张卡的 batch size**：当前 `batch_size=12` 时双卡全局 batch 为 24（未启用梯度累积时）。增加卡数不会自动修改学习率；可用 `--cfg-options optim_wrapper.optimizer.lr=0.0001` 显式调整。`max_iters` 仍是全局训练迭代数，不会自动除以卡数；若要保持相同样本预算，可按上文显式传入 `--max-iters`。配置文件只由主进程保存；日志、权重及评估同步沿用 MMEngine 行为。

### 按顺序批量训练多个模型

示例清单 `configs/batch/default.json` 包含 SegFormer-B0、DeepLabV3+、U-Net。批量训练每次只运行一个模型；每个模型占用所选的全部 GPU，完成后才开始下一个。

`configs/batch/extended.json` 包含全部 13 个实验，包括 ConvNeXt、Mask2Former、Swin、Twins、SegNeXt。完整安装后运行 `python scripts/batch_train.py configs/batch/extended.json --gpus 2`。所有模板使用每卡 batch=12；ConvNeXt 模板默认 AMP，Mask2Former 模板默认全精度。批量公共参数 `--amp` 会强制所有模型开启 AMP，只有需要时再加。

```bash
# 单卡批量训练
python scripts/batch_train.py configs/batch/default.json --gpus 1

# Linux / WSL2：每个模型使用双卡
python scripts/batch_train.py configs/batch/default.json --gpus 2 --gpu-ids 0,1 --amp

# 指定这次批量实验的新目录；遇到失败继续下一个模型
python scripts/batch_train.py configs/batch/default.json --gpus 2 --work-root work_dirs/comparison_01 --continue-on-error

# 仅查看所有将要执行的命令
python scripts/batch_train.py configs/batch/default.json --gpus 2 --print-only
```

不指定 `--work-root` 时自动生成 `work_dirs/batch/<独立批次ID>/`；每个模型保存到其中独立的命名目录。指定目录必须尚不存在，防止覆盖已有批次。`summary.json` 持续记录各实验命令、运行状态和退出码；默认失败即停止，后续实验标为 `pending`，使用 `--continue-on-error` 可继续。整个批次只要有失败就返回非零退出码；Ctrl+C 会记录中断并退出。`--dry-run` 会在报告中标为 `config_check`，不表示真实训练成功。批量编排只支持单节点。

清单接受配置路径字符串，或含 `name`、`config`、`args` 的对象；路径相对项目根目录。例如：

```json
[
  {
    "name": "segformer_b0",
    "config": "configs/experiments/segformer_b0.json",
    "args": ["--max-iters", "10000"]
  },
  {
    "name": "unet",
    "config": "configs/experiments/unet.json",
    "args": ["--amp", "--cfg-options", "optim_wrapper.optimizer.lr=0.001"]
  }
]
```

公共训练参数会传给每个模型；对象中的 `args` 追加到公共参数之后。名称必须唯一，只能使用字母、数字、下划线、连字符和点；不要在清单中指定 `--work-dir`，由批量脚本分配。

实现参考原生 [tools/train.py](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/tools/train.py)、[tools/dist_train.sh](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/tools/dist_train.sh) 和 [MMEngine Runner](https://github.com/open-mmlab/mmengine/blob/v0.10.7/mmengine/runner/runner.py)。目前已验证命令构造、参数透传、主进程配置保存、独立目录、失败处理和配置解析；**尚未进行真实双 GPU 模型训练验证**。

## 双卡并行训练八种轻量模型

运行 **`scripts/train_gpu_queue.py`**：每张卡同时只运行一个独立单卡模型，空闲后立即领取下一个模型。默认 GPU 0 先训练 ConvNeXt，GPU 1 先训练 DeepLabV3+；之后按空闲顺序调度剩余任务。该队列使用普通单卡 Runner，因此原生 Windows 也能使用这两张卡并行训练不同模型。单个模型每卡 batch=12、30000 iter，不会因为电脑有两张卡而合成全局 batch=24。

任务列表为 [`configs/batch/lightweight.json`](configs/batch/lightweight.json)，只包含以下八项，选择范围为项目固定 MMSeg v1.2.2 提供的轻量配置，并不包含另行移植其他库的 Nano 等骨干：

| 算法 | 所选版本 |
| --- | --- |
| ConvNeXt | Tiny + UPerNet |
| DeepLabV3+ | MobileNetV2，使用官方 ADE20K pipeline |
| Mask2Former | ResNet-50 |
| SegFormer | MiT-B0 |
| SegNeXt | MSCAN-Tiny |
| Swin | Tiny + UPerNet |
| Twins | SVT-S + FPN，使用更轻的 FPN 头 |
| U-Net | UNet-S5-D16 + FCN，继承官方 DRIVE 配置 |

轻量配置来源可核对 [MobileNetV2 DeepLabV3+](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/configs/mobilenet_v2/mobilenet-v2-d8_deeplabv3plus_4xb4-160k_ade20k-512x512.py) 和 [Twins SVT-S FPN](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/configs/twins/twins_svt-s_fpn_fpnhead_8xb4-80k_ade20k-512x512.py)。

```powershell
conda activate ./.venv-mmseg
# 可选：先做快速 JSON/数据路径/继承配置检查
python scripts/check_experiments.py --level config
# 正式入口：自动先做全部八个模型的真实单步 smoke 检查，通过后才开始排队训练
python scripts/train_gpu_queue.py --gpu-ids 0,1
```

需要独立进行完整训练前检查时：

```powershell
python scripts/check_experiments.py --level smoke --gpu-ids 0,1
# 检查和训练都强制开启 AMP；两阶段会使用一致的参数
python scripts/train_gpu_queue.py --gpu-ids 0,1 --amp
# 只预览八个任务命令，不启动进程、不写输出
python scripts/train_gpu_queue.py --gpu-ids 0,1 --print-only
```

检查脚本默认读取同一份 lightweight.json，也可传其他批次清单。`config` 检查 JSON 拼写及类型、官方继承、数据清单全部文件路径、集合重复和泄漏、完整配置展开；不构建模型。`build` 需要 CUDA 和完整依赖，加载注册模块并构建模型/初始化权重/构建优化器。`smoke` 在每个指定 GPU 上依次轮换检查模型，保留真实训练 batch、模型、优化器、AMP 和官方增强，实际运行一次前向/反向/优化更新，再对一个真实 val 样本和一个 test 样本执行推理评估。检查期间不保存模型权重，也不写入正式训练的 TensorBoard 曲线。失败时记录完整 traceback，包括缺依赖、算子问题和 OOM。

队列默认先顺序完成全部八个 smoke 检查，任何一个未通过都不启动正式训练；随后最多同时训练两个模型。`--check-level config` 或 `build` 可显式降低启动前检查级别。每个检查默认最多 900 秒，超时标记失败；独立检查用 `--timeout`、队列用 `--check-timeout` 修改。单步检查不能保证所有尺寸、随机裁剪和整个 30000 iter 都不出现问题，显存不足时需要显式调整 batch 或启用 AMP，不会自动改动配置。

队列本身在终端打印每个任务的起止与整体进度，例如：

```
Starting convnext_tiny on GPU 0 (1/8 started). Log: work_dirs/gpu_queue/<hash>/convnext_tiny.console.log
Finished convnext_tiny on GPU 0: exit 0 (1/8 done)
...
Queue finished: 8/8 succeeded.
```

每个模型训练的实时输出（loss、评估、进度条）默认写入 `work_dirs/gpu_queue/<hash>/<模型>.console.log`，TensorBoard 曲线写入 `work_dirs/gpu_queue/<hash>/<模型>/vis_data`。想同时在终端实时滚动这些日志，加 `--follow`：

```bash
python scripts/train_gpu_queue.py --gpu-ids 0,1 --follow
# 或事后跟随文件：
RUN=$(ls -t work_dirs/gpu_queue | head -1); tail -f work_dirs/gpu_queue/$RUN/*.console.log
```

`--follow` 下每行日志带 `[模型名]` 前缀，多个模型并行时不会混淆。

输出默认在 `work_dirs/gpu_queue/<批次ID>/`：每个模型有独立的 checkpoint、日志、TensorBoard 和运行状态；`<模型名>.console.log` 保存完整控制台输出；`preflight/summary.json` 汇总检查结果；`summary.json` 记录 GPU 分配、PID、退出码和任务状态。也可通过 `--work-root` 指定一个尚不存在的目录。正式训练中某个任务失败会标记失败并继续其他任务，整个批次最终返回非零退出码；添加 `--fail-fast` 后不再派发新任务，已运行的任务正常结束。Ctrl+C 停止此队列启动的训练进程。

## 配置兼容范围

自动适配标准 `EncoderDecoder` / `CascadeEncoderDecoder`：递归修改主头、辅助头的 `num_classes` 和普通分割输出 `out_channels`，保留模型结构与损失，清除不适用的多类 class weights；Mask2Former 另行处理 mask 特征通道和带无目标类别的分类损失。学习率调度和数据增强由上述实验参数控制。它不会原样重现上游的多阶段训练 recipe 或 benchmark 成绩。

U-Net、FCN、PSPNet、DeepLab、SegFormer、UPerNet 以及新增的 ConvNeXt、Mask2Former、Swin、Twins、SegNeXt 是适配目标。依赖 MMDetection/MMPRETRAIN 或额外算子的模型，仍需安装它们要求的组件；深度估计、开放词汇、多模态及其他自定义优化流程需要另外的 adapter。TTA 配置保留用于参考，当前入口默认单尺度推理。不能把“配置已保留/解析成功”当成“所有模型已训练验证”。

可在装有 MMEngine 的环境生成逐配置解析报告：

```powershell
python scripts/audit_configs.py
python scripts/audit_data_pipeline.py
python -m unittest discover -s tests -v
```

`reports/config_compatibility.json` 仅说明解析和二分类配置适配结果，**不会构建模型或证明运行依赖完整**。解析时推迟导入可选模型模块，报告中的 custom_imports 表示运行时还需要安装并加载哪些组件。测试覆盖数据校验、划分、mask 转换、原生增强配置、学习率调度，以及全部 13 种模板的真实 MMEngine 继承解析和展开配置重新加载。

当前本机检查结果：6,042 对真实图片/mask 全部通过校验，按 7:2:1 划分为训练 4,230、验证 1,208、测试 604。上游配置解析结果见 `reports/config_compatibility.json`；13 个模板的训练/验证/测试 pipeline、预测 pipeline、预处理器和 whole/slide 参数均与上游逐项对照，除黑白标签加载外一致。此轮解析未执行可选模型模块导入，不能据此认为运行依赖已经齐全。测试覆盖全部 13 个实验模板、独立训练 crop 覆写、数据集包装器、真实 MMEngine 小型 CPU 模型训练/断点恢复、原生 TensorBoard 事件与 JSON 写入/读回，以及训练失败时的状态记录和 writer 关闭。另已临时启动 TensorBoard 并验证 WebUI HTTP 200 和标量 API，测试事件为独立的合成测试数据，测试服务已停止。独立环境安装脚本已通过 PowerShell 语法检查，官方 Windows/Python 3.10 的 MMCV 2.1.0 CUDA 11.8 wheel 已核对；**尚未完整安装并实际训练/评估/预测新增的 MMSeg 模型，也未执行完整后端的数据增强或双卡训练**。本机测试仅临时使用已有 Python 环境和项目内的 `.test-deps` 配置依赖，没有更改已有环境。

添加自己的算法时，在本地 Python 配置中指定 `custom_imports` 导入你的注册模块，再将实验 JSON 的 `model_config` 指向该配置；原有自定义模块导入会与通用数据集模块一起保留。

## 项目结构与来源

```text
simple_seg/data.py       配对、严格检查、划分、mask 转换
simple_seg/dataset.py    通用 MMSeg Dataset 和 mask loader
simple_seg/adapter.py    上游配置转为二分类实验
simple_seg/pipelines.py  原生数据增强和学习率调度设置
simple_seg/logging_hooks.py  日志归档、运行状态及事件 flush
simple_seg/reporting.py  实际损失与优化器报告
scripts/webui.py         TensorBoard 浏览器入口
scripts/describe_experiments.py  生成逐实验损失与优化器报告
simple_seg/hooks.py      每次验证/最佳权重保存、按 iter 计数的早停
simple_seg/cli.py        train / test / predict 等入口
configs/experiments/    简短的个人实验配置
third_party/            固定版本上游配置与 LICENSE
scripts/setup.ps1       独立环境安装
tools/train.py          原生风格训练入口
tools/dist_train.sh     CONFIG GPUS 分布式启动脚本
scripts/train.py        单卡 / 多卡 Python 启动脚本
scripts/batch_train.py  顺序批量训练
configs/batch/          批量实验清单
```

上游模型和配置来自 [MMSegmentation v1.2.2](https://github.com/open-mmlab/mmsegmentation/tree/v1.2.2)，遵循其中 Apache-2.0 许可证；原始文件保留版权标记。环境兼容约束见 [MMSeg 安装文档](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/docs/en/get_started.md) 及其 [版本检查](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/__init__.py)。本项目没有复制一套模型源码，后端由 pip 包提供。
