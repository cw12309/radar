"""
推理引擎封装模块
封装真实RADAR模型的加载和推理逻辑。

本模块复用 RADAR_inference/inference_demo.py 中官方定义的模型结构与预处理流程，
将其改造为设备无关（支持 CPU / CUDA），并输出统一的结构化诊断结果。
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RADAR_INFERENCE_PATH = os.path.join(PROJECT_ROOT, 'RADAR_inference')

# 模型与配置根目录（inference_demo / med.py 通过环境变量读取，必须在导入前设置）
DEFAULT_CKPT_ROOT = os.path.join(PROJECT_ROOT, 'ckpt')

# CUDA 版 torch 位于项目内独立目录 .torch_cu126（因 user-site 目录无写权限，
# 改用 pip install --target 安装）。必须在 import torch 之前把该目录插到
# sys.path 最前面，否则 torch 会解析到 user-site 的 CPU 版而用不上 GPU。
# 目录不存在时静默跳过，自动回落到环境中已安装的 torch（CPU 亦可推理）。
_CUDA_TORCH_DIR = os.path.join(PROJECT_ROOT, '.torch_cu126')
if os.path.isdir(os.path.join(_CUDA_TORCH_DIR, 'torch')):
    if _CUDA_TORCH_DIR not in sys.path:
        sys.path.insert(0, _CUDA_TORCH_DIR)
    elif sys.path[0] != _CUDA_TORCH_DIR:
        sys.path.remove(_CUDA_TORCH_DIR)
        sys.path.insert(0, _CUDA_TORCH_DIR)

# 显存碎片治理：必须在 import torch（CUDA 分配器初始化）之前设置。
# 单窗前向峰值约需 6.3 GB，8 GB 显存余量本就吃紧；默认分配器在反复
# 分配/释放大块概率图后会留下大量 "reserved but unallocated" 空洞，
# 实测单帧病例因碎片导致 OOM（报错被 CUDA 异步机制伪装成
# "CUDA error: unknown error"，stderr 才可见真实的 allocation failure）。
# expandable_segments 让分配器按需增长段而非预留大块，消除空洞。
# 仅在用户未显式配置时注入，不覆盖外部指定。
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch


def available_memory_bytes():
    """返回系统当前可用物理内存字节数，无法获取时返回 None。

    优先使用 psutil；缺失时按平台回退到系统调用。
    """
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    if sys.platform.startswith('win'):
        import ctypes

        class _MSX(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
            ]

        try:
            stat = _MSX()
            stat.dwLength = ctypes.sizeof(_MSX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
        except Exception:
            return None

    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None

    return None


def available_commit_bytes():
    """返回系统还能提交多少内存（字节），无法获取时回退到可用物理内存。

    Windows 上真正决定「分配会不会失败」的是提交上限（物理内存 + 页面文件）
    的剩余量，而不是空闲物理内存：
      - Windows 会主动把空闲物理内存拿去做文件缓存，ullAvailPhys 常年偏低，
        实测 GPU 推理在 ullAvailPhys 跌到 0.69 GB 时仍能正常完成（系统换页），
        用它做判据会把本可成功的推理误判为「影像过大」；
      - 而 GPU 回传窗块失败（DefaultCPUAllocator: not enough memory）时，
        ullAvailPhys 尚有 1.25 GB，耗尽的是提交余量（仅 0.16 GB）。
    """
    if sys.platform.startswith('win'):
        import ctypes

        class _MSX(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
            ]

        try:
            stat = _MSX()
            stat.dwLength = ctypes.sizeof(_MSX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPageFile)
        except Exception:
            pass

    return available_memory_bytes()


def _postprocess_overlay_mask(mask, ct=None):
    """对叠加显示用的器官掩码做形态学后处理（仅改善可视化，不影响诊断）。

    模型原始 argmax 输出含散点噪声、器官内部裂缝，且器官之间夹着成片
    未标注软组织，直接叠加会呈现「填充不完整」的破碎观感。依次执行：
      0) 可靠性门控：剔除总体积<300 且最大连通域<100 体素的幻觉标签；
      1) 去除小于 8 像素的散点连通域；
      2) 半径 2 圆盘闭运算，连通器官内部裂缝；
      3) 填充封闭空洞（含被器官包围的气腔）；
      4) 若提供 ct：体内（非气体）未标注体素按最近邻归入相邻器官，
         使器官区域连续完整，气体与体外背景保持原样。
    合并时按器官总体积降序写入，大器官优先占位。诊断指标在导出前已基于
    原始掩码计算，不受影响。对已后处理的掩码再次调用是幂等的。
    """
    from scipy import ndimage

    mask = np.asarray(mask)
    # 可靠性门控：总体积 <300 且最大 3D 连通域 <100 体素的标签整器官剔除。
    # 这类标签是模型幻觉散点或严重碎片化预测（实测腹部病例「脑」244 体素
    # 散在 10 个连通域、最大 60；「肾」212 体素散在 11 个连通域），且会在
    # 体内补全阶段被放大成大片假区域，严重破坏叠加可信度。真实器官要么
    # 总体积足够大，要么呈单一紧凑连通域（如门静脉 200 体素单连通域），
    # 不会被误删。
    dropped = []
    for lab in (int(v) for v in np.unique(mask) if v > 0):
        vol = mask == lab
        total = int(vol.sum())
        if total >= 300:
            continue
        lbl, _n = ndimage.label(vol)
        max_comp = int(np.bincount(lbl.ravel())[1:].max())
        if max_comp < 100:
            dropped.append(lab)
    if dropped:
        mask = mask.copy()
        mask[np.isin(mask, dropped)] = 0

    labels = [int(v) for v in np.unique(mask) if v > 0]
    if not labels:
        return mask.astype(np.uint8)

    out = np.zeros(mask.shape, dtype=np.uint8)
    yy, xx = np.mgrid[-2:3, -2:3]
    struct = (yy ** 2 + xx ** 2) <= 4
    # 体积降序：大器官先占位，小器官只填空余，防止闭运算后相互吞噬
    for lab in sorted(labels, key=lambda l: -int((mask == l).sum())):
        vol = mask == lab
        for s in range(vol.shape[0]):
            sl = vol[s]
            if not sl.any():
                continue
            lbl, n = ndimage.label(sl)
            if n > 1:
                counts = np.bincount(lbl.ravel())
                counts[0] = 0
                sl = (counts >= 8)[lbl]
                if not sl.any():
                    continue
            sl = ndimage.binary_fill_holes(ndimage.binary_closing(sl, structure=struct))
            vol[s] = sl
        out[(vol) & (out == 0)] = lab

    if ct is None:
        return out

    # 体内未标注区最近邻补全；ct 可能是 [0,1] 或 [0,255]，统一归一后取气体阈值
    ctn = np.asarray(ct, dtype=np.float32)
    if ctn.max() > 1.5:
        ctn = ctn / 255.0
    for s in range(out.shape[0]):
        sl = out[s]
        unfilled = (ctn[s] > 0.03) & (sl == 0)
        if not unfilled.any() or not (sl > 0).any():
            continue
        _, nearest = ndimage.distance_transform_edt(sl == 0, return_indices=True)
        sl[unfilled] = sl[tuple(nearest[:, unfilled])]
        out[s] = sl
    return out


class RADARInferenceError(RuntimeError):
    """RADAR推理相关异常"""


class RADARInferenceEngine:
    """真实RADAR模型推理引擎"""

    # CPU推理内存预算：
    #   峰值 = full_mask(37通道，随影像尺寸线性增长) + 滑窗瞬时激活(roi_size
    #   固定，近似恒定)。仅估算 full_mask 会严重低估峰值，导致进程 OOM 崩溃
    #   （Windows 表现为 0xC0000005 access violation，会拖垮整个 Flask 进程）。
    #   瞬时常数 4.5 GB 依据端到端实测：small 用例完整推理峰值 5.34 GB，其中
    #   full_mask 1.30 GB，隐含瞬时 4.04 GB，加安全余量取 4.5 GB。
    #   （注：孤立测量单窗前向得 6.45 GB，但该值大于包含它的完整推理峰值
    #   5.34 GB，属并发进程干扰导致的虚高，故不采用。）
    CPU_WINDOW_TRANSIENT_BYTES = 4.5 * 1024 ** 3
    # GPU 推理：前向激活在显存，但系统内存仍需承载 full_mask 与每窗回传块，
    # 且 WDDM 会把超出显存的部分溢出记账到共享内存，故提交占用远高于回传块本身。
    # 瞬时常数按「提交内存」口径端到端实测标定（本病例，10 次前向，成功跑完）：
    #   infer 起始进程提交 3.24 GB → 峰值提交 14.19 GB，净增 10.95 GB，
    #   扣除常驻 full_mask 2.11 GB 得瞬时 8.84 GB，取 9.0 GB 留余量。
    #   单窗前向内提交 6.22 → 12.80 GB（约 6.6 GB/窗），
    #   第二轮补推理的 128 窗口更大（4.15 → 12.23 GB，约 8.1 GB）。
    # 校验：提交余量 8.93 GB 时实测失败，11.24/11.42 GB 时实测成功，
    #   est_peak 11.11 GB 正好落在两者之间。
    # 注意：不可按物理内存口径的 2.0 GB 标定 —— 那会低估约 4 倍，使守卫在提交
    #   余量不足时放行，最终触发 DefaultCPUAllocator 分配失败。
    GPU_WINDOW_TRANSIENT_BYTES = 9.0 * 1024 ** 3
    CPU_MEMORY_SAFETY_RATIO = 0.9
    # GPU 按提交余量判断时安全系数取 1.0（est_peak 不超过提交余量即放行）：
    #   实测阈值恰好等于 est_peak（11.24 GB 成功 / 8.93 GB 失败），再乘 0.9 会把
    #   提交余量 11.0 GB 这类本可通过的情形误判为「影像过大」。
    #   且 GPU 路径耗尽时抛出的是可恢复 RuntimeError（返回 500 且 worker 存活），
    #   不像 CPU 路径会 0xC0000005 拖垮整个 Flask 进程，故无需额外余量。
    GPU_COMMIT_SAFETY_RATIO = 1.0

    def __init__(self, model_root=None, configs_root=None, device=None, use_gpu=True):
        self.model_root = model_root or DEFAULT_CKPT_ROOT
        self.configs_root = configs_root or self.model_root
        self.use_gpu = bool(use_gpu)
        self.device = self._resolve_device(device)

        self.model = None
        self.pad_func = None
        self.demo_module = None
        self.text_feat_dict = None

        self.organs = []
        self.test_items = []
        self.english_mapping = {}

    def _resolve_device(self, device):
        if device:
            return torch.device(device)
        if self.use_gpu and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    # ------------------------------------------------------------------ #
    # 模型加载
    # ------------------------------------------------------------------ #
    def load_organ_metadata(self):
        """仅加载器官/诊断项元数据（不加载模型权重、不初始化 CUDA context）。

        DataFolder.__init__ 只做一次 os.listdir 与常量赋值，主进程（Flask）
        可安全调用它拿到 36 个器官名供 overlay meta/mesh 路由显示，无需
        常驻数 GB 的模型权重与显存。load_model() 也复用本方法。
        """
        if self.organs and self.demo_module is not None:
            return

        # 设置环境变量后再导入，确保 XBertEncoder / RADAR 读取到正确的 ckpt 路径
        os.environ.setdefault('MODEL_ROOT', self.model_root)
        os.environ.setdefault('CONFIGS_ROOT', self.configs_root)
        os.environ['MODEL_ROOT'] = self.model_root
        os.environ['CONFIGS_ROOT'] = self.configs_root

        if RADAR_INFERENCE_PATH not in sys.path:
            sys.path.insert(0, RADAR_INFERENCE_PATH)

        try:
            import inference_demo as demo
        except Exception as exc:  # pragma: no cover - 依赖缺失时的显式错误
            raise RADARInferenceError(f'无法导入RADAR推理模块: {exc}') from exc

        self.demo_module = demo

        # 器官与诊断项来自官方 DataFolder 定义
        sample_folder = demo.DataFolder(self.model_root)
        self.organs = list(sample_folder.organs)
        self.test_items = list(sample_folder.test_items)
        self.english_mapping = dict(getattr(sample_folder, 'english_mapping', {}))

    def load_model(self):
        """加载RADAR模型（首次调用时执行，之后复用）。"""
        if self.model is not None:
            return

        self.load_organ_metadata()
        demo = self.demo_module

        self.pad_func = demo.transforms.DivisiblePadd(
            keys=['image', 'label'], k=32, mode='constant', constant_values=0, method='end'
        )

        vision_encoder = demo.VisionBranch()
        text_encoder = demo.XBertEncoder.from_config({}, from_pretrained=True)

        model = demo.RADAR(image_encoder=vision_encoder, text_encoder=text_encoder)

        ckpt_path = os.path.join(self.model_root, 'checkpoint_radar_pretrain.pth')
        if not os.path.exists(ckpt_path):
            raise RADARInferenceError(f'未找到RADAR权重文件: {ckpt_path}')

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)
        model.eval()
        model.to(self.device)

        self.model = model

        # 预加载正/负样本文本嵌入
        emb_path = os.path.join(self.model_root, 'infer_text_embedding_radar.pt')
        if os.path.exists(emb_path):
            self.text_feat_dict = torch.load(emb_path, map_location=self.device, weights_only=False)
        else:
            self.text_feat_dict = None

    # ------------------------------------------------------------------ #
    # 预处理
    # ------------------------------------------------------------------ #
    def _preprocess_single(self, image_path):
        """复用官方 DataFolder.__getitem__ 对单个 NIfTI 文件做预处理。"""
        demo = self.demo_module
        parent_dir = os.path.dirname(os.path.abspath(image_path))
        datafolder = demo.DataFolder(parent_dir)
        # 定位目标文件在目录中的索引
        file_name = os.path.basename(image_path)
        try:
            index = [os.path.basename(p) for p in datafolder.img_paths].index(file_name)
        except ValueError:
            raise RADARInferenceError(f'预处理失败，找不到文件: {image_path}')
        image_tensor, test_items, meta_info = datafolder[index]
        return image_tensor, test_items, meta_info

    # ------------------------------------------------------------------ #
    # 推理
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def infer(self, image_path, overlay_path=None):
        """对单个 NIfTI 影像执行RADAR推理。

        Args:
            image_path: NIfTI 影像路径 (.nii / .nii.gz)
            overlay_path: 可选，导出「分割叠加」数据 (.npz) 的目标路径。
                导出与 stitched_mask 逐体素对齐的预处理 CT 灰度与器官标签，
                供前端内置查看器做半透明器官叠加。为 None 时不导出。

        Returns:
            dict: 结构化诊断结果
        """
        self.load_model()

        if self.text_feat_dict is None:
            raise RADARInferenceError('缺少文本嵌入文件 infer_text_embedding_radar.pt')

        demo = self.demo_module
        start = time.time()

        image, test_items, meta_info = self._preprocess_single(image_path)
        image = image[None].to(self.device)  # [1, C, D, H, W]

        roi_size = (96, 256, 384)
        overlap = 0.25
        sw_batch_size = 1

        image_size = list(image.shape[2:])
        num_spatial_dims = len(image.shape) - 2

        # 内存保护：滑动窗口需在系统内存分配 37 通道的拼接掩码(full_mask)。
        # full_mask 现固定常驻系统内存（GPU 推理亦然，见下），因此 GPU 路径
        # 同样必须检查系统内存。峰值 = full_mask + 单窗瞬时激活。
        # 仅估算 full_mask 会严重低估，导致进程 OOM 崩溃（access violation）。
        is_cpu = self.device.type == 'cpu'
        transient = (self.CPU_WINDOW_TRANSIENT_BYTES if is_cpu
                     else self.GPU_WINDOW_TRANSIENT_BYTES)
        mask_bytes = 37 * int(np.prod(image_size)) * 4
        est_peak = mask_bytes + transient
        if is_cpu:
            # CPU 路径的前向激活也落在系统内存，峰值以工作集(物理内存)口径标定，
            # 实测本类病例需约 8 GB；不足时 Windows 直接 0xC0000005 拖垮整个
            # Flask 进程，因此必须保守地按物理内存余量判断。
            avail = available_memory_bytes()
            ratio = self.CPU_MEMORY_SAFETY_RATIO
            avail_name = '系统可用内存'
        else:
            # GPU 路径前向激活在显存，系统内存只承载 full_mask 与每窗回传块。
            # 空闲物理内存被 Windows 拿去当文件缓存、常年偏低（实测低至 0.69 GB
            # 时推理仍成功），按它判断会把可成功的推理误报为「影像过大」；
            # 真正决定分配成败的是提交余量，故这里按提交内存判断。
            avail = available_commit_bytes()
            ratio = self.GPU_COMMIT_SAFETY_RATIO
            avail_name = '系统可提交内存'
        if avail is not None:
            budget = avail * ratio
            if est_peak > budget:
                dev_name = 'CPU' if is_cpu else 'GPU'
                raise RADARInferenceError(
                    f'影像过大：{dev_name}推理系统内存峰值约需 '
                    f'{est_peak / 1024**3:.1f} GB'
                    f'（拼接掩码 {mask_bytes / 1024**3:.1f} GB + 瞬时 '
                    f'{transient / 1024**3:.1f} GB），'
                    f'而当前{avail_name}仅 {avail / 1024**3:.1f} GB。'
                    f'请关闭其他占用内存的程序，或上传较小体数据。'
                )

        scan_interval = demo._get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
        slices = demo.dense_patch_slices(image_size, roi_size, scan_interval)
        num_win = len(slices)

        fid = meta_info['file_name']
        test_organs = meta_info['test_organ_names']
        organ_feat_dict = {}
        organ_logits = dict(zip(test_items, [[] for _ in test_items]))

        # 拼接掩码(full_mask, 37通道)固定常驻系统内存：
        # GPU 推理时若将其留在显存，会与单窗前向峰值叠加超出显存容量
        # （实测 merlin 病例 649+6219+2511 = 9379 MiB > 8188 MiB 物理显存），
        # 触发 WDDM 换页后单窗前向从 2s 劣化到 45s，并最终 CUDA OOM。
        # 实测改放系统内存后：6.91s/窗、显存峰值 6385 MiB，
        # 相比 CPU 基线 46.17s/窗 提速 6.7x；每窗 PCIe 回传仅约 0.8s。
        full_mask = torch.zeros((1, 37) + tuple(image_size), device='cpu')
        # 重叠计数在37个通道上完全一致，使用单通道以显著降低内存占用，结果等价
        count_map = torch.zeros((1, 1) + tuple(image_size), device='cpu')

        for slice_g in range(0, num_win, sw_batch_size):
            slice_range = range(slice_g, min(slice_g + sw_batch_size, num_win))
            unravel_slice = [
                [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
                for idx in slice_range
            ]
            window_patches = torch.cat(
                [image[tuple(win_slice)] for win_slice in unravel_slice]
            ).to(self.device)

            organ_logits, pred_window_seg_prob = self.model.forward_test_win(
                window_patches, None, organ_logits, test_organs,
                self.text_feat_dict, organ_feat_dict, None
            )

            interpolated_seg_prob = torch.nn.functional.interpolate(
                pred_window_seg_prob, size=window_patches.shape[2:], mode='trilinear'
            )
            # full_mask/count_map 常驻系统内存，GPU 结果需先回传再累加，
            # 否则跨设备就地相加报错。回传后立即释放该窗的显存张量，
            # 避免显存碎片累积（reserved 峰值实测可达 9492 MiB）。
            interpolated_seg_prob = interpolated_seg_prob.to('cpu', non_blocking=False)
            for ii, slice_idx in enumerate(slice_range):
                full_slice = tuple(unravel_slice[ii])
                full_mask[full_slice] += interpolated_seg_prob[ii]
                count_map[full_slice] += 1
            del window_patches, pred_window_seg_prob, interpolated_seg_prob
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        count_map = torch.clamp(count_map, min=1)
        # 就地除法，避免 (full_mask / count_map) 额外分配一个同尺寸37通道张量，
        # 否则峰值内存翻倍，是CPU推理OOM崩溃的直接诱因。
        full_mask /= count_map
        stitched_mask = full_mask.argmax(1).unsqueeze(0)
        del full_mask, count_map

        # 第二轮：对被边界裁断、首轮未产出logits的器官，裁剪后补推理
        for k, v in list(organ_logits.items()):
            if len(v):
                continue
            organ_name = k.split('_')[0]
            if organ_name not in self.organs:
                continue
            organ_id = self.organs.index(organ_name)
            try:
                window_patch, window_mask = demo.center_crop(
                    image, torch.eq(stitched_mask, organ_id + 1), crop_size=roi_size
                )
            except Exception:
                continue
            if window_mask.sum() == 0:
                continue
            window_mask = window_mask.float()
            window_mask[window_mask == 1] = organ_id + 1
            pad_data = self.pad_func({'image': window_patch[0], 'label': window_mask[0]})
            window_patch, window_mask = pad_data['image'], pad_data['label']
            organ_logits, _ = self.model.forward_test_win(
                window_patch[None].to(self.device), None, organ_logits, test_organs,
                self.text_feat_dict, organ_feat_dict, None, skip_organ=organ_id
            )

        elapsed = time.time() - start
        result = self._build_result(fid, organ_logits, stitched_mask, elapsed)

        # 导出与分割逐体素对齐的叠加数据（预处理 CT 灰度 + 器官标签），
        # 供前端内置查看器做半透明器官叠加。必须在释放 image / stitched_mask 前完成。
        overlay_meta = None
        if overlay_path:
            try:
                overlay_meta = self._export_overlay(image, stitched_mask, overlay_path)
            except Exception:
                overlay_meta = None
        result['overlay'] = overlay_meta

        # 显式归还本次推理的内存。PyTorch 的缓存分配器不会把已释放的块还给系统，
        # 单次推理后 worker 会长期占住 ~13.5 GB 提交内存 / ~7.9 GB 显存，
        # 导致 available_memory_bytes() 偏低、下一次诊断被内存守卫误判为
        # 「影像过大」而失败（实测第二次调用仅剩 1.9 GB 可用）。
        del image, stitched_mask, organ_logits, organ_feat_dict
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return result

    def _export_overlay(self, image, stitched_mask, overlay_path):
        """导出器官分割叠加数据。

        image: [1, C, D, H, W]，已归一化到 [0,1]（HU 截断 [-300,400] 后 min-max）。
        stitched_mask: argmax 后的器官标签（0=背景，1..36=器官索引+1），
            由 `full_mask.argmax(1).unsqueeze(0)` 得到，实际为 [1,1,D,H,W]。

        两者处于同一「模型输入空间」，逐体素对齐；前端按切片叠加即可，
        无需再做逆变换，从根本上规避坐标错位。统一剥离前导单维到 (D,H,W)。
        导出前对 mask 做形态学后处理（去斑点/闭运算/填洞），仅改善叠加
        显示效果，不影响诊断指标（诊断在导出前已基于原始 mask 计算）。
        """
        ct_t = image.detach().cpu()
        while ct_t.dim() > 3:
            ct_t = ct_t[0]
        mask_t = stitched_mask.detach().cpu()
        while mask_t.dim() > 3:
            mask_t = mask_t[0]
        ct = ct_t.numpy()                                # (D,H,W) float [0,1]
        mask = _postprocess_overlay_mask(mask_t.numpy(), ct)  # (D,H,W) int
        ct_u8 = np.clip(ct * 255.0, 0, 255).astype(np.uint8)
        mask_u8 = mask.astype(np.uint8)
        os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
        np.savez_compressed(overlay_path, ct=ct_u8, mask=mask_u8)
        present = [int(v) for v in np.unique(mask_u8) if v > 0]
        return {
            'num_slices': int(ct_u8.shape[0]),
            'height': int(ct_u8.shape[1]),
            'width': int(ct_u8.shape[2]),
            'organ_ids': present,
        }

    # ------------------------------------------------------------------ #
    # 结果构造
    # ------------------------------------------------------------------ #
    def _severity_from_confidence(self, confidence):
        if confidence >= 0.8:
            return 'high'
        if confidence >= 0.6:
            return 'medium'
        return 'low'

    def _detected_organs(self, stitched_mask):
        """根据分割结果确定完整出现在视野内的器官。"""
        detected = []
        try:
            ids = torch.unique(stitched_mask).long().cpu().tolist()
            for organ_id in ids:
                idx = organ_id - 1
                if 0 <= idx < len(self.organs):
                    detected.append(self.organs[idx])
        except Exception:
            pass
        return detected

    def _build_result(self, patient_id, organ_logits, stitched_mask, elapsed):
        scored = []
        for item, probs in organ_logits.items():
            if not len(probs):
                continue
            confidence = float(np.concatenate(probs).mean(0)[1])
            organ, _, finding = item.partition('_')
            scored.append({
                'organ': organ,
                'finding': finding,
                'item': item,
                'confidence': confidence,
                'severity': self._severity_from_confidence(confidence),
                'finding_en': self.english_mapping.get(item, '').split('_', 1)[-1],
            })

        scored.sort(key=lambda x: x['confidence'], reverse=True)

        diagnoses = []
        for entry in scored:
            if entry['confidence'] <= 0.5:
                continue
            diagnoses.append({
                'organ': entry['organ'],
                'finding': entry['finding'],
                'item': entry['item'],
                'finding_en': entry['finding_en'],
                'confidence': round(entry['confidence'], 4),
                'severity': entry['severity'],
                'evidence': f"RADAR视觉-文本匹配对“{entry['item']}”输出阳性概率 {entry['confidence']:.1%}。",
                'recommendation': self._recommendation(entry),
            })

        organs_detected = self._detected_organs(stitched_mask)
        top_scores = [{'item': e['item'], 'confidence': round(e['confidence'], 4)} for e in scored[:10]]

        return {
            'patient_id': patient_id,
            'engine': 'RADAR (real)',
            'organs_detected': organs_detected,
            'diagnoses': diagnoses,
            'explanation': {
                'method': 'RADAR视觉分支对CT进行器官分割，逐器官提取视觉特征并与诊断项文本嵌入做匹配，得到各诊断项的阳性概率。',
                'summary': '置信度来自视觉-文本匹配的softmax阳性概率，阈值以上的结果按严重程度分级，最终仍需临床医生结合原始影像审核。',
                'device': str(self.device),
                'processing_time_sec': round(elapsed, 2),
                'total_items': len(scored),
                'top_scores': top_scores,
                'limitations': [
                    '模型输出为器官级诊断项概率，未提供像素级病灶热力图',
                    '低置信度或罕见征象可能漏检',
                    '结果不能替代临床诊断',
                ],
            },
            'raw_output': {
                'model': 'checkpoint_radar_pretrain.pth',
                'processing_time': round(elapsed, 2),
                'device': str(self.device),
            },
        }

    def _recommendation(self, entry):
        if entry['severity'] == 'high':
            return f"{entry['organ']}提示{entry['finding']}（高置信度），建议尽快由专科医生结合临床进一步检查确认。"
        if entry['severity'] == 'medium':
            return f"{entry['organ']}提示{entry['finding']}（中等置信度），建议近期复查并由专科医生评估。"
        return f"{entry['organ']}提示{entry['finding']}（低置信度），建议结合临床随访观察。"

    def format_result(self, result):
        """将诊断结果格式化为可读文本。"""
        lines = [f"患者ID: {result.get('patient_id')}", '=' * 50, '检测到的器官:']
        organs = result.get('organs_detected', [])
        lines.append('  ' + ('、'.join(organs) if organs else '无'))
        lines += ['', '=' * 50, '诊断结果:']
        diagnoses = result.get('diagnoses', [])
        if not diagnoses:
            lines.append('  未发现达到阈值的异常')
        else:
            for d in diagnoses:
                lines.append(f"  • {d['organ']} - {d['finding']} (置信度 {d['confidence']:.2%})")
        return '\n'.join(lines)


class MockInferenceEngine(RADARInferenceEngine):
    """模拟推理引擎（仅用于无模型环境下的开发测试）。"""

    def load_model(self):
        self.model = 'mock_model'

    @torch.inference_mode()
    def infer(self, image_path, overlay_path=None):
        time.sleep(0.2)
        return {
            'patient_id': Path(image_path).stem,
            'engine': 'RADAR (mock)',
            'overlay': None,
            'organs_detected': ['肝', '肾', '脾', '胰腺', '胆囊', '主动脉'],
            'diagnoses': [
                {
                    'organ': '肝', 'finding': '肝囊肿', 'confidence': 0.92, 'severity': 'low',
                    'evidence': '肝实质内见边界清晰、低密度囊性影，形态符合囊肿表现。',
                    'recommendation': '结合既往影像随访；如有增大或症状，建议专科评估。',
                },
                {
                    'organ': '肾', 'finding': '囊肿', 'confidence': 0.85, 'severity': 'low',
                    'evidence': '肾脏局灶性圆形低密度影，边缘光整，提示良性囊性改变。',
                    'recommendation': '建议结合超声或增强检查由专科医生确认。',
                },
            ],
            'explanation': {
                'method': '模拟引擎（未加载真实模型）',
                'summary': '当前为开发测试用模拟结果。',
                'limitations': ['非真实模型输出'],
            },
            'raw_output': {'segmentation_quality': 'mock', 'processing_time': 0.2},
        }


# 测试代码
if __name__ == '__main__':
    engine = RADARInferenceEngine()
    engine.load_model()
    print('RADAR模型加载成功，设备:', engine.device)
    print('诊断项数量:', len(engine.test_items))
