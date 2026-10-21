# -*- coding: utf-8 -*-
"""TotalSegmentator 精细分割 worker（独立环境 radar-ts 运行）。

用法: python fine_seg_worker.py <dicom_dir> <out_npz>

流程:
  1. TotalSegmentator 对原始 DICOM 序列做 104 类精细分割（单一 nii.gz，
     保留原始分辨率）；
  2. 按 RADAR 训练时的 104→36 合并表映射到 36 器官标签（与主模型本体一致，
     映射表抄自 RADAR_train/preprocess_code/process_img_mask.py）；
  3. 导出 fine_seg.npz：ct(uint8, HU[-300,400]归一化×255) / mask(uint8, 标签
     1..36) / spacing(float32, z/y/x mm)，供 Flask 叠加显示与 3D 重建使用。

仅用于显示层（2D 叠加与 3D 网格更精细），不参与诊断计算。
"""
import os
import sys

import numpy as np

# TotalSegmentator total 任务 104 类 id -> RADAR 36 器官 0-based 索引
TS104_TO_RADAR36 = {
    1: 29,                                   # spleen
    2: 19, 3: 19,                            # kidney_right/left
    4: 10,                                   # gallbladder
    5: 20,                                   # liver
    6: 30,                                   # stomach
    7: 1,                                    # aorta
    8: 18,                                   # inferior_vena_cava
    9: 23,                                   # portal_vein_and_splenic_vein
    10: 22,                                  # pancreas
    11: 0, 12: 0,                            # adrenal_gland_right/left
    13: 21, 14: 21, 15: 21, 16: 21, 17: 21,  # lung lobes
    18: 34, 19: 34, 20: 34, 21: 34, 22: 34,  # vertebrae L5..L1
    23: 35, 24: 35, 25: 35, 26: 35, 27: 35, 28: 35, 29: 35,
    30: 35, 31: 35, 32: 35, 33: 35, 34: 35,  # vertebrae T12..T1
    35: 33, 36: 33, 37: 33, 38: 33, 39: 33, 40: 33, 41: 33,  # C7..C1
    42: 7,                                   # esophagus
    43: 31,                                  # trachea
    44: 12, 45: 12, 46: 12, 47: 12, 48: 12,  # heart chambers/myocardium
    49: 24,                                  # pulmonary_artery
    50: 3,                                   # brain
    51: 15, 52: 15,                          # iliac_artery_left/right
    53: 16, 54: 16,                          # iliac_vena_left/right
    55: 28,                                  # small_bowel
    56: 6,                                   # duodenum
    57: 5,                                   # colon
    58: 25, 59: 25, 60: 25, 61: 25, 62: 25, 63: 25, 64: 25, 65: 25,
    66: 25, 67: 25, 68: 25, 69: 25, 70: 25, 71: 25, 72: 25, 73: 25,
    74: 25, 75: 25, 76: 25, 77: 25, 78: 25, 79: 25, 80: 25, 81: 25,  # ribs
    82: 14, 83: 14,                          # humerus_left/right
    84: 27, 85: 27,                          # scapula_left/right
    86: 4, 87: 4,                            # clavicula_left/right
    88: 9, 89: 9,                            # femur_left/right
    90: 13, 91: 13,                          # hip_left/right
    92: 26,                                  # sacrum
    93: 8,                                   # face
    94: 11, 95: 11, 96: 11, 97: 11, 98: 11, 99: 11,   # gluteus
    100: 2, 101: 2,                          # autochthon (竖脊肌)
    102: 17, 103: 17,                        # iliopsoas
    104: 32,                                 # urinary_bladder
}


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    dicom_dir, out_npz = sys.argv[1], sys.argv[2]

    import nibabel as nib
    import SimpleITK as sitk
    from totalsegmentator.python_api import totalsegmentator

    tmp_dir = out_npz + '.tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    seg_nii = os.path.join(tmp_dir, 'segmentation.nii.gz')

    print(f'[fine_seg] 输入: {dicom_dir}', flush=True)
    totalsegmentator(dicom_dir, seg_nii, device='cpu', fast=False, quiet=False)
    if not os.path.exists(seg_nii):
        raise RuntimeError('TotalSegmentator 未产出分割结果')

    seg = np.asarray(nib.load(seg_nii).dataobj).astype(np.int32)
    merged = np.zeros(seg.shape, dtype=np.uint8)
    present = [int(v) for v in np.unique(seg) if v > 0]
    for lab in present:
        target = TS104_TO_RADAR36.get(lab)
        if target is None:
            continue
        vol = seg == lab
        # 大器官优先占位，避免重叠标签相互覆盖
        merged[vol & (merged == 0)] = target + 1
    print(f'[fine_seg] 合并完成: TS 标签 {len(present)} 类 -> 器官 '
          f'{len(np.unique(merged)) - 1} 个', flush=True)

    # 原始 DICOM 灰度与物理间距
    series = sitk.ReadImage(dicom_dir)
    arr = sitk.GetArrayFromImage(series).astype(np.float32)  # (D,H,W) HU
    sp = series.GetSpacing()
    spacing = [float(sp[2]), float(sp[1]), float(sp[0])]     # z/y/x mm

    # HU [-300,400] 截断归一化到 uint8（与主引擎 overlay 一致）
    ct_u8 = np.clip(np.clip((arr + 300.0) / 700.0, 0, 1) * 255.0, 0, 255
                    ).astype(np.uint8)

    if merged.shape != ct_u8.shape:
        raise RuntimeError(f'分割与灰度形状不一致: {merged.shape} vs {ct_u8.shape}')

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(out_npz, ct=ct_u8, mask=merged,
                        spacing=np.array(spacing, dtype=np.float32))
    print(f'[fine_seg] 已导出 {out_npz} shape={ct_u8.shape} spacing={spacing}',
          flush=True)


if __name__ == '__main__':
    try:
        main()
    finally:
        # 清理 Flask 侧的防重复标记（成功与否都清，失败允许重试）
        marker = sys.argv[2].replace('.npz', '.running') if len(sys.argv) > 2 else None
        if marker and os.path.exists(marker):
            os.remove(marker)
