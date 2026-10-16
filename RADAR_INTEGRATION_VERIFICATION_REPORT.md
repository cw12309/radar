# RADAR 模型接入功能验证报告

| 项目 | 内容 |
|---|---|
| 报告日期 | 2026-10-02 |
| 验证对象 | RADAR 医学影像模型接入 `radar-web` / `flask-app` 平台 |
| 模型权重 | `ckpt/checkpoint_radar_pretrain.pth`（131,080,317 参数） |
| 验证环境 | Windows + Python 3.10.21（`D:\anaconda\envs\radar`）+ NVIDIA RTX 4060 Laptop（8188 MiB，sm_89） |
| 总体结论 | **通过**（共发现并修复 5 处缺陷，其中 1 处为阻塞级；遗留 6 项建议，见第 7 节） |

---

## 1. 结论摘要

| 验证项 | 结果 | 关键数据 |
|---|---|---|
| 模型加载与权重完整性 | 通过 | 131.1M 参数全部载入，GPU 占用 0.49 GB，加载 27s |
| CPU 推理可行性 | 通过（有限制） | small 单窗用例 65.5s 跑通、峰值 5.34 GB；merlin 真实病例完整滑窗估算峰值 9.46 GB，内存不足时被守卫拦截 |
| GPU 推理可行性 | 通过（修复后） | 修复前 CUDA OOM 崩溃；修复后 6.91s/窗、4/4 病例跑通 |
| 数值等价性（改动无副作用） | 通过 | 6/6 用例逐体素一致（含 27 窗重叠） |
| 分割精度（Dice vs 金标准） | 通过 | 4/4 病例，整体平均 **0.8405** |
| 内存守卫（防崩溃） | 通过 | 内存不足时正确拦截并给出提示，不崩溃 |
| Web 端到端（上传→诊断→报告→历史） | 通过（CPU 路径实测） | HTTP 200，PDF 5728 字节，历史记录写入成功 |
| Web 路径自动启用 GPU | 通过 | 无需设置任何环境变量，`engine.device=cuda`（引擎层实测） |

---

## 2. 验证范围与方法

### 2.1 被测入口

| 入口 | 启动方式 | 推理引擎 |
|---|---|---|
| `flask-app/app.py` | `start_radar_system.ps1` | `RADARInferenceEngine`（真实模型） |
| `radar-web/app.py` | `start_web.bat` | `RADARInferenceEngine` / `MockInferenceEngine` |
| `radar-web/app_v2.py` | `start_web_v2.bat` | 同上 |
| `radar-web/app_v3.py` | `start_web_v3.bat` | 同上 |
| `radar-web/app_v4.py` | `start_web_v4.bat` | 仅 `MockInferenceEngine` |

### 2.2 测试数据集

- **Merlin 真实病例**：`data/merlin_data_train_demo/` 共 4 例（`resized_images` + `resized_masks` 配对金标准）
- **合成用例**：small `(96,256,384)` 单窗、standard 128 张 DICOM 序列
- **演示用例**：`data/demo_cases/AC423ccbe.nii.gz`

### 2.3 方法

采用「假设 → 插桩 → 复现 → 分析 → 修复 → 验证」流程，所有性能与精度数据均为实测，原始记录保存在 `.dbg/*.json`（见第 8 节索引）。精度评估以 Merlin 官方金标准掩膜为参照，金标准施加与影像**完全相同**的预处理变换（resize / transpose / crop / pad）以保证同空间可比，标签重采样使用最近邻。

---

## 3. 缺陷与修复

### 3.1 【阻塞级】GPU 完整推理 CUDA OOM 崩溃

**现象**

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.30 GiB.
GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free.
14.63 GiB is allocated by PyTorch        ← 物理显存仅 8 GiB
```

**根因（显存账，以 merlin AC4214dbd 为例）**

| 组成 | 占用 |
|---|---|
| CUDA context + 模型权重 | 649 MiB |
| 单窗前向峰值分配 | 6219 MiB |
| 拼接掩码 `full_mask`（37 通道，常驻） | 2511 MiB |
| **合计** | **9379 MiB > 8188 MiB 物理显存** |

超出部分被 Windows WDDM 换页至系统共享内存，导致单窗前向耗时从 **2.0s 劣化到 13–45s**（GPU 利用率 97–100% 但功率仅约 25W，典型换页特征），并最终 OOM。

**修复**（`radar-web/modules/inference_engine.py`）

1. `full_mask` / `count_map` 改为固定常驻**系统内存**，GPU 仅承载单窗前向所需显存
2. 每窗推理结果经 PCIe 回传 CPU 后累加，随即释放该窗显存张量
3. 每窗结束调用 `torch.cuda.empty_cache()` 抑制显存碎片累积

**修复前后对比（实测）**

| 策略 | 每窗耗时 | GPU 峰值 alloc | 结果 |
|---|---|---|---|
| CPU 基线 | 46.17s | — | 可行但慢 |
| GPU（修复前，掩码常驻显存） | 65.5s | 14630 MiB | **OOM 崩溃** |
| **GPU（修复后，掩码放系统内存）** | **6.91s** | **6385 MiB** | **通过，提速 6.7x** |

修复后各阶段耗时分解（4 窗合计 27.65s）：前向 21.78s、插值 1.05s、PCIe 回传 3.47s、累加 1.34s。**回传开销仅每窗约 0.8s，代价可忽略。**

> **「CPU 基线 46.17s/窗」的口径说明**：该值来自 `calib_mem.json` 的**分段标定推算**（单窗前向 39.4s + 插值 6.77s），**不是 CPU 端到端完整滑窗的实测值**。CPU 端到端跑通的只有 small 单窗用例（65.5s，`calib_full_peak.json`）；merlin 真实病例的 CPU 完整滑窗估算峰值 9.46 GB 超过当时可用内存 9.09 GB，会被内存守卫拦截，因此从未端到端跑通。据此推算 merlin 4 窗 CPU 耗时约 184.7s，属**估算而非实测**。「提速 6.7x」为「GPU 实测 6.91s/窗」对比「CPU 推算 46.17s/窗」，量级可信但非同口径实测对比。

### 3.2 【一般级】内存守卫未覆盖 GPU 路径

修复 3.1 后，拼接掩码无论 CPU 还是 GPU 推理都占用**系统内存**，但原守卫仅在 `device.type == 'cpu'` 时检查，GPU 路径会绕过守卫直接崩溃。

**修复**：守卫对两种设备统一生效，并按设备区分瞬时常数（`inference_engine.py` 第 86–97 行）：

| 设备 | 瞬时常数 | 依据 |
|---|---|---|
| CPU | `CPU_WINDOW_TRANSIENT_BYTES` = 4.5 GB | 前向激活也在系统内存（实测 small 峰值 5.34 GB − 掩码 1.30 GB = 4.04 GB + 余量） |
| GPU | `GPU_WINDOW_TRANSIENT_BYTES` = 2.0 GB | 前向激活在显存，系统内存只需承载每窗回传的概率块（1426 MiB） |

**守卫实测有效性（CPU 口径，`guard_unit_test.json`，瞬时常数 4.5 GB，与最终代码一致）**

以真实调用 `infer` 守卫路径的方式，在不同可用内存下验证放行/拦截判定：

| 用例 | 峰值估算 | @8 GB 可用 | @16 GB 可用 | @32 GB 可用 |
|---|---|---|---|---|
| small 96×256×384 | 5.80 GB | 放行 | 放行 | 放行 |
| merlin 87×437×437 | 6.79 GB | 放行 | 放行 | 放行 |
| merlin 106×460×460 | 7.59 GB | **拦截** | 放行 | 放行 |
| large 400×512×512 | 18.95 GB | **拦截** | **拦截** | 放行 |

- **单调性**：可用内存越大越容易放行，判定单调（`monotonic: true`）
- **不过度保守**：内存充裕时全部放行，公式判定与真实调用一致（`not_over_conservative: true`）
- **内存不足必拦截**：`blocks_when_insufficient: true`
- **错误信息质量**：8 项检查全通过 —— 属 `RADARInferenceError`、含具体数字、说明掩码与瞬时构成、说明当前可用内存、给出可操作建议、不泄漏临时文件路径。真实消息样本（merlin 106×460×460 @8 GB）：

  > 影像过大：CPU推理峰值约需 7.6 GB（拼接掩码 3.1 GB + 滑窗瞬时 4.5 GB），而当前系统可用内存仅 8.0 GB。请启用GPU、关闭其他占用内存的程序，或上传较小体数据。

**⚠ 验证局限（须知）**

1. 上述守卫测试均在 **CPU 口径**（瞬时常数 4.5 GB）下执行。GPU 口径守卫（`GPU_WINDOW_TRANSIENT_BYTES = 2.0 GB`）虽已在代码中实现且与 CPU 分支共用同一判定逻辑与消息模板，但**尚无端到端的 GPU 拦截实测记录** —— 4 例 Dice 验证全部成功，未触发任何拦截（`dice_full_sw_gpu.json` 中无 `blocked` / `skipped` 记录）。GPU 口径守卫的正确性目前依赖代码审阅与 CPU 口径测试的类推，建议补一轮低内存下的 GPU 拦截实测。
2. 上表仅覆盖 @8 / @16 / @32 GB 三个内存点。`guard_out.txt` 另记录了 @4 GB（4 例全部拦截）与 @64 GB（4 例全部放行）两个极端点，但该次运行的瞬时常数为 **6.50 GB**（早期标定值，峰值相应偏高为 7.80 / 8.79 / 9.59 / 20.95 GB），与最终代码的 4.5 GB 不一致，故其绝对数值不可与上表混用，仅「两端行为正确」这一定性结论可参考。`guard_decision.json`（transient 5.5 GB、avail 7.38 GB、budget 6.64 GB）同属早期标定快照，且使用裁剪前原始尺寸，一并仅作演进留档。

**⚠ 守卫的副作用：会拦截此前可用的用例（行为回归风险）**

`guard_regression.json`（CPU 口径，瞬时常数 4.5 GB，可用内存 5.04 GB、预算 4.54 GB）中两个被拦截用例均标记 `would_block_previously_working = true`：

| 用例 | 尺寸 / 窗数 | 需求峰值 | 预算 | 裁决 | 此前实测 |
|---|---|---|---|---|---|
| 128 张 DICOM 序列（standard 等价） | 96×256×384 / 1 窗 | 5.80 GB | 4.54 GB | 拦截 | **曾成功跑通 508.2s**（见 6.1 节） |
| `demo_cases/AC423ccbe` | 96×360×384 / 2 窗 | 6.33 GB | 4.54 GB | 拦截 | 演示用例 |

即：同一份 128 张 DICOM 影像，在 6.1 节实测能完整跑通（当时可用内存充裕），在可用内存降至 5.04 GB 时则被守卫拦截。这**是预期的保护行为**（拦截优于崩溃），但意味着：

- 系统是否受理某次请求，**取决于当时的空闲内存**，而非影像本身的固定阈值，用户体验上会表现为「同样的文件有时能传有时不能」
- `CPU_WINDOW_TRANSIENT_BYTES = 4.5 GB` 为静态经验常数，未随窗数 / 尺寸自适应。建议后续按更多实测标定收紧，或改为「先试算实际张量分配量」而非固定常数

### 3.3 【轻微级】拼接阶段峰值内存翻倍

`(full_mask / count_map).argmax(1)` 会额外分配一个同尺寸 37 通道张量，使峰值内存翻倍，是 CPU 推理 OOM 的直接诱因。

**修复**：改为就地除法 `full_mask /= count_map`；`count_map` 由 37 通道降为单通道（重叠计数在 37 个通道上完全一致，已由 T6 验证等价）。

### 3.4 【轻微级】非元组多维索引弃用告警

`image[win_slice]` 传入 list 触发 PyTorch 弃用告警，未来版本语义将变更。**修复**：改为 `image[tuple(win_slice)]`。

### 3.5 【一般级】模块重命名导致的导入回归

引擎模块重写时删除了 `InferenceEngine` 类，但 `app.py`、`app_v2.py`、`app_v3.py` 共 4 处仍在导入该名，导致启动即 `ImportError`。

**修复**：统一改为 `RADARInferenceEngine`（构造签名兼容），全部 `py_compile` 通过。

---

## 4. 数值等价性验证

三处改动（掩码移至系统内存、就地除法、计数图单通道）必须保证**不改变推理结果**。使用合成的多窗重叠概率图复现完整拼接过程，对比改动前后算法：

| 用例 | 场景 | 结果 |
|---|---|---|
| T1 | 单窗（无重叠） | 通过，一致体素 2048/2048 |
| T2 | 27 窗重叠（考验计数归一化 + 广播） | 通过，最大重叠计数 8，一致体素 32000/32000 |
| T3 | 概率幅值放大 1000 倍（考验累加精度） | 通过，27 窗重叠 8，一致体素 32000/32000 |
| T4 | 概率幅值缩小至 1e-6（考验 argmax 稳定性） | 通过，27 窗重叠 8，一致体素 32000/32000 |
| T5 | GPU 路径 vs CPU 路径（考验设备差异） | 通过，新算法 GPU == 旧算法 GPU；旧算法 GPU == 旧算法 CPU |
| T6 | 37 通道计数逐体素相同（单通道前提） | 通过，重叠计数确 >1，广播等价成立 |

**6/6 全部通过**，改动为纯性能优化，无行为变更。

> **测试有效性的自我纠偏（须知）**：本测试首版**未能真正覆盖重叠路径**。首版 `image_size=(12,24,24)` 配 `roi=(8,16,16)`、`stride=(6,12,12)` 时，三个维度的 `range` 各自只产生 `[0]`，实际仅生成 **1 个窗**，重叠计数恒为 1 —— 此时 T2 声称的「多窗重叠」与 T6 声称的「37 通道计数相同」都是**平凡成立**，没有验证价值。
>
> 已修正为 `image_size=(20,40,40)`，实际生成 **27 个窗、最大重叠计数 8**；并在 T2/T6 中加入 `n_win > 1 and max_cnt > 1` 断言，确保测试不会再次静默退化为平凡通过。表中数据为修正后的结果。

---

## 5. 分割精度验证（GPU 完整滑窗 vs Merlin 金标准）

走**生产完整滑窗拼接路径**（即被修改的代码路径），而非简化的单窗法。

### 5.1 整体结果

**分割精度（4/4 病例，来自最终一轮评估 `dice_full_sw_gpu.json`）**

| 病例 | 影像尺寸 (D×H×W) | 形状匹配 | 评分器官 | 平均 Dice |
|---|---|---|---|---|
| AC4214dbd | 96×424×437 | 是 | 31 | 0.8223 |
| AC4240fff | 106×420×460 | 是 | 31 | 0.8354 |
| AC4242a2f | 100×430×474 | 是 | 30 | 0.8440 |
| AC4242a55 | 96×397×448 | 是 | 30 | 0.8601 |
| **整体** | — | **4/4** | **32 个器官** | **0.8405** |

**推理性能（GPU 完整滑窗实测）**

| 病例 | 推理耗时 | GPU 峰值 alloc | GPU 峰值 reserved | 检出器官 | 诊断数 | 数据来源 |
|---|---|---|---|---|---|---|
| AC4214dbd | 99.5s | 6682 MiB | 9492 MiB | 31 | 13 | `gpu_full_infer.json` |
| AC4240fff | 78.1s | 6780 MiB | 8848 MiB | 31 | 20 | `gpu_full_infer.json` |
| AC4240fff | 53.4s | 6782 MiB | — | 31 | 20 | `dice_full_sw_gpu.json`（另一次运行） |
| AC4242a2f | 171.9s | 7401 MiB | — | 30 | 41 | `dice_full_sw_gpu.json` |
| AC4214dbd / AC4242a55 | 复用缓存，无计时 | — | — | — | — | `dice_full_sw_gpu.json`（`from_cache: true`） |

> **数据口径说明**：
> 1. `dice_full_sw_gpu.json` 的 `total_infer_seconds = 225.3s` 仅等于该次运行**实际推理**的两例之和（AC4240fff 53.4s + AC4242a2f 171.9s）。AC4214dbd 与 AC4242a55 在该次运行中复用了已缓存的 `swseg_*.npy` / `swgold_*.npy`（`from_cache: true`），JSON 中**没有** `infer_seconds` 字段，因此无法从该文件取得这两例的耗时。
> 2. AC4240fff 存在两次独立运行的耗时（78.1s 与 53.4s），差异源于首次运行含模型加载与 CUDA 内核预热（`model_load_seconds` 约 27s），属正常现象。
> 3. `reserved` 峰值超过物理显存 8188 MiB，说明 PyTorch 缓存池仍有段溢出至系统共享内存，详见第 7 节遗留项。

### 5.2 逐器官平均 Dice（跨 4 例）

| 分级 | 器官（Dice） |
|---|---|
| 优秀（≥0.90） | 肝 0.977、肺 0.965、心脏 0.961、股骨 0.956、竖脊肌 0.946、髋关节 0.940、臀肌 0.938、胃 0.936、髂腰肌 0.932、主动脉 0.929、下腔静脉 0.926、骶骨 0.916、胰腺 0.904 |
| 良好（0.80–0.90） | 小肠 0.897、大肠 0.893、髂静脉 0.892、胸椎 0.874、食管 0.872、膀胱 0.867、门静脉 0.864、十二指肠 0.856、肩胛骨 0.853、腰椎 0.848、髂动脉 0.840、肾 0.835、肋骨 0.814、肾上腺 0.814 |
| 偏低（<0.80） | 脾 0.785、胆囊 0.704、肱骨 0.321 |
| 零分 | 脑 0.0、锁骨 0.0 |

### 5.3 零分器官归因（关键排查）

脑与锁骨 Dice 为 0.0，需判定是**模型固有误报**还是**拼接实现缺陷**（后者才是接入 bug）。判据与实测：

| 病例 / 器官 | 预测体素 | 金标准体素 | 交集 | 裁决 |
|---|---|---|---|---|
| AC4214dbd / 脑 | 2 | 0 | 0 | 假阳性（金标准视野外） |
| AC4214dbd / 锁骨 | 13 | 0 | 0 | 假阳性（金标准视野外） |
| AC4240fff / 脑 | 4 | 0 | 0 | 假阳性（金标准视野外） |
| AC4242a55 / 脑 | 3 | 0 | 0 | 假阳性（金标准视野外） |
| 其余各例 / 锁骨 | 0 | 0 | 0 | 双方皆空，不计入 Dice |

**裁决：疑似拼接缺陷 0 处**。脑、锁骨属腹部 CT 视野外器官，金标准中不存在（0 体素），模型产生 2–13 个体素的微量误报，属模型固有行为，与本次接入改动无关。未发现任何「金标准非空但预测为空」的漏检或「双方非空但零交集」的坐标错位。

> **裁决标准的局限（须知）**：上述判据只能捕捉「完全漏检」与「零交集坐标错位」两类**拼接实现缺陷**，无法识别「部分退化」。实测中确有大幅偏离的个例：
>
> | 病例 / 器官 | 预测体素 | 金标准体素 | Dice | 性质 |
> |---|---|---|---|---|
> | AC4242a2f / 胆囊 | 226 | 4838 | 0.0671 | 严重欠分割 |
> | AC4242a2f / 肱骨 | 1249 | 14 | 0.0095 | 严重过分割 |
>
> 两者交集均非零，故被归入 `ok`。因其余 3 例同一器官表现正常（胆囊 0.88–0.94、肱骨 0.29–0.58），可判定为**模型对该病例的个例预测偏差**，而非拼接逻辑缺陷 —— 拼接缺陷会同时影响全部病例。但严格的「非拼接缺陷」证明需叠加第 4 节的数值等价性结论（6/6 逐体素一致）共同支撑，单凭本节不足以完全排除。

### 5.4 精度结论

主体腹腔器官（肝、肺、脾、肾、胰、胃、心脏、大血管）Dice 普遍 0.87–0.98，达到可用分割质量。肱骨（0.321）、胆囊（0.704）偏低，脾（0.785）中等，且**逐病例方差较大**（胆囊 0.067–0.941、肱骨 0.0095–0.581），属模型在小目标 / 低对比 / 视野边缘器官上的固有能力边界，非接入实现问题，已列入遗留项。

---

## 6. Web 端到端验证

### 6.1 全链路（CPU 路径）

以 `flask-app` 接口实测，上传 128 张连续 DICOM 切片：

| 环节 | 状态 | 耗时 / 产出 |
|---|---|---|
| 上传解析 | HTTP 200 | 1.6s，识别为 `series` 模式，形状 128×256×256，spacing 1.0/1.0/3.0 mm |
| 诊断推理 | HTTP 200，`success=true` | 508.2s（模型 362.6s），`engine="RADAR (real)"` |
| 结果 | — | 检出 14 器官、3 项诊断（脾-脾脏淋巴瘤 70.8% 等） |
| PDF 报告 | HTTP 200 | 5596 字节 |
| 历史记录 | HTTP 200 | 累计 6 条，最新患者号 P20261001130046 |

另一轮端到端记录（`e2e_report2.json`，同为 CPU 路径）：上传 1.5s（形状 128×256×256）→ 诊断 189.7s（模型 157.52s，`use_gpu=false`、`device=cpu`）→ 16 器官 / 4 诊断 → PDF 5728 字节 → 历史 5 条，`patient_id_leaked_path=false`（患者号未泄漏文件路径），历史摘要为「脾-脾脏淋巴瘤、心脏-心包积液、脾-脾大、心脏-心影（脏）增大」。

### 6.2 GPU 自动启用验证（关键）

CUDA 版 PyTorch 安装于项目内独立目录 `.torch_cu126`（3.93 GB），原因是用户级 site-packages 目录无写权限，故采用 `pip install --target` 安装。启用方式为在 `inference_engine.py` 的 `import torch` **之前**将该目录插入 `sys.path` 首位。

**验证条件（刻意最严苛）**：清空 `PYTHONPATH`，完全复刻 `flask-app/app.py` 的 `sys.path` 设置与导入顺序。

```
PYTHONPATH = '<unset>'
torch 在 inference_engine 之前已导入? False
torch 在 file_handler 之后已导入?   False
torch  = ...\.torch_cu126\torch\__init__.py
版本   = 2.14.0+cu126 | cuda_available = True
engine.device = cuda
推理成功: 136.7s | 31 器官 | 13 诊断 | engine 报告设备 = cuda
GPU 显存峰值 6682 MiB
```

**结论：web 路径自动使用 GPU，无需任何环境变量或启动脚本改动。** 该注入点覆盖全部 5 个入口（Flask 主应用 + 4 个 Streamlit 版本），因为 `inference_engine.py` 是唯一直接导入 torch 的模块，且实测确认各入口的前置模块（`file_handler`、`image_viewer`、`numpy`、`pydicom`、`flask`）均不会先导入 torch。

`.torch_cu126` 目录不存在时注入静默跳过，自动回落到环境中已安装的 torch（CPU 亦可推理），不会因缺 CUDA 而启动失败。

---

## 7. 遗留问题与建议

| 级别 | 问题 | 说明与建议 |
|---|---|---|
| 中 | GPU reserved 峰值仍超物理显存 | 修复后 `alloc` 峰值 6385–7401 MiB 已在 8188 MiB 限内且不再 OOM，但 `reserved` 峰值达 9492 MiB，说明 PyTorch 缓存池仍有少量段溢出至系统共享内存。建议后续评估 **fp16 推理**（前向峰值 6219 MiB 可近乎减半）以彻底消除溢出并进一步提速 |
| 中 | 系统内存紧张时会被守卫拦截 | 本机 31.5 GB 总内存常被其他进程占用（实测可用一度降至 2.5 GB）。守卫会正确拦截而非崩溃，但用户需关闭其他程序后重试。建议前端将该提示转为可操作的引导 |
| 中 | Web 端到端未验证 GPU 全流程 | 6.2 节验证的是引擎层 GPU 自动启用与推理；`upload→diagnose→report→history` 全链路实测走的是 CPU 路径。建议补一轮 GPU 全链路回归 |
| 低 | 部分器官 Dice 偏低 | 肱骨 0.321、胆囊 0.704、脾 0.785。属模型固有能力边界，与接入实现无关，如需提升须重新训练或后处理 |
| 低 | 大影像推理耗时较长 | merlin 病例 GPU 完整推理最慢一例 171.9s，CPU 路径更长（端到端 508s 级，含预处理与后处理）。建议对长耗时推理采用异步任务 + 进度反馈，避免 HTTP 请求超时 |
| 低 | 路径注入的隐式前置条件 | 当前依赖「`inference_engine` 是首个导入 torch 的模块」。若将来新增模块在其之前导入 torch，`sys.modules` 缓存会使注入失效而静默回落 CPU。建议保留启动时的 `torch.__file__` 与 `cuda_available` 日志断言 |

---

## 8. 验证证据索引

所有原始实测数据位于 `.dbg/`（已加入 `.gitignore`）：

| 文件 | 内容 |
|---|---|
| `calib_full_peak.json` | CPU 基线：small 用例 65.5s、峰值 5.34 GB、隐含瞬时 4.04 GB |
| `calib_mem.json` | CPU 单窗前向 39.4s、插值 6.77s、模型加载 27.2s |
| `gpu_spill_check.json` | 显存溢出验证：context 649 MiB、前向峰值 6219 MiB、smi 占用 7875 MiB |
| `gpu_forward_perf.json` | 修复前前向耗时剧烈波动（4.1–105.6s），GPU 稳定态反慢于 CPU |
| `gpu_mask_placement.json` | 修复方案验证：6.91s/窗、alloc 峰值 6385 MiB、各阶段耗时分解 |
| `gpu_full_infer.json` | GPU 完整推理：2 例成功，31 器官 / 13 与 20 诊断 |
| `test_stitch_equiv.json` | 数值等价性 6/6 通过明细 |
| `dice_full_sw_gpu.json` | 4 例完整滑窗 Dice 逐器官明细，整体 0.8405 |
| `zero_dice_check.json` | 零分器官归因裁决：0 处疑似拼接缺陷 |
| `verify_web_gpu_live.json` | web 路径 GPU 自动启用与端到端推理验证 |
| `guard_regression.json` | 内存守卫拦截行为验证 |
| `e2e_report.json` / `e2e_report2.json` / `e2e_upload_fixed.json` | Web 全链路 HTTP 级验证记录 |
| `swseg_*.npy` / `swgold_*.npy` | 分割结果与金标准缓存，可离线复算 Dice 而无需重跑推理 |

### 验证脚本

| 脚本 | 用途 |
|---|---|
| `.dbg/test_stitch_equiv.py` | 等价性测试（无需模型，CPU 即可跑） |
| `.dbg/dice_full_sw_gpu.py` | 完整滑窗 Dice 评估（含结果缓存复用） |
| `.dbg/analyze_zero_dice.py` | 零分器官归因裁决 |
| `.dbg/gpu_mask_placement.py` | 掩码放置策略对比 |
| `.dbg/verify_web_gpu_live.py` | web 导入路径 GPU 自动启用验证 |

### 复现方式

```powershell
$py = "D:\anaconda\envs\radar\python.exe"
$env:MODEL_ROOT = "$PWD\ckpt"
$env:CONFIGS_ROOT = "$PWD\ckpt"
$env:PYTHONIOENCODING = "utf-8"
# 无需设置 PYTHONPATH：inference_engine.py 会自动注入 CUDA torch 路径
& $py ".dbg\verify_web_gpu_live.py"
```

---

## 9. 生产代码改动清单

| 文件 | 改动 |
|---|---|
| `radar-web/modules/inference_engine.py` | CUDA torch 路径注入；拼接掩码常驻系统内存；每窗回传 + `empty_cache`；就地除法；计数图单通道；元组索引修复；内存守卫扩展至 GPU 路径并新增 `GPU_WINDOW_TRANSIENT_BYTES` |
| `radar-web/app.py`、`app_v2.py`、`app_v3.py` | `InferenceEngine` → `RADARInferenceEngine`（4 处导入回归修复） |
| `.gitignore` | 新增 `.torch_cu126`、`.dbg` |

所有改动文件 `py_compile` 通过（含 4 个 Streamlit 入口与 Flask 主应用）。
