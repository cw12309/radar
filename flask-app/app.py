"""
RADAR医学影像诊断系统 - Flask后端
基于Flask + 原生JS + 现代CSS的专业医学影像平台
"""
from flask import Flask, render_template, request, jsonify, send_file, session, make_response
from flask_cors import CORS
import os
import sys
import json
import base64
import tempfile
import threading
import time
import subprocess
import shutil
import zipfile
import tarfile
from collections import OrderedDict
from pathlib import Path
from datetime import datetime
import numpy as np
from werkzeug.utils import secure_filename
import io
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage, generate_uid
# 添加模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
radar_web_dir = os.path.join(current_dir, '..', 'radar-web')
sys.path.insert(0, os.path.join(radar_web_dir, 'modules'))
sys.path.insert(0, os.path.join(radar_web_dir, 'utils'))

from file_handler import FileHandler
from inference_engine import RADARInferenceEngine, RADARInferenceError, _postprocess_overlay_mask
from pdf_generator import PDFReportGenerator
from batch_processor import BatchProcessor
from history_manager import HistoryManager

# 初始化Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'radar-medical-imaging-2026'
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['SETTINGS_FILE'] = os.path.join(current_dir, 'settings.json')
CORS(app)

# OHIF 查看器端口。默认 3001 而非 3000，因为 3000 已被本机 Docker 容器
# （Milvus 管理 GUI attu）占用，OHIF 若沿用 3000 会导致 iframe 打到别的服务上。
# 必须与 start_radar_system.ps1 中的 $OhifPort 保持一致，可用环境变量覆盖。
OHIF_PORT = int(os.environ.get('OHIF_PORT', 3001))

# 每次上传都会把体数据展开成整套 DICOM 序列写入 %TEMP%\ohif\<study_uid>\，
# 一份中等病例就是几十 MB，且原先没有任何回收逻辑（实测累积到 22 个 study、
# 1335 张 DICOM、577 MB），同时会让 /dicomweb/studies 的列表越拖越长。
# 这里按数量上限回收最旧的 study，当前正在看的那个始终保留。
MAX_OHIF_STUDIES = int(os.environ.get('MAX_OHIF_STUDIES', 5))

DEFAULT_SETTINGS = {
    'theme': 'dark',
    'language': 'zh-CN',
    'confidence_threshold': 0.7,
    'use_gpu': True,
    'viewer_source': 'local5000'
}


def read_settings():
    try:
        with open(app.config['SETTINGS_FILE'], encoding='utf-8') as f:
            settings = json.load(f)
        return {**DEFAULT_SETTINGS, **settings}
    except (OSError, json.JSONDecodeError):
        return DEFAULT_SETTINGS.copy()


def write_settings(settings):
    with open(app.config['SETTINGS_FILE'], 'w', encoding='utf-8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)

# 初始化组件
file_handler = FileHandler()
pdf_generator = PDFReportGenerator()
history_manager = HistoryManager("radar_history.db")

# RADAR推理引擎（按设置决定是否使用GPU），首次诊断时惰性加载模型
_inference_engines = {}


def get_inference_engine():
    """根据系统设置返回真实RADAR推理引擎（按 use_gpu 缓存实例）。"""
    use_gpu = bool(read_settings().get('use_gpu', True))
    engine = _inference_engines.get(use_gpu)
    if engine is None:
        engine = RADARInferenceEngine(use_gpu=use_gpu)
        _inference_engines[use_gpu] = engine
    return engine


# 器官名缓存：overlay meta/mesh 路由需要 36 个器官名做图例，但主进程不再
# 常驻模型权重。organs 是 DataFolder 里的硬编码列表，用 load_organ_metadata()
# 轻量加载即可（只 import demo、os.listdir，不碰权重与 CUDA context），
# 首次诊断后也由子进程回传刷新。
_ORGAN_NAMES = []


def get_organ_names():
    """返回 36 个器官名（供 overlay 图例），优先用缓存，缺失时轻量加载。"""
    global _ORGAN_NAMES
    if _ORGAN_NAMES:
        return _ORGAN_NAMES
    try:
        engine = get_inference_engine()
        engine.load_organ_metadata()
        _ORGAN_NAMES = list(engine.organs)
    except Exception as exc:  # noqa: BLE001 - 拿不到器官名不应让 overlay 崩
        app.logger.warning('[organs] 加载器官元数据失败: %s', exc)
    return _ORGAN_NAMES


def run_inference_subprocess(nifti_path, overlay_path=None, timeout=1800):
    """在独立子进程运行 RADAR 推理，返回 (result_dict, organs, error_str)。

    主进程不常驻模型权重（避免「常驻 4.7GB + 推理峰值 9.5GB」叠加超出系统
    提交余量）；子进程独占内存跑完整推理，结束后随进程退出释放。结果经临时
    JSON 文件回传，stderr 转发到主进程 logger。
    """
    result_dir = os.path.dirname(overlay_path) if overlay_path \
        else tempfile.gettempdir()
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, 'infer_result.json')
    if os.path.exists(result_path):
        os.remove(result_path)

    use_gpu = bool(read_settings().get('use_gpu', True))
    worker = os.path.join(current_dir, 'infer_worker.py')
    cmd = [sys.executable, worker, '--nifti', nifti_path, '--result', result_path]
    if overlay_path:
        cmd += ['--overlay', overlay_path]
    if use_gpu:
        cmd += ['--use-gpu']

    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding='utf-8',
        errors='replace', timeout=timeout, cwd=current_dir)

    if proc.stderr:
        for line in proc.stderr.strip().splitlines():
            app.logger.info('[infer-worker] %s', line)

    if not os.path.exists(result_path):
        return None, [], (proc.stderr.strip()[-500:]
                          or f'推理子进程未产出结果（exit={proc.returncode}）')
    with open(result_path, encoding='utf-8') as f:
        payload = json.load(f)
    organs = payload.get('organs') or []
    if organs:
        global _ORGAN_NAMES
        _ORGAN_NAMES = list(organs)
    if not payload.get('success'):
        return None, organs, payload.get('error', '推理失败')
    return payload.get('result'), organs, None

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'nii', 'gz', 'npy', 'dcm', 'dicom', 'png', 'jpg', 'jpeg'}
# 压缩包：从 PACS/光盘拿到的整套 DICOM 通常就是一个 zip，让用户先手动解压
# 再逐个多选几百个文件很不现实，故直接支持上传压缩包由后端解开。
# 注意不能把裸 'gz' 放进来 —— .nii.gz 是最主要的体数据格式，会被误判成压缩包。
# .tar.gz/.tgz 由 is_archive_name 的复合后缀分支整体判断。
ARCHIVE_EXTENSIONS = {'zip', 'tar', 'tgz', 'bz2'}
ARCHIVE_SUFFIXES = ('.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar')
# 压缩包内必须跳过的目录/噪声文件。
# 注意不能按扩展名白名单筛切片——从 PACS 导出的 DICOM 常常没有扩展名
# （IM0001、纯 UID 命名），必须靠文件头嗅探判断。
ARCHIVE_SKIP_NAMES = {'dicomdir', 'thumbs.db', 'desktop.ini', '.ds_store'}


def is_archive_name(filename):
    """判断文件名是否是压缩包。

    .tar.gz/.tgz 这类复合后缀必须整体判断，不能只看最后一段，
    否则 .nii.gz 会被当成压缩包。
    """
    name = filename.lower()
    if name.endswith(ARCHIVE_SUFFIXES):
        return True
    if '.' not in name:
        return False
    return name.rsplit('.', 1)[1] in ARCHIVE_EXTENSIONS


def _explicit_extension(filename):
    """返回文件名的真实扩展名（小写，不含点）；无扩展名返回 None。

    纯 UID 命名（如 '1.2.840.113619.2.55.3'）虽然含点，但末段是数字而非
    扩展名。这类 PACS 导出极为常见，必须返回 None 交给文件头嗅探，
    否则整批数据会被判成「不支持的文件格式」。
    """
    base = os.path.basename(filename)
    if '.' not in base:
        return None
    ext = base.rsplit('.', 1)[1]
    if not ext or not ext.isalpha() or len(ext) > 6:
        return None
    return ext.lower()


def allowed_file(filename):
    """检查文件扩展名是否允许。

    压缩包直接放行；无扩展名（含纯 UID 命名）也放行，真伪交给落盘后的
    文件头嗅探判定 —— 只有带明确不支持扩展名的（.txt/.pdf 等）才在此拒绝。
    """
    if is_archive_name(filename):
        return True
    ext = _explicit_extension(filename)
    if ext is None:
        return True
    return ext in ALLOWED_EXTENSIONS


def is_dicom_file(path, header_bytes=512):
    """靠文件头判断是否为 DICOM，兼容无扩展名文件。

    PACS/光盘导出的切片经常叫 IM0001 或纯 UID，没有 .dcm 后缀，单看扩展名
    会把整批数据判成「不支持的格式」。DICOM 文件在偏移 128 处有固定
    'DICM' 前缀（Preamble + Prefix），这是最可靠的判据。
    """
    try:
        with open(path, 'rb') as f:
            head = f.read(header_bytes)
    except OSError:
        return False
    if len(head) >= 132 and head[128:132] == b'DICM':
        return True
    # 无 Preamble 的裸 DICOM：直接以 (0002,xxxx) 或 (0008,xxxx) 元素开头，
    # 小端序下 tag 的字节形态为 02 00 xx xx / 08 00 xx xx。
    if len(head) >= 4 and head[0:2] in (b'\x02\x00', b'\x08\x00'):
        try:
            pydicom.dcmread(path, stop_before_pixels=True)
            return True
        except Exception:
            return False
    return False


def looks_like_dicom_name(filename):
    """按文件名粗筛「可能是 DICOM」的条目，用于压缩包解压后的过滤。

    真正的判定仍要靠文件头，这里只是先排掉明显的噪声（.xml/.txt/.jpg/
    DICOMDIR 等），避免对上千个文件逐个开文件句柄。无扩展名与纯 UID 命名
    一律放行，交给文件头嗅探。
    """
    name = filename.lower()
    base = os.path.basename(name)
    if base in ARCHIVE_SKIP_NAMES or base.startswith('._'):
        return False
    if is_archive_name(name):
        return False
    ext = _explicit_extension(name)
    if ext in {'xml', 'txt', 'csv', 'json', 'pdf', 'html', 'htm', 'rtf',
               'jpg', 'jpeg', 'png', 'gif', 'bmp', 'mp4', 'avi', 'ini',
               'dll', 'exe', 'lnk', 'nii', 'npy'}:
        return False
    return True


# 解压安全上限：防 zip bomb。整套腹部 CT 的 DICOM 一般几百 MB，
# 给到 4 GB / 5000 个文件已远超实际需要。
EXTRACT_MAX_TOTAL_BYTES = 4 * 1024 ** 3
EXTRACT_MAX_FILES = 5000


def extract_archive_to(archive_path, dest_dir):
    """解压 zip/tar 到 dest_dir，返回解出的文件绝对路径列表（已按名排序）。

    安全措施：
      - 路径穿越（zip slip）：成员路径解析后必须仍在 dest_dir 内；
      - zip bomb：累计解出体积与文件数超限即中止。
    """
    name = os.path.basename(archive_path).lower()
    extracted = []
    total = 0
    root = os.path.realpath(dest_dir)

    def _safe_target(member_name):
        """校验成员路径不逃出 dest_dir，返回安全的目标绝对路径或 None。"""
        target = os.path.realpath(os.path.join(dest_dir, member_name))
        if target == root or target.startswith(root + os.sep):
            return target
        app.logger.warning('[upload] 跳过路径穿越条目: %s', member_name)
        return None

    if name.endswith('.zip'):
        with zipfile.ZipFile(archive_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if len(extracted) >= EXTRACT_MAX_FILES:
                    raise ValueError(f'压缩包内文件数超过上限 {EXTRACT_MAX_FILES} 个')
                total += info.file_size
                if total > EXTRACT_MAX_TOTAL_BYTES:
                    raise ValueError('压缩包解压后体积超过上限 4 GB')
                target = _safe_target(info.filename)
                if target is None:
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zf.open(info) as src, open(target, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                extracted.append(target)
    else:
        mode = 'r:*'
        if name.endswith(('.tar.gz', '.tgz')):
            mode = 'r:gz'
        elif name.endswith('.tar.bz2'):
            mode = 'r:bz2'
        elif name.endswith('.tar'):
            mode = 'r:'
        with tarfile.open(archive_path, mode) as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                if len(extracted) >= EXTRACT_MAX_FILES:
                    raise ValueError(f'压缩包内文件数超过上限 {EXTRACT_MAX_FILES} 个')
                total += member.size
                if total > EXTRACT_MAX_TOTAL_BYTES:
                    raise ValueError('压缩包解压后体积超过上限 4 GB')
                target = _safe_target(member.name)
                if target is None:
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                src = tf.extractfile(member)
                if src is None:
                    continue
                with src, open(target, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                extracted.append(target)

    extracted.sort(key=lambda p: os.path.basename(p).lower())
    return extracted


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


def _safe_member_name(filename, index):
    """为上传条目生成落盘文件名。

    secure_filename 会把纯中文名/纯 UID 名清成空串（如 '1.2.840.113619' 里的点
    被当扩展名处理、中文被剥离），导致 file_id 退化成裸时间戳、后续读取失败。
    故保留序号前缀并在 secure_filename 结果无效时回退到 index 命名。
    """
    safe = secure_filename(os.path.basename(filename or ''))
    if not safe or safe.startswith('.'):
        safe = f'slice_{index:05d}'
    return f'{index:05d}_{safe}'


def _collect_dicom_series(paths):
    """从一批已落盘的文件里挑出同一个序列的 DICOM 切片，按名排序返回。

    两步筛选，都不能省：
      1. 文件名粗筛 + 头部读取。PACS/光盘导出的切片常无扩展名，同时文件夹里
         必然混有 DICOMDIR、说明文档、缩略图等噪声；用 looks_like_dicom_name
         先排掉明显噪声，再 dcmread 头部确认真的是图像对象。
      2. 按 SeriesInstanceUID 分组，只取切片数最多的那一组。一次 CT 检查通常
         含多个序列（定位像/平扫/动脉期/静脉期），它们的行列尺寸往往完全相同，
         若混拼会静默生成一个解剖上错乱的「假序列」，诊断结果全盘失真且无任何
         报错——比直接失败更危险。RADAR 只吃单一序列，故取最大的那组作为主序列。

    用 dcmread(stop_before_pixels=True) 只读头、不解码像素（每切片几百 KB），
    判别图像对象靠 Rows/Columns 属性——图像一定有，而 DICOMDIR/结构集/RT Plan 没有。
    """
    series_groups = {}
    for p in sorted(paths, key=lambda x: os.path.basename(x).lower()):
        if not os.path.isfile(p):
            continue
        if not looks_like_dicom_name(p):
            continue
        try:
            # 注意：stop_before_pixels=True 时 PixelData 属性不会加载，
            # 不能用 hasattr(ds, 'PixelData') 判断，那会把所有真实切片都排除掉。
            # 图像类对象一定有 Rows/Columns，而 DICOMDIR、结构集、RT Plan 没有，
            # 故用它作判别，同时避免解码像素（每切片几百 KB）。
            # 不用 force=True：force 会把非 DICOM 的二进制垃圾也强行解析成
            # 数据集，削弱校验；真实 PACS 导出都带规范头，标准读取即可。
            ds = pydicom.dcmread(p, stop_before_pixels=True)
        except Exception:
            # 非 DICOM（说明文档、缩略图、DB 文件等），静默跳过
            continue
        if not hasattr(ds, 'Rows') or not hasattr(ds, 'Columns'):
            continue
        if getattr(ds, 'SamplesPerPixel', 1) != 1:
            # 彩色 DICOM（超声/内镜截图等），RADAR 只支持单通道 CT
            continue
        series_uid = str(getattr(ds, 'SeriesInstanceUID', '')) or '_unknown'
        series_groups.setdefault(series_uid, []).append(p)

    if not series_groups:
        return []
    if len(series_groups) > 1:
        picked = max(series_groups, key=lambda k: (len(series_groups[k]), k))
        skipped = sorted(
            f'{uid}({len(v)}张)' for uid, v in series_groups.items() if uid != picked
        )
        app.logger.info(
            '[upload] 检出 %d 个序列，取切片数最多的 %s（%d 张），已忽略: %s',
            len(series_groups), picked, len(series_groups[picked]), ', '.join(skipped)
        )
        return series_groups[picked]
    return next(iter(series_groups.values()))


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上传单个影像（含单张DICOM切片）、组成同一序列的多张DICOM切片、
    整个文件夹（前端多选/拖拽展开），或含整套 DICOM 的压缩包。
    """
    try:
        files = request.files.getlist('files') or request.files.getlist('file')
        files = [item for item in files if item and item.filename]
        if not files:
            return jsonify({'success': False, 'message': '没有文件上传'}), 400

        # 压缩包必须单独上传（不能与切片混传），否则无法区分「一个包」与
        # 「包里若干切片 + 另外几个切片」这种语义不明的组合。
        archives = [item for item in files if is_archive_name(item.filename)]
        if archives and len(files) != 1:
            return jsonify({'success': False,
                            'message': '压缩包请单独上传，不要与其他文件混选'}), 400
        # 非压缩包文件的扩展名校验。多文件/文件夹路径放宽为「无扩展名也放行」，
        # 因为无扩展名 DICOM 极为常见，真伪交给落盘后的文件头嗅探判定。
        for item in files:
            if is_archive_name(item.filename):
                continue
            if len(files) == 1 and not allowed_file(item.filename):
                return jsonify({'success': False, 'message': '包含不支持的文件格式'}), 400

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        geometry = {}
        series_datasets = None

        if len(files) == 1 and not archives:
            filename = _safe_member_name(files[0].filename, 0).split('_', 1)[1] or 'upload'
            file_id = f'{timestamp}_{filename}'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
            files[0].save(filepath)
            # 单个无扩展名文件也可能是 DICOM（单张切片），嗅探后再决定读取路径
            if os.path.splitext(filepath)[1].lower() not in ('.nii', '.gz', '.npy', '.dcm', '.dicom', '.png', '.jpg', '.jpeg'):
                if is_dicom_file(filepath):
                    os.replace(filepath, filepath + '.dcm')
                    filepath += '.dcm'
            image_data, file_type, geometry = load_image_file_with_geometry(filepath)
            display_name = filename
            size_bytes = os.path.getsize(filepath)
            slice_count = int(image_data.shape[0]) if image_data.ndim == 3 else 1
            upload_mode = 'single'
        else:
            # 序列路径：压缩包解压 / 文件夹多选 / 直接多选切片，三者归一。
            # 一律先落到 <file_id>_stage 暂存目录，筛出目标序列后再只把选中的
            # 切片移进最终目录。原因：序列的 file_id 指向目录，而
            # /api/image/slice 等读取路径会对目录下每个文件做 dcmread，
            # 若目录里残留压缩包、DICOMDIR、说明文档等非 DICOM 文件会直接抛
            # 异常，导致上传成功却看不了片。
            file_id = f'{timestamp}_dicom_series'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
            stage_dir = filepath + '_stage'
            shutil.rmtree(stage_dir, ignore_errors=True)
            os.makedirs(stage_dir, exist_ok=True)

            dicom_paths = []
            volume_path = None      # 压缩包内只有单个 NIfTI 的情况
            source_label = None
            series_dir_created = False
            try:
                if archives:
                    source_label = archives[0].filename
                    archive_path = os.path.join(stage_dir, _safe_member_name(source_label, 0))
                    archives[0].save(archive_path)
                    extract_dir = os.path.join(stage_dir, 'extracted')
                    os.makedirs(extract_dir, exist_ok=True)
                    # extract_archive_to 递归返回包内所有文件（含嵌套子目录）
                    extracted = extract_archive_to(archive_path, extract_dir)
                    dicom_paths = _collect_dicom_series(extracted)
                    if not dicom_paths:
                        # 整套体数据打成包（单个 nii.gz）也是常见形态，走单文件读取
                        volumes = [p for p in extracted
                                   if p.lower().endswith(('.nii', '.nii.gz'))]
                        if len(volumes) != 1:
                            raise ValueError('压缩包内未找到有效的DICOM切片或NIfTI体数据')
                        volume_path = volumes[0]
                else:
                    saved_paths = []
                    for index, item in enumerate(files):
                        saved_path = os.path.join(stage_dir, _safe_member_name(item.filename, index))
                        item.save(saved_path)
                        saved_paths.append(saved_path)
                    # 文件夹里必然混有非影像文件，靠文件头嗅探筛出真正的切片；
                    # 原先按 .dcm 扩展名硬校验会把无扩展名的 PACS 导出全判失败。
                    dicom_paths = _collect_dicom_series(saved_paths)
                    if not dicom_paths:
                        raise ValueError('所选文件里没有可识别的DICOM切片'
                                         '（多文件/文件夹上传仅支持同一序列的连续DICOM）')
                    if len(saved_paths) > len(dicom_paths):
                        app.logger.info('[upload] 已忽略 %d 个非目标序列文件',
                                        len(saved_paths) - len(dicom_paths))

                if volume_path is not None:
                    final_volume = os.path.join(
                        app.config['UPLOAD_FOLDER'],
                        f'{timestamp}_{os.path.basename(volume_path)}')
                    shutil.move(volume_path, final_volume)
                    file_id = os.path.basename(final_volume)
                    filepath = final_volume
                    image_data, file_type, geometry = load_image_file_with_geometry(filepath)
                    display_name = source_label
                    size_bytes = os.path.getsize(filepath)
                    slice_count = int(image_data.shape[0]) if image_data.ndim == 3 else 1
                    upload_mode = 'single'
                else:
                    # 只把选中的序列切片移进最终目录；序号前缀既保证唯一，
                    # 也让目录遍历顺序与上传顺序一致（解剖顺序仍由
                    # load_dicom_datasets 的 SliceLocation 排序键决定）
                    os.makedirs(filepath, exist_ok=True)
                    series_dir_created = True
                    final_paths = []
                    for index, src in enumerate(dicom_paths):
                        dst = os.path.join(filepath, f'{index:05d}_{os.path.basename(src)}')
                        shutil.move(src, dst)
                        final_paths.append(dst)
                    series_datasets, image_data, geometry = load_dicom_datasets(final_paths)
                    file_type = 'dicom-series'
                    display_name = (f'{source_label}（{len(final_paths)}张切片）'
                                    if source_label else f'{len(final_paths)}张DICOM切片')
                    size_bytes = sum(os.path.getsize(p) for p in final_paths)
                    slice_count = int(image_data.shape[0])
                    upload_mode = 'series'
            except Exception:
                # 失败时别留下半成品目录，否则 file_id 会指向一个残缺序列
                if series_dir_created:
                    shutil.rmtree(filepath, ignore_errors=True)
                raise
            finally:
                shutil.rmtree(stage_dir, ignore_errors=True)

        # 提取患者/检查元数据（DICOM 头 或 NIfTI 头），供前端信息卡片展示
        metadata = {}
        try:
            if file_type == 'dicom-series' and series_datasets:
                metadata = extract_dicom_metadata(series_datasets)
            elif file_type == 'dicom':
                ds_header = pydicom.dcmread(filepath, stop_before_pixels=True)
                metadata = extract_dicom_metadata([ds_header])
            elif file_type == 'nifti':
                metadata = extract_nifti_metadata(filepath, image_data, geometry)
        except Exception:
            metadata = {}

        ohif_study_uid = export_ohif_study(file_id, image_data, file_type, geometry)
        # 导出新 study 后回收超额的历史 study，保留当前这个
        prune_ohif_studies(keep_uid=ohif_study_uid)
        session['current_file_id'] = file_id
        session['current_filepath'] = filepath
        info = {
            'filename': display_name,
            'file_type': file_type,
            'upload_mode': upload_mode,
            'uploaded_files': len(files),
            'slice_count': slice_count,
            'shape': list(image_data.shape),
            'dtype': str(image_data.dtype),
            'size_mb': round(size_bytes / 1024 / 1024, 2),
            'min_value': float(np.min(image_data)),
            'max_value': float(np.max(image_data)),
            'spacing': {
                'row': geometry.get('row_spacing', 1.0),
                'column': geometry.get('column_spacing', 1.0),
                'slice': geometry.get('slice_spacing', 1.0),
            } if geometry else None,
            'metadata': metadata,
            'ohif_study_uid': ohif_study_uid
        }
        if upload_mode == 'series':
            message = f'已导入{slice_count}张连续DICOM切片'
        elif slice_count > 1:
            message = f'已导入含{slice_count}帧的影像'
        else:
            message = '单个切片上传成功'
        return jsonify({'success': True, 'file_id': file_id, 'info': info, 'message': message})
    except ValueError as e:
        # 数据本身问题（切片尺寸不一致、彩色DICOM等）属于客户端输入错误
        return jsonify({'success': False, 'message': f'上传失败: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': f'上传失败: {str(e)}'}), 500


# 体数据缓存：/api/image/slice 与 /api/image/multiview 此前每次请求都完整解压
# 一遍 NIfTI（AC423ccbe 94MB/368层，单次解压数百毫秒），翻层时逐帧请求导致
# viewer 明显卡顿。按 (路径, mtime) 缓存解码后的体数据，LRU 上限 2 份。
_VOLUME_CACHE = OrderedDict()
_VOLUME_CACHE_MAX = 2
# app.run(threaded=True) 后多个请求会并发进入本缓存，OrderedDict 的
# move_to_end/popitem 组合不是原子的：命中与 move_to_end 之间键可能已被淘汰，
# 直接抛 KeyError 让取切片请求 500。故用锁串行化缓存本身的读写。
_VOLUME_CACHE_LOCK = threading.Lock()


def _cached_volume(filepath):
    key = (filepath, os.path.getmtime(filepath))
    with _VOLUME_CACHE_LOCK:
        hit = _VOLUME_CACHE.get(key)
        if hit is not None:
            _VOLUME_CACHE.move_to_end(key)
            return hit[0]
    # 解压放到锁外：单次耗时数百毫秒，持锁会把其他切片的请求全部堵住，
    # 并发预取时反而退化成串行。代价是同一文件可能被重复解压一次，可接受。
    volume, _ft, _geom = load_image_file_with_geometry(filepath)
    with _VOLUME_CACHE_LOCK:
        _VOLUME_CACHE[key] = (volume,)
        _VOLUME_CACHE.move_to_end(key)
        while len(_VOLUME_CACHE) > _VOLUME_CACHE_MAX:
            _VOLUME_CACHE.popitem(last=False)
    return volume


@app.route('/api/image/slice/<file_id>/<int:slice_idx>', methods=['GET'])
def get_image_slice(file_id, slice_idx):
    """
    获取影像切片
    参数:
        file_id: 文件ID
        slice_idx: 切片索引
        view: 视图方向 (axial/sagittal/coronal)
        window_center: 窗位
        window_width: 窗宽
    """
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': '文件不存在'}), 404
        
        # 获取参数
        view = request.args.get('view', 'axial')
        window_center = int(request.args.get('window_center', 40))
        window_width = int(request.args.get('window_width', 400))
        
        # 加载影像（走体数据缓存，避免逐帧请求重复解压整卷）
        image_data = _cached_volume(filepath)
        
        # 获取切片
        if len(image_data.shape) == 3:
            if view == 'axial':
                slice_data = image_data[slice_idx, :, :]
            elif view == 'sagittal':
                slice_data = image_data[:, :, slice_idx]
            elif view == 'coronal':
                slice_data = image_data[:, slice_idx, :]
            else:
                slice_data = image_data[slice_idx, :, :]
        else:
            slice_data = image_data
        
        # 应用窗位窗宽
        slice_data = apply_window(slice_data, window_center, window_width)
        
        # 转换为PNG图像
        from PIL import Image
        img = Image.fromarray(slice_data)
        
        # 转换为base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        return jsonify({
            'success': True,
            'image': img_base64,
            'shape': list(slice_data.shape)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'获取切片失败: {str(e)}'}), 500


@app.route('/api/image/multiview/<file_id>', methods=['GET'])
def get_multiview_images(file_id):
    """
    获取三视图影像
    返回轴位、矢状位、冠状位的中心切片
    """
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': '文件不存在'}), 404
        
        # 加载影像（走体数据缓存）
        image_data = _cached_volume(filepath)
        
        if len(image_data.shape) != 3:
            return jsonify({'success': False, 'message': '仅支持3D影像'}), 400
        
        # 获取中心切片
        depth, height, width = image_data.shape
        
        views = {
            'axial': image_data[depth // 2, :, :],
            'sagittal': image_data[:, :, width // 2],
            'coronal': image_data[:, height // 2, :]
        }
        
        # 转换为base64
        result = {}
        for view_name, slice_data in views.items():
            slice_data = apply_window(slice_data, 40, 400)
            
            from PIL import Image
            img = Image.fromarray(slice_data)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            result[view_name] = {
                'image': img_base64,
                'shape': list(slice_data.shape)
            }
        
        return jsonify({
            'success': True,
            'views': result,
            'dimensions': {'depth': depth, 'height': height, 'width': width}
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'获取多视图失败: {str(e)}'}), 500


# ============ 分割叠加（内置查看器）============
# 叠加数据在诊断时由推理引擎导出为 overlay.npz（与 stitched_mask 逐体素对齐的
# 预处理 CT 灰度 + 器官标签）。这里按切片懒加载，避免一次性把整卷读进内存。
_OVERLAY_CACHE = {}
_OVERLAY_CACHE_MAX = 2
_MESH_CACHE = {}
# threaded=True 下淘汰 _MESH_CACHE 的 pop(next(iter(...))) 不是原子的，
# 并发时可能弹到别的线程刚插入的键而抛 KeyError；用锁保护淘汰动作。
_MESH_CACHE_LOCK = threading.Lock()


# 36 个器官的固定配色（索引 0..35 对应器官标签 1..36，顺序与模型 organs 列表一致）。
# 人工挑选：相邻索引色相错开、明度分层，骨骼用米色系、血管用红/蓝系、
# 实质器官各占独立色相，避免黄金角 HSV 调色板在 29 个器官上出现
# 7 绿 / 6 蓝 / 5 紫几乎同色的问题。
_ORGAN_PALETTE = [
    (255, 195, 0),    # 肾上腺 金
    (215, 25, 28),    # 主动脉 绯红
    (140, 80, 40),    # 竖脊肌 棕
    (245, 130, 200),  # 脑 粉
    (235, 235, 210),  # 锁骨 象牙
    (230, 110, 20),   # 大肠 深橙
    (250, 120, 110),  # 十二指肠 鲑红
    (105, 95, 210),   # 食管 板岩蓝
    (255, 215, 175),  # 面部 桃
    (200, 170, 130),  # 股骨 褐黄
    (40, 200, 90),    # 胆囊 绿
    (110, 55, 25),    # 臀肌 深棕
    (250, 60, 60),    # 心脏 亮红
    (185, 160, 120),  # 髋关节 卡其
    (215, 200, 165),  # 肱骨 浅褐
    (170, 10, 60),    # 髂动脉 暗红
    (20, 90, 190),    # 髂静脉 蓝
    (165, 105, 50),   # 髂腰肌 中棕
    (40, 120, 220),   # 下腔静脉 中蓝
    (190, 90, 160),   # 肾 玫红
    (150, 45, 35),    # 肝 栗
    (140, 210, 245),  # 肺 天蓝
    (250, 170, 190),  # 胰腺 浅粉
    (60, 150, 250),   # 门静脉 蔚蓝
    (90, 20, 100),    # 肺动脉 暗紫
    (225, 220, 185),  # 肋骨 浅骨色
    (195, 185, 150),  # 骶骨 骨色
    (245, 240, 205),  # 肩胛骨 亮骨色
    (255, 180, 100),  # 小肠 浅橙
    (150, 50, 180),   # 脾 紫
    (210, 220, 80),   # 胃 黄绿
    (60, 180, 220),   # 气管 青
    (80, 100, 230),   # 膀胱 靛蓝
    (175, 165, 130),  # 颈椎 骨灰
    (160, 150, 115),  # 腰椎 深骨灰
    (145, 135, 100),  # 胸椎 最深骨灰
]


def _overlay_palette(n):
    """返回器官标签(1..n)到 RGB 的固定映射，超出表长时循环取模兜底。"""
    palette = {}
    for i in range(n):
        palette[i + 1] = list(_ORGAN_PALETTE[i % len(_ORGAN_PALETTE)])
    return palette


def _overlay_organ_stats(mask):
    """统计每个器官标签的 Z 轴范围与体素体积，供图例展示与切片跳转。"""
    stats = {}
    for lab in (int(v) for v in np.unique(mask) if v > 0):
        vol = mask == lab
        zs = np.where(vol.any(axis=(1, 2)))[0]
        stats[lab] = {
            'z_min': int(zs.min()),
            'z_max': int(zs.max()),
            'volume': int(vol.sum()),
        }
    return stats


def _load_overlay(file_id, mode='full'):
    """加载并缓存叠加数据，返回 (ct, mask, stats, spacing)。

    mode:
      full —— 默认。模型掩码 + 引擎后处理（幻觉门控/去斑点/闭运算/填洞/
              体内补全），历史诊断导出的旧 overlay 也能呈现完整填充；
      raw  —— 模型原始 argmax 输出（仅可靠性门控），无补全，展示真实
              预测边界；旧 overlay 无 mask_raw 字段时回退 full；
      fine —— TotalSegmentator 精细分割（原始 DICOM 分辨率，诊断后后台
              生成），不存在时返回 None。

    ct/mask 为 (D,H,W) uint8 数组，stats 为器官标签到 Z 范围/体积的映射，
    spacing 为 (z,y,x) 物理间距（mm），3D 网格生成按它套用尺度。
    后处理幂等，重复加载安全。
    """
    cache_key = (file_id, mode)
    if cache_key in _OVERLAY_CACHE:
        return _OVERLAY_CACHE[cache_key]
    infer_dir = os.path.join(
        app.config['UPLOAD_FOLDER'], 'radar_infer', file_id
    )
    if mode == 'fine':
        overlay_path = os.path.join(infer_dir, 'fine_seg.npz')
        if not os.path.exists(overlay_path):
            return None
        with np.load(overlay_path) as npz:
            ct = npz['ct']
            mask = npz['mask']
            spacing = npz['spacing'].astype(np.float64).tolist()
    else:
        overlay_path = os.path.join(infer_dir, 'overlay.npz')
        if not os.path.exists(overlay_path):
            return None
        with np.load(overlay_path) as npz:
            ct = npz['ct']
            if mode == 'raw' and 'mask_raw' in npz.files:
                mask = npz['mask_raw']
            else:
                mask = _postprocess_overlay_mask(npz['mask'], ct)
        spacing = [5.0, 1.0, 1.0]
    stats = _overlay_organ_stats(mask)
    # 限制缓存条目，避免多次诊断累积占用内存
    while len(_OVERLAY_CACHE) >= _OVERLAY_CACHE_MAX:
        _OVERLAY_CACHE.pop(next(iter(_OVERLAY_CACHE)))
    _OVERLAY_CACHE[cache_key] = (ct, mask, stats, spacing)
    return ct, mask, stats, spacing


@app.route('/api/overlay/<file_id>/meta', methods=['GET'])
def get_overlay_meta(file_id):
    """返回叠加数据的维度、器官标签清单、器官名与配色。

    支持 ?mode=full|raw|fine 选择分割来源（见 _load_overlay）；fine 不存在
    时返回 404，前端据此隐藏对应选项。
    """
    try:
        mode = request.args.get('mode', 'full')
        if mode not in ('full', 'raw', 'fine'):
            mode = 'full'
        loaded = _load_overlay(file_id, mode)
        if loaded is None:
            return jsonify({'success': False, 'message': '叠加数据不存在，请先执行AI诊断'}), 404
        ct, mask, stats, spacing = loaded
        organs = get_organ_names()
        present = sorted({int(v) for v in np.unique(mask) if v > 0})
        palette = _overlay_palette(len(organs) if organs else 36)
        organ_legend = [
            {
                'id': oid,
                'name': organs[oid - 1] if 0 < oid <= len(organs) else f'器官{oid}',
                'color': palette.get(oid, [128, 128, 128]),
                'z_min': stats.get(oid, {}).get('z_min', 0),
                'z_max': stats.get(oid, {}).get('z_max', 0),
                'volume': stats.get(oid, {}).get('volume', 0),
            }
            for oid in present
        ]
        infer_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'radar_infer', file_id)
        return jsonify({
            'success': True,
            'mode': mode,
            'num_slices': int(ct.shape[0]),
            'height': int(ct.shape[1]),
            'width': int(ct.shape[2]),
            'spacing': spacing,
            'has_raw': os.path.exists(os.path.join(infer_dir, 'overlay.npz')),
            'has_fine': os.path.exists(os.path.join(infer_dir, 'fine_seg.npz')),
            'best_slice': int(np.argmax((mask > 0).sum(axis=(1, 2)))),
            'organ_legend': organ_legend,
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'获取叠加元数据失败: {str(e)}'}), 500


@app.route('/api/overlay/<file_id>/slice/<int:slice_idx>', methods=['GET'])
def get_overlay_slice(file_id, slice_idx):
    """返回单张切片的 CT 灰度（PNG）与器官标签（原始字节 base64）。

    支持 ?mode=full|raw|fine 与 meta 一致的分割来源选择。
    """
    try:
        mode = request.args.get('mode', 'full')
        if mode not in ('full', 'raw', 'fine'):
            mode = 'full'
        loaded = _load_overlay(file_id, mode)
        if loaded is None:
            return jsonify({'success': False, 'message': '叠加数据不存在，请先执行AI诊断'}), 404
        ct, mask, _stats, _spacing = loaded
        if slice_idx < 0 or slice_idx >= ct.shape[0]:
            return jsonify({'success': False, 'message': '切片索引越界'}), 400

        from PIL import Image
        ct_slice = ct[slice_idx]
        img = Image.fromarray(ct_slice, mode='L')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        ct_b64 = base64.b64encode(buffer.getvalue()).decode()

        mask_slice = np.ascontiguousarray(mask[slice_idx], dtype=np.uint8)
        mask_b64 = base64.b64encode(mask_slice.tobytes()).decode()

        return jsonify({
            'success': True,
            'ct': ct_b64,
            'mask': mask_b64,
            'height': int(ct_slice.shape[0]),
            'width': int(ct_slice.shape[1]),
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'获取叠加切片失败: {str(e)}'}), 500


@app.route('/api/overlay/<path:file_id>/mesh', methods=['GET'])
def overlay_mesh(file_id):
    """返回各器官的 3D 表面网格（marching cubes），供前端 three.js 重建。

    支持 ?mode=full|raw|fine 选择分割来源（见 _load_overlay）：fine 为
    TotalSegmentator 原始分辨率精细分割，3D 表面最光滑。网格按数据物理
    间距（spacing，z/y/x mm）生成。为消除二值掩码直接 marching cubes 的
    阶梯状块面：先在全分辨率对器官体积场做高斯平滑（σ=1.5 体素），再降
    采样 2 倍提取等值面（实测粗糙度 14.1°→7.3°）。另从上传影像的 CT 灰度
    提取平滑的体表轮廓（id=0）作为半透明解剖背景，让 3D 视图与原始 DICOM
    影像对应。响应 gzip 压缩并缓存；顶点列序为 (z, y, x)，前端映射渲染
    坐标时套用物理间距。
    """
    try:
        mode = request.args.get('mode', 'full')
        if mode not in ('full', 'raw', 'fine'):
            mode = 'full'
        cache_key = (file_id, mode)
        payload = _MESH_CACHE.get(cache_key)
        if payload is None:
            loaded = _load_overlay(file_id, mode)
            if loaded is None:
                return jsonify({'success': False, 'message': '叠加数据不存在，请先执行AI诊断'}), 404
            ct, mask, stats, spacing = loaded
            organs = get_organ_names()
            palette = _overlay_palette(len(organs) if organs else 36)
            from skimage import measure
            from scipy import ndimage
            import gzip
            import json as _json

            down = 2
            sigma = 1.5
            meshes = []

            def _extract(vol_full):
                """全分辨率高斯平滑 → 降采样 → marching cubes，返回 (verts, faces)。"""
                smooth = ndimage.gaussian_filter(vol_full.astype(np.float32), sigma,
                                                 mode='constant')
                small = ndimage.zoom(smooth, 1.0 / down, order=1)
                verts, faces, _n, _v = measure.marching_cubes(small, level=0.5)
                return verts * down, faces

            # 体表轮廓：CT 灰度阈值化 + 填洞 + 平滑，作为解剖背景（id=0）
            try:
                body = (ct > 15).astype(np.float32)
                body = ndimage.binary_fill_holes(body).astype(np.float32)
                verts, faces = _extract(body)
                meshes.append({
                    'id': 0,
                    'name': '体表轮廓',
                    'color': [176, 190, 205],
                    'verts': [round(float(v), 1) for v in verts.ravel().tolist()],
                    'faces': [int(f) for f in faces.ravel().tolist()],
                })
            except (ValueError, RuntimeError):
                pass

            for lab in sorted(stats):
                vol = (mask == lab)
                if vol.sum() < 5:
                    continue
                try:
                    verts, faces = _extract(vol)
                except (ValueError, RuntimeError):
                    continue
                meshes.append({
                    'id': lab,
                    'name': organs[lab - 1] if 0 < lab <= len(organs) else f'器官{lab}',
                    'color': palette.get(lab, [128, 128, 128]),
                    'verts': [round(float(v), 1) for v in verts.ravel().tolist()],
                    'faces': [int(f) for f in faces.ravel().tolist()],
                })
            body = _json.dumps({'success': True, 'spacing': spacing, 'meshes': meshes})
            payload = gzip.compress(body.encode('utf-8'), 6)
            with _MESH_CACHE_LOCK:
                while len(_MESH_CACHE) >= 4:
                    _MESH_CACHE.pop(next(iter(_MESH_CACHE)))
                _MESH_CACHE[cache_key] = payload
        resp = make_response(payload)
        resp.headers['Content-Type'] = 'application/json'
        resp.headers['Content-Encoding'] = 'gzip'
        return resp
    except Exception as e:
        return jsonify({'success': False, 'message': f'网格生成失败: {str(e)}'}), 500


@app.route('/ohif/<study_uid>')
def ohif_viewer(study_uid):
    """打开本地OHIF专业影像查看器"""
    ohif_url = f"http://localhost:{OHIF_PORT}/viewer/local5000?StudyInstanceUIDs={study_uid}"
    return jsonify({'success': True, 'url': ohif_url, 'study_uid': study_uid})


_FOLDER_STUDY_CACHE = None

# DICOMweb 索引缓存：study 目录 -> {UID 映射, 已编码的头 JSON}
# 原先每个 /frames/<n> 请求都要 pydicom.dcmread 遍历整个 study（368 个文件，
# 且每个都把像素解出来）才能定位目标实例，单帧就要 2.6s；OHIF 打开 viewer 会
# 并发预取 25~100 张，实测 20 张并发耗时 37.6s，请求被拖到超时/中断，
# 前端表现为 dicomweb-client 抛 "Error: request failed"。
# 建索引只读文件头（stop_before_pixels），之后取帧退化为一次字典查表。
_DICOM_INDEX = OrderedDict()
# 容量必须能同时容纳"全部 study"，否则 /dicomweb/studies 遍历各 study 时，
# 前面刚建好的索引会被后面的挤出，下一次请求又要重建（缓存抖动）。study 总数
# 上限 = MAX_OHIF_STUDIES 个上传目录 + 1 个内置演示序列，再留 2 个余量给
# 重新上传瞬间新旧目录并存的情况。原值写死 3，在 6 个 study 时实测每轮
# /studies 都会把缓存清空（metadata 首次请求 229ms，命中后仅 41ms）。
_DICOM_INDEX_MAX = MAX_OHIF_STUDIES + 3
_DICOM_INDEX_LOCK = threading.Lock()
# 每个缓存 key 一把"重建锁"，用于把并发未命中的请求收敛成一次重建。
_DICOM_INDEX_BUILD_LOCKS = {}


def find_folder_study():
    """Find the bundled multi-slice test study without importing/copying it."""
    global _FOLDER_STUDY_CACHE
    if _FOLDER_STUDY_CACHE is not None:
        return _FOLDER_STUDY_CACHE
    study_dir = Path(current_dir).parent / 'data' / 'test_ct_dicom_series'
    files = sorted(study_dir.glob('*.dcm')) if study_dir.is_dir() else []
    if not files:
        return None, None
    dataset = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
    _FOLDER_STUDY_CACHE = (str(dataset.StudyInstanceUID), str(dataset.SeriesInstanceUID))
    return _FOLDER_STUDY_CACHE


def get_dicom_study_dir(study_uid):
    upload_dir = Path(app.config['UPLOAD_FOLDER']) / 'ohif' / study_uid
    if upload_dir.is_dir():
        return upload_dir
    folder_uid, _ = find_folder_study()
    if folder_uid == study_uid:
        return Path(current_dir).parent / 'data' / 'test_ct_dicom_series'
    return None


def dicom_files(study_dir):
    return sorted(path for path in study_dir.glob('*.dcm')) if study_dir and study_dir.is_dir() else []


def dicom_json(dataset):
    return dataset.to_json_dict()


def _study_dir_stamp(study_dir):
    """目录指纹：文件数 + 最新 mtime。

    重新上传（新目录）或 prune_ohif_studies 回收后指纹会变，缓存自然失效，
    不需要显式清理，也不会读到被删目录的旧数据。

    必须用 os.scandir 而不是 Path.glob + 逐个 stat：本函数在每个 DICOMweb
    请求（含每张切片的 /frames/1）的热路径上都要跑一次，368 个文件的
    glob+getmtime 实测 387ms，而 scandir 复用目录项自带的 stat 结果只要
    1.6ms。OHIF 并发预取上百张时，这 385ms 的差距直接决定请求是否超时。
    """
    count = 0
    newest = 0.0
    try:
        with os.scandir(study_dir) as it:
            for entry in it:
                if not entry.name.endswith('.dcm'):
                    continue
                count += 1
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    continue
                if mtime > newest:
                    newest = mtime
    except OSError:
        return None
    if count == 0:
        return None
    return (count, newest)


def get_dicom_index(study_uid):
    """返回 study 的 DICOM 索引；目录不存在或为空时返回 None。

    索引结构：
      by_uid      (series_uid, sop_uid) -> (文件路径, 头 JSON)
      series_json series_uid -> 头 JSON（与旧实现一致：同序列后者覆盖前者）
      series_meta series_uid -> [头 JSON, ...]（该序列全部实例）
      study_json  首个实例的头 JSON（study 级摘要）
      all_meta    [头 JSON, ...]（study 内全部实例，study 级 metadata 用）
    """
    study_dir = get_dicom_study_dir(study_uid)
    if study_dir is None:
        return None
    stamp = _study_dir_stamp(study_dir)
    if stamp is None:
        return None
    key = (str(study_dir), stamp)

    with _DICOM_INDEX_LOCK:
        cached = _DICOM_INDEX.get(key)
        if cached is not None:
            _DICOM_INDEX.move_to_end(key)
            return cached
        # 未命中：为这个 key 取一把专属重建锁。OHIF 打开 viewer 会并发预取
        # 25~100 张帧，这些请求的 key 完全相同；若各自往下走，就会同时把同一个
        # study 的索引重建 N 遍（128 个文件 × 25 次 dcmread），实测 25 帧并发
        # 要 6.0s，请求被拖到浏览器超时 -> dicomweb-client 抛 "request failed"。
        # 有了这把锁，只有第一个线程真正重建，其余线程排队后直接命中缓存。
        build_lock = _DICOM_INDEX_BUILD_LOCKS.setdefault(key, threading.Lock())

    with build_lock:
        # 双重检查：排队期间可能已被前一个持锁线程建好
        with _DICOM_INDEX_LOCK:
            cached = _DICOM_INDEX.get(key)
            if cached is not None:
                _DICOM_INDEX.move_to_end(key)
                return cached

        index = {'by_uid': {}, 'series_json': {}, 'series_meta': {}, 'study_json': None, 'all_meta': []}
        for path in dicom_files(study_dir):
            try:
                ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            except Exception as exc:  # noqa: BLE001 - 单个坏文件不应拖垮整个 study
                app.logger.warning('[dicomweb] 跳过无法解析的 DICOM %s: %s', path.name, exc)
                continue
            meta = dicom_json(ds)
            series_uid = str(ds.SeriesInstanceUID)
            sop_uid = str(ds.SOPInstanceUID)
            index['by_uid'][(series_uid, sop_uid)] = (path, meta)
            index['series_json'][series_uid] = meta
            index['series_meta'].setdefault(series_uid, []).append(meta)
            index['all_meta'].append(meta)
            if index['study_json'] is None:
                index['study_json'] = meta

        if not index['by_uid']:
            with _DICOM_INDEX_LOCK:
                _DICOM_INDEX_BUILD_LOCKS.pop(key, None)
            return None

        with _DICOM_INDEX_LOCK:
            _DICOM_INDEX[key] = index
            _DICOM_INDEX.move_to_end(key)
            while len(_DICOM_INDEX) > _DICOM_INDEX_MAX:
                stale_key, _ = _DICOM_INDEX.popitem(last=False)
                # 淘汰缓存项的同时回收其重建锁，避免长时间运行后字典无界增长
                if stale_key != key:
                    _DICOM_INDEX_BUILD_LOCKS.pop(stale_key, None)
        return index


@app.route('/dicomweb/studies', methods=['GET'])
def dicomweb_studies():
    """Provide uploaded studies and the bundled multi-slice test series to OHIF."""
    study_uid = request.args.get('StudyInstanceUID')
    studies = []
    folder_uid, _ = find_folder_study()
    candidate_uids = [study_uid] if study_uid else [folder_uid]
    upload_root = Path(app.config['UPLOAD_FOLDER']) / 'ohif'
    if not study_uid and upload_root.is_dir():
        candidate_uids.extend(path.name for path in upload_root.iterdir() if path.is_dir())
    for uid in dict.fromkeys(uid for uid in candidate_uids if uid):
        index = get_dicom_index(uid)
        if index is not None:
            studies.append(index['study_json'])
    return jsonify(studies)


@app.route('/dicomweb/studies/<study_uid>/series', methods=['GET'])
def dicomweb_series(study_uid):
    index = get_dicom_index(study_uid)
    if index is None:
        # study 不存在（多为被 prune_ohif_studies 回收）时必须返回 404，
        # 不能返回 200 + []：OHIF 收到 200 会认为"study 存在但无内容"，
        # 继续构建 display set，最终在路由层抛异常，viewer 整页停在
        # "Error in Route /viewer/local5000"；返回 404 才能被 dicomweb-client
        # 正常捕获并提示 study 不可用。
        return jsonify({'error': 'study not found',
                        'detail': 'study 已不存在，可能被历史 study 回收清理，'
                                  '请重新上传后再打开阅片'}), 404
    return jsonify(list(index['series_json'].values()))


@app.route('/dicomweb/studies/<study_uid>/metadata', methods=['GET'])
def dicomweb_study_metadata(study_uid):
    """Study 级 metadata：返回该 study 内全部实例的头信息。

    DICOMweb 规范要求此端点存在；此前未实现直接 404，而 OHIF 在
    enableStudyLazyLoad=false 的取数路径（retrieveMetadataLoaderSync ->
    client.retrieveStudyMetadata）上必打这个 URL，dicomweb-client 收到
    非 2xx 就抛 "Error: request failed"，viewer 停在加载态。
    """
    index = get_dicom_index(study_uid)
    if index is None:
        return jsonify({'error': 'study not found'}), 404
    return jsonify(index['all_meta'])


@app.route('/dicomweb/studies/<study_uid>/series/<series_uid>/metadata', methods=['GET'])
def dicomweb_series_metadata(study_uid, series_uid):
    index = get_dicom_index(study_uid)
    if index is None:
        return jsonify({'error': 'study not found'}), 404
    # study 在、但该 series_uid 不属于它：这是合法的空查询结果，返回 200 + []
    return jsonify(index['series_meta'].get(series_uid, []))


@app.route('/dicomweb/studies/<study_uid>/series/<series_uid>/instances', methods=['GET'])
def dicomweb_instances(study_uid, series_uid):
    index = get_dicom_index(study_uid)
    if index is None:
        return jsonify({'error': 'study not found'}), 404
    return jsonify(index['series_meta'].get(series_uid, []))


@app.route('/dicomweb/studies/<study_uid>/series/<series_uid>/instances/<instance_uid>/frames/<int:frame_number>', methods=['GET'])
def dicomweb_instance_frame(study_uid, series_uid, instance_uid, frame_number):
    if frame_number != 1:
        return jsonify({'error': 'frame not found'}), 404
    index = get_dicom_index(study_uid)
    if index is None:
        return jsonify({'error': 'frame not found'}), 404
    entry = index['by_uid'].get((series_uid, instance_uid))
    if entry is None:
        return jsonify({'error': 'frame not found'}), 404
    # 只解这一个文件的像素（索引阶段用的是 stop_before_pixels，不含像素）
    ds = pydicom.dcmread(str(entry[0]))
    return send_file(io.BytesIO(ds.PixelData), mimetype='application/octet-stream')


@app.route('/dicomweb/studies/<study_uid>/series/<series_uid>/instances/<instance_uid>/metadata', methods=['GET'])
@app.route('/dicomweb/studies/<study_uid>/series/<series_uid>/instances/<instance_uid>', methods=['GET'])
def dicomweb_instance(study_uid, series_uid, instance_uid):
    index = get_dicom_index(study_uid)
    if index is None:
        return jsonify({'error': 'instance not found'}), 404
    entry = index['by_uid'].get((series_uid, instance_uid))
    if entry is None:
        return jsonify({'error': 'instance not found'}), 404
    path, meta = entry
    if request.path.endswith('/metadata'):
        return jsonify([meta])
    return send_file(path, mimetype='application/dicom')


def prune_ohif_studies(keep_uid=None):
    """回收 %TEMP%\\ohif 下的历史 study 目录，最多保留 MAX_OHIF_STUDIES 个。

    按目录修改时间从新到旧排序，删除超额的旧目录；keep_uid（本次刚导出的
    study）无条件保留，避免用户上传完立刻打开 viewer 却发现数据被删。
    内置的 data/test_ct_dicom_series 不在此目录下，不受影响。
    删除失败（文件被占用等）只跳过，不影响上传主流程。
    """
    import shutil

    root = Path(app.config['UPLOAD_FOLDER']) / 'ohif'
    if not root.is_dir():
        return
    dirs = [p for p in root.iterdir() if p.is_dir() and p.name != keep_uid]
    if len(dirs) <= max(0, MAX_OHIF_STUDIES - 1):
        return
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in dirs[max(0, MAX_OHIF_STUDIES - 1):]:
        try:
            shutil.rmtree(stale)
        except OSError as exc:
            app.logger.warning('[ohif] 清理历史 study 失败 %s: %s', stale.name, exc)


def _pick_window(data):
    """按数据实际值域挑选 DICOM 窗位/窗宽，返回 (WindowCenter, WindowWidth)。

    - 真实 CT（HU 值域宽，含 -1024 附近的空气）用腹部软组织窗 WL=40/WW=400，
      这是 OHIF 打开 CT 的常规默认，组织对比最好；
    - 窄值域数据（如器官标签 mask，值域 0~36）若沿用软组织窗，36 全部落在
      窗内偏下、几乎全黑，故直接按数据的 1%/99% 分位撑满灰阶；
    - 分位数而非 min/max，是为了避开个别异常体素把窗宽拉得过宽。
    DICOM 要求 WindowWidth 为正数，故下限取 1。
    """
    values = np.asarray(data, dtype=np.float32).ravel()
    if values.size == 0:
        return 40, 400
    # 大体数据（如 87x388x388 = 1310 万体素）上做分位数要排序整个副本，
    # 会明显拖慢上传，故抽样到百万级即可，对 1%/99% 分位的估计足够准。
    if values.size > 1_000_000:
        values = values[:: max(1, values.size // 1_000_000)]
    lo, hi = float(np.percentile(values, 1)), float(np.percentile(values, 99))
    span = hi - lo
    # 判据：值域跨度足够大且下探到 CT 的空气/骨密度区间，才认为是真实 CT
    if span >= 500 and lo <= -100:
        return 40, 400
    width = max(1.0, span)
    center = (hi + lo) / 2.0
    return center, width


def export_ohif_study(file_id, image_data, file_type, geometry=None):
    """将上传影像导出为OHIF可加载的DICOM序列，保留真实体素间距。"""
    data = np.asarray(image_data)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    geometry = geometry or {}
    row_spacing = float(geometry.get('row_spacing', 1.0))
    column_spacing = float(geometry.get('column_spacing', 1.0))
    slice_spacing = float(geometry.get('slice_spacing', 1.0))
    study_uid = generate_uid()
    series_uid = generate_uid()
    out_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'ohif', study_uid)
    os.makedirs(out_dir, exist_ok=True)

    # 窗宽窗位：不写这两个标签时 OHIF 只能用自动/默认窗，而 CT 软组织窗
    # (WL=40, WW=400) 套到窄值域数据上会几乎全黑（例如器官标签 mask 值域仅
    # 0~36，落在窗内的只有 33 个灰度级），viewer 看上去是一片死黑。
    window_center, window_width = _pick_window(data)

    for index, slice_data in enumerate(data):
        # HU值为浮点，四舍五入后再存入int16，避免直接截断造成灰度偏差
        pixels = np.rint(np.asarray(slice_data, dtype=np.float32)).astype(np.int16)
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b'\x00' * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.Modality = 'CT'
        ds.PatientName = 'RADAR^Demo'
        ds.PatientID = 'RADAR-DEMO'
        ds.InstanceNumber = index + 1
        ds.Rows, ds.Columns = pixels.shape
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.SamplesPerPixel = 1
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleIntercept = 0
        ds.RescaleSlope = 1
        ds.PixelSpacing = [row_spacing, column_spacing]
        ds.SliceThickness = slice_spacing
        ds.SpacingBetweenSlices = slice_spacing
        # 沿Z轴按真实层间距定位，保证OHIF的MPR重建与窗宽窗距正确
        ds.ImagePositionPatient = [0.0, 0.0, float(index) * slice_spacing]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.SliceLocation = float(index) * slice_spacing
        ds.WindowCenter = window_center
        ds.WindowWidth = window_width
        ds.PixelData = pixels.tobytes()
        ds.save_as(os.path.join(out_dir, f'{index:05d}.dcm'))
    with open(os.path.join(out_dir, 'series.uid'), 'w', encoding='utf-8') as f:
        f.write(series_uid)
    return study_uid


def find_series_uid(study_dir):
    try:
        with open(os.path.join(study_dir, 'series.uid'), encoding='utf-8') as f:
            return f.read().strip()
    except OSError:
        return None


def _spawn_fine_segmentation(file_id, dicom_dir):
    """后台拉起 radar-ts 环境的 TotalSegmentator worker 生成精细分割。

    默认关闭：CPU 推理单次需数十分钟，需设置环境变量
    RADAR_ENABLE_FINE_SEG=1 才在诊断后自动触发。独立 conda 环境（radar-ts）
    运行，与主推理环境隔离；detached 子进程不阻塞诊断返回。用
    fine_seg.running 标记防重复拉起，日志写入 infer 目录。
    """
    if os.environ.get('RADAR_ENABLE_FINE_SEG', '0') != '1':
        return
    infer_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'radar_infer', file_id)
    out_npz = os.path.join(infer_dir, 'fine_seg.npz')
    marker = os.path.join(infer_dir, 'fine_seg.running')
    if os.path.exists(out_npz) or os.path.exists(marker):
        return
    ts_python = os.path.join(os.path.expanduser('~'), '.conda', 'envs',
                             'radar-ts', 'python.exe')
    worker = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'radar-web', 'modules', 'fine_seg_worker.py')
    if not (os.path.exists(ts_python) and os.path.exists(worker)):
        return
    try:
        with open(marker, 'w', encoding='utf-8') as fh:
            fh.write('running')
        log = open(os.path.join(infer_dir, 'fine_seg.log'), 'w', encoding='utf-8')
        env = dict(os.environ, PYTHONNOUSERSITE='1')
        creationflags = 0x00000008 | 0x08000000  # DETACHED_PROCESS | CREATE_NO_WINDOW
        subprocess.Popen(
            [ts_python, worker, dicom_dir, out_npz],
            stdout=log, stderr=subprocess.STDOUT, env=env,
            creationflags=creationflags,
        )
    except Exception:
        if os.path.exists(marker):
            os.remove(marker)


@app.route('/api/diagnose/<file_id>', methods=['POST'])
def diagnose(file_id):
    """
    执行AI诊断
    """
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': '文件不存在'}), 404

        # 加载影像（同时取得真实体素间距，供RADAR重采样使用）
        image_data, file_type, geometry = load_image_file_with_geometry(filepath)

        if np.asarray(image_data).ndim != 3:
            return jsonify({'success': False, 'message': 'RADAR诊断需要三维CT影像（请上传体数据或连续DICOM序列）'}), 400

        # 将影像准备为RADAR所需的独立NIfTI文件
        nifti_path = prepare_nifti_for_inference(file_id, filepath, image_data, geometry)

        # 分割叠加数据与 NIfTI 同目录存放，供 /api/overlay/<file_id> 读取
        overlay_path = os.path.join(
            app.config['UPLOAD_FOLDER'], 'radar_infer', file_id, 'overlay.npz'
        )
        # 在独立子进程执行推理：主进程不常驻模型权重，避免常驻内存与推理
        # 峰值叠加超出系统提交余量（实测独立进程可成功、常驻进程被守卫拦截）。
        result, _organs, err = run_inference_subprocess(nifti_path, overlay_path)
        if err:
            status = 503 if '过大' in err or '内存' in err or '显存' in err else 500
            return jsonify({'success': False, 'message': f'诊断失败: {err}'}), status

        settings = read_settings()
        threshold = settings['confidence_threshold']
        result['diagnoses'] = [
            diagnosis for diagnosis in result.get('diagnoses', [])
            if float(diagnosis.get('confidence', 0)) >= threshold
        ]
        result['settings'] = {
            'confidence_threshold': threshold,
            'use_gpu': settings['use_gpu']
        }

        # 提取患者/检查元数据（DICOM 头 或 NIfTI 头），随诊断结果一并返回，
        # 供结果区「影像与检查信息」卡片展示；DICOM PatientID 优先作为患者标识。
        # 元数据只取首帧文件头（stop_before_pixels），避免为拿头信息重复读取整卷像素。
        metadata = {}
        try:
            if file_type == 'dicom-series' and os.path.isdir(filepath):
                dicom_paths = sorted(
                    str(p) for p in Path(filepath).iterdir() if p.is_file()
                )
                if dicom_paths:
                    ds_header = pydicom.dcmread(dicom_paths[0], stop_before_pixels=True)
                    metadata = extract_dicom_metadata([ds_header])
                    metadata['num_instances'] = len(dicom_paths)
            elif file_type == 'dicom':
                ds_header = pydicom.dcmread(filepath, stop_before_pixels=True)
                metadata = extract_dicom_metadata([ds_header])
            elif file_type == 'nifti':
                metadata = extract_nifti_metadata(filepath, image_data, geometry)
        except Exception:
            metadata = {}
        result['metadata'] = metadata

        # 患者ID：优先用 DICOM 真实 PatientID，缺失时回退到时间戳生成的占位ID
        patient_id = metadata.get('patient_id') or f"P{datetime.now().strftime('%Y%m%d%H%M%S')}"
        # 引擎内部使用的是临时NIfTI路径，这里统一替换为面向用户的患者ID
        result['patient_id'] = patient_id
        case_id = history_manager.add_case(
            patient_id=patient_id,
            file_name=os.path.basename(filepath),
            file_path=filepath,
            diagnosis_result=result,
            metadata={'image_shape': list(image_data.shape), 'file_type': file_type}
        )
        
        # 保存到session
        session['current_diagnosis'] = result
        session['current_case_id'] = case_id

        # 后台生成 TotalSegmentator 精细分割（仅 DICOM 序列，独立环境运行，
        # 不阻塞诊断返回；完成后前端「分割来源」出现精细分割选项）。
        if file_type == 'dicom-series':
            _spawn_fine_segmentation(file_id, filepath)

        return jsonify({
            'success': True,
            'result': result,
            'case_id': case_id,
            'message': 'AI诊断完成'
        })
        
    except RADARInferenceError as e:
        return jsonify({'success': False, 'message': f'诊断失败: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': f'诊断失败: {str(e)}'}), 500


def prepare_nifti_for_inference(file_id, filepath, image_data, geometry=None):
    """将上传影像转换为RADAR推理所需的独立NIfTI文件，并保留真实体素间距。

    为每份影像创建一个仅含该文件的目录，避免 DataFolder 扫描到其他无关文件。

    RADAR 预处理会依据 NIfTI 头中的 spacing 重采样到参考间距 (1, 1, 5) mm。
    若此处丢失真实间距（退化为 1mm），Z 轴重采样目标尺寸会偏差数倍，直接影响
    诊断准确率，因此必须把 DICOM 的真实间距写入输出文件。
    """
    import SimpleITK as sitk

    infer_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'radar_infer', file_id)
    os.makedirs(infer_dir, exist_ok=True)
    nifti_path = os.path.join(infer_dir, f'{Path(filepath).stem}.nii.gz')
    if os.path.exists(nifti_path):
        return nifti_path

    # 已是NIfTI，直接复制到独立目录（其affine已含正确spacing）
    if filepath.endswith(('.nii', '.nii.gz')) and os.path.isfile(filepath):
        import shutil
        shutil.copy(filepath, nifti_path)
        return nifti_path

    volume = np.asarray(image_data, dtype=np.float32)
    sitk_image = sitk.GetImageFromArray(volume)
    geometry = geometry or {}
    if geometry:
        # volume 轴序为 (Z, Y, X)，SimpleITK 的 spacing 顺序为 (X, Y, Z)
        sitk_image.SetSpacing((
            float(geometry.get('column_spacing', 1.0)),
            float(geometry.get('row_spacing', 1.0)),
            float(geometry.get('slice_spacing', 1.0)),
        ))
    sitk.WriteImage(sitk_image, nifti_path)
    return nifti_path


@app.route('/api/settings', methods=['GET', 'POST', 'PUT'])
def settings_api():
    if request.method == 'GET':
        return jsonify({'success': True, 'settings': read_settings()})
    try:
        payload = request.get_json(silent=True) or {}
        settings = read_settings()
        if 'theme' in payload and payload['theme'] in {'dark', 'light'}:
            settings['theme'] = payload['theme']
        if 'language' in payload and payload['language'] in {'zh-CN', 'en'}:
            settings['language'] = payload['language']
        if 'confidence_threshold' in payload:
            threshold = float(payload['confidence_threshold'])
            if not 0 <= threshold <= 1:
                raise ValueError('置信度阈值必须在0到1之间')
            settings['confidence_threshold'] = threshold
        if 'use_gpu' in payload:
            settings['use_gpu'] = bool(payload['use_gpu'])
        if 'viewer_source' in payload and payload['viewer_source'] in {'local5000'}:
            settings['viewer_source'] = payload['viewer_source']
        write_settings(settings)
        return jsonify({'success': True, 'settings': settings, 'message': '设置已保存'})
    except (TypeError, ValueError, OSError) as e:
        return jsonify({'success': False, 'message': f'保存设置失败: {str(e)}'}), 400


def _heat_color(t):
    """置信度 0→1 映射为 蓝→青→绿→黄→红，与前端热力图例一致。"""
    c = min(1.0, max(0.0, float(t)))
    stops = [
        (0.0, (0, 64, 255)),
        (0.25, (0, 220, 255)),
        (0.5, (80, 255, 80)),
        (0.75, (255, 230, 0)),
        (1.0, (255, 40, 0)),
    ]
    for i in range(1, len(stops)):
        if c <= stops[i][0]:
            (p0, c0), (p1, c1) = stops[i - 1], stops[i]
            k = (c - p0) / (p1 - p0)
            return tuple(int(a + (b - a) * k) for a, b in zip(c0, c1))
    return stops[-1][1]


def _render_evidence_slice(ct, mask, stats, organ_names, palette,
                           organ_conf, slice_idx, mode):
    """把单张切片渲染成 PNG 字节（分割填充 或 注意力热力）。

    mode='seg'  : 器官按固定配色半透明填充（轮廓加深）
    mode='heat' : 仅受累器官，按诊断置信度走热力色，边缘用距离变换羽化
    """
    from PIL import Image as PILImage
    from scipy import ndimage

    gray = ct[slice_idx]
    seg = mask[slice_idx]
    rgb = np.dstack([gray] * 3).astype(np.float32)

    labels = [lab for lab in (stats or {}).keys()
              if mode == 'seg' or organ_conf.get(organ_names.get(lab, '')) is not None]
    if not labels:
        buf = io.BytesIO()
        PILImage.fromarray(rgb.astype(np.uint8), mode='RGB').save(buf, format='PNG')
        return buf.getvalue()

    if mode == 'heat':
        # 距离变换：器官内部核心浓、边缘淡，避免色块生硬切割 CT 纹理
        dist = np.zeros(seg.shape, dtype=np.float32)
        for lab in labels:
            vol = seg == lab
            if not vol.any():
                continue
            d = ndimage.distance_transform_edt(vol)
            d = np.minimum(d / 6.0, 1.0)
            dist = np.maximum(dist, np.where(vol, d, 0.0))

    alpha = 0.5
    for lab in labels:
        vol = seg == lab
        if not vol.any():
            continue
        if mode == 'heat':
            name = organ_names.get(lab, '')
            conf = organ_conf.get(name)
            if conf is None:
                continue
            color = np.array(_heat_color(conf), dtype=np.float32)
            a = alpha * (0.45 + 0.55 * dist) * vol
            a = a[..., None]
        else:
            color = np.array(palette.get(lab, [128, 128, 128]), dtype=np.float32)
            a = np.where(vol, alpha, 0.0)[..., None]
        rgb = rgb * (1 - a) + color * a

    out = np.clip(rgb, 0, 255).astype(np.uint8)
    # 器官轮廓描边，让边界在报告缩略图上依然可辨
    if mode == 'seg':
        for lab in labels:
            vol = (seg == lab).astype(np.uint8)
            if not vol.any():
                continue
            edge = vol ^ ndimage.binary_erosion(vol, np.ones((2, 2), bool)).astype(np.uint8)
            out[edge > 0] = palette.get(lab, [128, 128, 128])

    buf = io.BytesIO()
    PILImage.fromarray(out, mode='RGB').save(buf, format='PNG')
    return buf.getvalue()


def _build_report_evidence(file_id, diagnosis_result, limit=4):
    """生成 PDF「影像证据」截图：受累器官的分割叠加 + 注意力热力对照。

    每张图选在对应器官体素最多的层面（证据最充分），标题带器官名、发现、
    置信度与层号，医师可据此在原始序列上定位复核。渲染失败不影响报告生成。
    """
    if not file_id:
        return []
    try:
        loaded = _load_overlay(file_id)
        if loaded is None:
            return []
        ct, mask, stats, _spacing = loaded
        organ_names = {i + 1: n for i, n in enumerate(get_organ_names() or [])}
        palette = _overlay_palette(len(organ_names) or 36)
        diagnoses = diagnosis_result.get('diagnoses') or []

        # 器官名 -> 该器官最高置信度诊断项
        organ_conf, best_diag = {}, {}
        for d in diagnoses:
            organ, conf = d.get('organ'), float(d.get('confidence') or 0)
            if organ is None:
                continue
            if conf > (organ_conf.get(organ) or 0):
                organ_conf[organ] = conf
                best_diag[organ] = d
        if not organ_conf:
            return []

        # 按置信度降序，每个受累器官选其体素最多的一层
        items = []
        for organ in sorted(organ_conf, key=lambda o: -organ_conf[o]):
            lab = next((l for l, n in organ_names.items() if n == organ), None)
            if lab is None or lab not in stats:
                continue
            s = stats[lab]
            zs = np.where((mask == lab).sum(axis=(1, 2)) > 0)[0]
            if not len(zs):
                continue
            z = int(zs[np.argmax((mask == lab).sum(axis=(1, 2))[zs])])
            d = best_diag.get(organ, {})
            items.append((organ, lab, z, d, s.get('volume', 0)))
            if len(items) >= max(1, limit // 2):
                break

        evidence = []
        for organ, lab, z, d, volume in items:
            title = (f"{organ} · {d.get('finding', '异常发现')}"
                     f"　置信度 {float(d.get('confidence') or 0):.1%}"
                     f"　第 {z + 1}/{ct.shape[0]} 层")
            for mode, tag in (('seg', '分割叠加'), ('heat', '注意力热力')):
                try:
                    evidence.append({
                        'title': f'{title}（{tag}）',
                        'png': _render_evidence_slice(
                            ct, mask, {lab: stats[lab]}, organ_names, palette,
                            organ_conf, z, mode),
                    })
                except Exception as exc:  # noqa: BLE001 - 单图失败不影响报告
                    app.logger.warning('[report] 证据图渲染失败 %s/%s: %s', organ, tag, exc)
        return evidence[:limit]
    except Exception as exc:  # noqa: BLE001
        app.logger.warning('[report] 生成影像证据失败: %s', exc)
        return []


@app.route('/api/report/generate', methods=['POST'])
def generate_report():
    """生成PDF诊断报告（含患者信息与CT影像证据截图）。"""
    try:
        data = request.get_json()
        diagnosis_result = data.get('diagnosis_result')
        patient_info = data.get('patient_info', {})
        file_id = data.get('file_id')

        if not diagnosis_result:
            return jsonify({'success': False, 'message': '缺少诊断结果'}), 400

        pdf_path = os.path.join(
            app.config['UPLOAD_FOLDER'],
            f"RADAR_Report_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.pdf"
        )
        pdf_path = pdf_generator.generate_report(
            output_path=pdf_path,
            diagnosis_result={**diagnosis_result, **patient_info},
            metadata=diagnosis_result.get('metadata') or patient_info,
            evidence_images=_build_report_evidence(file_id, diagnosis_result),
        )
        
        # 读取PDF文件
        with open(pdf_path, 'rb') as f:
            pdf_data = base64.b64encode(f.read()).decode()
        
        return jsonify({
            'success': True,
            'pdf_data': pdf_data,
            'filename': f"RADAR_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            'message': 'PDF报告生成成功'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'生成报告失败: {str(e)}'}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """
    获取历史记录列表
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        patient_id = request.args.get('patient_id')

        cases = history_manager.list_cases(patient_id=patient_id, limit=limit, offset=offset)

        history_list = []
        for case in cases:
            detail = history_manager.get_case(case['id']) or {}
            diagnoses = detail.get('diagnoses', [])
            findings = [f"{d['organ']}-{d['finding']}" for d in diagnoses]
            history_list.append({
                'id': case['id'],
                'case_id': case['id'],
                'patient_id': case['patient_id'],
                'file_name': case['file_name'],
                'status': case.get('status'),
                'timestamp': case.get('upload_time'),
                'organs_detected': detail.get('organs_detected', []),
                'diagnosis': {
                    # 前端展示的“诊断结论”摘要
                    'diagnosis': '、'.join(findings) if findings else '未发现达到阈值的异常',
                    'diagnoses': diagnoses,
                }
            })

        return jsonify({
            'success': True,
            'records': history_list,
            'total': len(history_list)
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f'获取历史记录失败: {str(e)}'}), 500


@app.route('/api/batch/upload', methods=['POST'])
def batch_upload():
    """
    批量上传文件
    """
    try:
        if 'files' not in request.files:
            return jsonify({'success': False, 'message': '没有文件上传'}), 400
        
        files = request.files.getlist('files')
        results = []
        
        for file in files:
            try:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_id = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{filename}"
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
                    file.save(filepath)
                    
                    # 加载并分析
                    image_data, file_type = load_image_file(filepath)
                    
                    results.append({
                        'success': True,
                        'filename': filename,
                        'file_id': file_id,
                        'file_type': file_type,
                        'shape': list(image_data.shape)
                    })
                else:
                    results.append({
                        'success': False,
                        'filename': file.filename,
                        'message': '不支持的文件格式'
                    })
            except Exception as e:
                results.append({
                    'success': False,
                    'filename': file.filename,
                    'message': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(files),
            'success_count': sum(1 for r in results if r['success'])
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'批量上传失败: {str(e)}'}), 500


def _safe_float(value, fallback):
    """安全转换为float，失败时返回fallback。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _dicom_sort_key(datasets):
    """选择切片排序依据。

    真实DICOM可能缺失SliceLocation，因此按可用性依次回退到
    ImagePositionPatient(第3分量) → SliceLocation → InstanceNumber → 上传顺序。
    """
    probe = datasets[0]
    if getattr(probe, 'ImagePositionPatient', None) is not None:
        return lambda index, ds: (
            _safe_float(getattr(ds, 'ImagePositionPatient', [None, None, 0])[2], 0.0), index
        )
    if getattr(probe, 'SliceLocation', None) is not None:
        return lambda index, ds: (_safe_float(getattr(ds, 'SliceLocation', None), 0.0), index)
    if getattr(probe, 'InstanceNumber', None) is not None:
        return lambda index, ds: (_safe_float(getattr(ds, 'InstanceNumber', None), 0.0), index)
    return lambda index, ds: index


def _dicom_to_hu(dataset):
    """读取单帧/多帧像素并应用RescaleSlope与RescaleIntercept得到HU值。"""
    pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
    slope = _safe_float(getattr(dataset, 'RescaleSlope', None), 1.0)
    intercept = _safe_float(getattr(dataset, 'RescaleIntercept', None), 0.0)
    if slope != 1.0 or intercept != 0.0:
        pixels = pixels * slope + intercept
    if pixels.ndim == 2:
        pixels = pixels[np.newaxis, ...]
    return pixels


def load_dicom_datasets(paths):
    """读取并按解剖顺序排列DICOM数据集，返回(dataset列表, 几何信息)。"""
    paths = [str(path) for path in paths]
    datasets = []
    for path in paths:
        dataset = pydicom.dcmread(path)
        if getattr(dataset, 'SamplesPerPixel', 1) != 1:
            raise ValueError(f'DICOM为彩色图像，RADAR仅支持单通道CT: {Path(path).name}')
        if not hasattr(dataset, 'PixelData'):
            raise ValueError(f'DICOM缺少像素数据: {Path(path).name}')
        datasets.append(dataset)
    if not datasets:
        raise ValueError('未找到有效的DICOM文件')

    sort_key = _dicom_sort_key(datasets)
    datasets = [ds for _, ds in sorted(enumerate(datasets), key=lambda pair: sort_key(*pair))]

    frames = [_dicom_to_hu(ds) for ds in datasets]
    base_shape = frames[0].shape[1:]
    mismatched = [
        Path(datasets[index].filename).name
        for index, frame in enumerate(frames)
        if frame.shape[1:] != base_shape
    ]
    if mismatched:
        raise ValueError(f'DICOM切片尺寸不一致，无法组成序列: {", ".join(mismatched[:5])}')

    return datasets, np.concatenate(frames, axis=0).astype(np.float32), extract_dicom_geometry(datasets)


def extract_dicom_geometry(datasets):
    """从DICOM元数据提取真实体素间距，用于OHIF导出。"""
    first = datasets[0]
    pixel_spacing = getattr(first, 'PixelSpacing', None)
    if pixel_spacing is not None:
        row_spacing = _safe_float(pixel_spacing[0], 1.0)
        column_spacing = _safe_float(pixel_spacing[1], 1.0)
    else:
        row_spacing = column_spacing = 1.0

    # 优先用相邻切片的空间位置差计算层间距，缺失时回退到SliceThickness
    slice_spacing = None
    if len(datasets) > 1:
        positions = []
        for ds in datasets:
            pos = getattr(ds, 'ImagePositionPatient', None)
            if pos is None:
                positions = []
                break
            positions.append(np.asarray([_safe_float(v, 0.0) for v in pos], dtype=np.float32))
        if len(positions) == len(datasets):
            gaps = [float(np.linalg.norm(positions[i + 1] - positions[i])) for i in range(len(positions) - 1)]
            gaps = [gap for gap in gaps if gap > 0]
            if gaps:
                slice_spacing = float(np.median(gaps))
    if slice_spacing is None:
        slice_spacing = _safe_float(getattr(first, 'SliceThickness', None), 1.0)

    return {
        'row_spacing': row_spacing,
        'column_spacing': column_spacing,
        'slice_spacing': slice_spacing,
        'slice_count': len(datasets),
    }


# DICOM 值字段 → (内部键, 展示标签)。顺序即前端展示顺序。
_DICOM_META_FIELDS = [
    ('PatientName', 'patient_name', '患者姓名'),
    ('PatientID', 'patient_id', '患者ID'),
    ('PatientSex', 'patient_sex', '性别'),
    ('PatientAge', 'patient_age', '年龄'),
    ('PatientWeight', 'patient_weight', '体重(kg)'),
    ('Modality', 'modality', '检查类型'),
    ('BodyPartExamined', 'body_part', '检查部位'),
    ('StudyDate', 'study_date', '检查日期'),
    ('StudyTime', 'study_time', '检查时间'),
    ('StudyDescription', 'study_description', '检查描述'),
    ('SeriesDescription', 'series_description', '序列描述'),
    ('Manufacturer', 'manufacturer', '设备厂商'),
    ('InstitutionName', 'institution', '机构'),
    ('SliceThickness', 'slice_thickness', '层厚(mm)'),
    ('KVP', 'kvp', '管电压(kVp)'),
    ('XRayTubeCurrent', 'tube_current', '管电流(mA)'),
    ('ConvolutionKernel', 'conv_kernel', '重建核'),
    ('WindowCenter', 'window_center', '窗位'),
    ('WindowWidth', 'window_width', '窗宽'),
]


def _dicom_value_str(dataset, key):
    """读取单个 DICOM 字段为字符串，缺失或空返回 None。"""
    if not hasattr(dataset, key):
        return None
    value = getattr(dataset, key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _format_dicom_date(raw):
    """DICOM 日期 YYYYMMDD → YYYY-MM-DD，无法解析时原样返回。"""
    if raw and len(raw) == 8 and raw.isdigit():
        return f'{raw[0:4]}-{raw[4:6]}-{raw[6:8]}'
    return raw


def _format_dicom_time(raw):
    """DICOM 时间 HHMMSS[.fff] → HH:MM:SS，无法解析时原样返回。"""
    if raw and len(raw) >= 6 and raw[:6].isdigit():
        return f'{raw[0:2]}:{raw[2:4]}:{raw[4:6]}'
    return raw


def extract_dicom_metadata(datasets):
    """从 DICOM 数据集提取患者/检查/采集元数据（只读头，不含像素）。

    返回扁平字典，仅包含实际存在且非空的字段；序列级信息取首帧。
    """
    if not datasets:
        return {}
    first = datasets[0]
    meta = {}
    for dicom_key, out_key, label in _DICOM_META_FIELDS:
        value = _dicom_value_str(first, dicom_key)
        if value is None:
            continue
        if dicom_key == 'StudyDate':
            value = _format_dicom_date(value)
        elif dicom_key == 'StudyTime':
            value = _format_dicom_time(value)
        elif dicom_key == 'PatientName':
            # DICOM 姓名以 ^ 分隔（姓^名），转为空格分隔更易读
            value = value.replace('^', ' ').strip()
        meta[out_key] = value
    meta['label_map'] = {out_key: label for _, out_key, label in _DICOM_META_FIELDS}
    meta['source'] = 'dicom'
    meta['num_instances'] = len(datasets)
    return meta


def extract_nifti_metadata(filepath, data, geometry):
    """从 NIfTI 头提取可展示的元数据（方向、体素间距、数据类型等）。"""
    import nibabel as nib
    try:
        img = nib.load(filepath)
        hdr = img.header
    except Exception:
        return {}

    meta = {'source': 'nifti'}
    try:
        codes = nib.aff2axcodes(img.affine)
        meta['orientation'] = ''.join(codes)
    except Exception:
        pass
    zooms = geometry or {}
    meta['voxel_spacing'] = (
        f"{zooms.get('row_spacing', 0):.3f} × "
        f"{zooms.get('column_spacing', 0):.3f} × "
        f"{zooms.get('slice_spacing', 0):.3f} mm"
    )
    meta['data_type'] = str(np.asarray(data).dtype)
    try:
        descrip = bytes(hdr['descrip']).decode('utf-8', 'replace').strip().strip('\x00')
        if descrip:
            meta['description'] = descrip[:120]
    except Exception:
        pass
    meta['label_map'] = {
        'orientation': '方向编码',
        'voxel_spacing': '体素间距',
        'data_type': '数据类型',
        'description': '描述',
    }
    return meta


def load_nifti_volume(filepath):
    """加载NIfTI并统一为与DICOM一致的 (Z, Y, X) 轴序，同时返回真实体素间距。

    NIfTI原始轴序由affine决定（常见为 (X, Y, Z)），若直接使用会导致切片数、
    OHIF层厚与切片浏览全部错位。这里按解剖方向重定向到 (S, P, L)。
    """
    import nibabel as nib

    nifti_img = nib.load(filepath)
    data = np.asarray(nifti_img.get_fdata(), dtype=np.float32)
    zooms = [float(z) for z in nifti_img.header.get_zooms()[:3]]

    if data.ndim != 3:
        return data, {}

    target_codes = ('S', 'P', 'L')
    current_codes = nib.aff2axcodes(nifti_img.affine)
    if tuple(current_codes) != target_codes:
        transform = nib.orientations.ornt_transform(
            nib.orientations.axcodes2ornt(current_codes),
            nib.orientations.axcodes2ornt(target_codes),
        )
        data = nib.orientations.apply_orientation(data, transform)
        # 间距随轴重排同步调整，避免层厚与平面间距错位
        zooms = [zooms[int(axis)] for axis in transform[:, 0]]

    geometry = {
        'row_spacing': zooms[1],
        'column_spacing': zooms[2],
        'slice_spacing': zooms[0],
        'slice_count': int(data.shape[0]),
    }
    return data, geometry


def load_image_file_with_geometry(filepath):
    """加载影像并返回 (image_data, file_type, geometry)。

    geometry 仅在DICOM/NIfTI来源下非空，包含真实体素间距。
    """
    if filepath.endswith('.npy'):
        return np.load(filepath), 'numpy', {}
    if filepath.endswith(('.nii', '.nii.gz')):
        image_data, geometry = load_nifti_volume(filepath)
        return image_data, 'nifti', geometry
    if os.path.isdir(filepath):
        datasets, volume, geometry = load_dicom_datasets(
            sorted(path for path in Path(filepath).iterdir() if path.is_file())
        )
        return volume, 'dicom-series', geometry
    if filepath.endswith(('.dcm', '.dicom')):
        # 与序列上传共用同一pydicom读取路径，避免SimpleITK转换导致的轴序转置
        _, volume, geometry = load_dicom_datasets([filepath])
        return volume, 'dicom', geometry
    if filepath.endswith(('.png', '.jpg', '.jpeg')):
        from PIL import Image
        return np.array(Image.open(filepath)), 'image', {}
    raise ValueError(f"不支持的文件格式: {filepath}")


def load_image_file(filepath):
    """
    加载影像文件
    返回: (image_data, file_type)
    """
    image_data, file_type, _ = load_image_file_with_geometry(filepath)
    return image_data, file_type


def apply_window(image_data, window_center, window_width):
    """
    应用窗位窗宽
    """
    data = image_data.astype(np.float32)
    
    # 应用窗位窗宽
    min_value = window_center - window_width / 2
    max_value = window_center + window_width / 2
    
    data = np.clip(data, min_value, max_value)
    
    # 归一化到0-255
    data = (data - min_value) / (max_value - min_value + 1e-8)
    data = (data * 255).astype(np.uint8)
    
    return data


def warmup_inference_engine():
    """后台预热：仅加载器官元数据，不常驻模型权重。

    诊断已改由独立子进程执行（见 run_inference_subprocess），主进程常驻模型
    权重会与子进程推理峰值叠加、超出系统提交余量（实测主进程常驻时 avail
    仅 9.16GB < 推理峰值 10.7GB，被守卫拦截；子进程独占时 avail 最低仍余
    0.52GB，能成功）。故这里只轻量加载 36 个器官名供 overlay 图例即时可用，
    避免首次 overlay 请求等待；模型权重的加载交给诊断子进程。
    """
    try:
        t0 = time.time()
        organs = get_organ_names()
        app.logger.info(
            '[warmup] 器官元数据预热完成，%d 个器官，耗时 %.1fs',
            len(organs), time.time() - t0,
        )
    except Exception as exc:  # noqa: BLE001 - 预热失败不应影响服务启动
        app.logger.warning('[warmup] 器官元数据预热失败: %s', exc)


if __name__ == '__main__':
    # 必须关闭自动重载器：诊断耗时数十秒，而首次推理会加载 monai/pydicom，
    # 期间 watchdog 监测到 site-packages 下的文件变化（如 monai\data\image_reader.py）
    # 就会重启 worker，把正在处理的请求连连接一起掐断（客户端表现为 HTTP 000 /
    # curl exit 56，即「诊断失败」且进程像崩溃了一样）。
    threading.Thread(target=warmup_inference_engine, name='radar-warmup',
                     daemon=True).start()
    # 必须开多线程：Werkzeug 开发服务器默认 threaded=False（单线程串行处理），
    # 而 OHIF viewer 打开一个 study 会并发预取 25~100 个 /frames/<n> 请求。
    # 单线程下这些请求排队，实测 20 张并发要 37.6s，后面的请求被浏览器判定超时/
    # 中断，dicomweb-client 直接抛 "Error: request failed"，viewer 卡在加载态。
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False, threaded=True)
