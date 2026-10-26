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
import socket
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
from library_service import get_library, LIB_PREFIX

# 初始化Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'radar-medical-imaging-2026'
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['SETTINGS_FILE'] = os.path.join(current_dir, 'settings.json')
CORS(app)

# 院内影像库注册表：把 lib_<hash> 映射到磁盘上的原始序列目录。与 settings.json
# 分开存放，因为注册表随导入动态增长，而 settings 是人工维护的配置。
LIBRARY_REGISTRY_FILE = os.path.join(current_dir, 'library_registry.json')

# OHIF 查看器端口。默认 3002 而非 3000/3001，因为本机 Docker 容器已占用：
# 3000 = apisix-prod-grafana-1，3001 = bzh-server-bzh-server-1。
# OHIF 若沿用被占端口，iframe 会打到别的服务上，前端 QIDO 拿不到 study，
# viewer 就抛 "Invalid study URL"。必须与 start_radar_system.ps1 的 $OhifPort
# 及 static/js/app.js 的 OHIF_PORT 保持一致，可用环境变量覆盖。
OHIF_PORT = int(os.environ.get('OHIF_PORT', 3002))

# 每次上传都会把体数据展开成整套 DICOM 序列写入 %TEMP%\ohif\<study_uid>\，
# 一份中等病例就是几十 MB，且原先没有任何回收逻辑（实测累积到 22 个 study、
# 1335 张 DICOM、577 MB），同时会让 /dicomweb/studies 的列表越拖越长。
# 这里按数量上限回收最旧的 study，当前正在看的那个始终保留。
MAX_OHIF_STUDIES = int(os.environ.get('MAX_OHIF_STUDIES', 5))

# 院内影像库快照目录（由 qinghua-report 的 export_abdomen.py 生成）。可用
# LIBRARY_ROOTS 环境变量以 os.pathsep 分隔覆盖，便于换机器部署时改路径。
DEFAULT_LIBRARY_ROOTS = [
    {
        'name': '清华长庚·腹部影像',
        'path': '/home/rimagai/data2/qinghua-report/datasets/abdomen-export',
    },
]

DEFAULT_SETTINGS = {
    'theme': 'dark',
    'language': 'zh-CN',
    'confidence_threshold': 0.7,
    'use_gpu': True,
    'viewer_source': 'local5000',
    # 院内影像库快照根目录。导入时把整个原始 study 目录注册给 DICOMweb，
    # OHIF 因此能切换平扫/动脉期/静脉期/冠矢状位等全部真实序列。
    'library_roots': DEFAULT_LIBRARY_ROOTS
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


def resolve_library_roots():
    """解析生效的影像库根目录列表（环境变量 > settings > 默认值）。"""
    env = os.environ.get('LIBRARY_ROOTS')
    if env:
        return [{'path': p} for p in env.split(os.pathsep) if p.strip()]
    return read_settings().get('library_roots') or DEFAULT_LIBRARY_ROOTS


library = get_library()
library.configure(resolve_library_roots(), LIBRARY_REGISTRY_FILE)


def resolve_file_path(file_id):
    """file_id → 磁盘真实路径。

    全项目原先有 6 处 os.path.join(UPLOAD_FOLDER, file_id) 的硬编码，只能指向
    上传目录。影像库引入 lib_<hash> 形式的 file_id 后，这些位置必须统一走本
    函数：库内 id 经注册表解析到原始序列目录（零拷贝，47 GB 的腹部数据不必
    先复制进 temp），其余 id 仍按上传目录拼接。
    """
    if isinstance(file_id, str) and file_id.startswith(LIB_PREFIX):
        path = library.resolve(file_id)
        if path is not None:
            return path
        # 注册表丢失（例如手工删了 library_registry.json）时给出明确错误，
        # 而不是静默落到 UPLOAD_FOLDER 下拼出一个不存在的路径、再报「文件不存在」。
        raise FileNotFoundError(f'影像库条目 {file_id} 未注册或已失效，请重新从影像库导入')
    return os.path.join(app.config['UPLOAD_FOLDER'], file_id)

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


INFER_SOCK = os.path.join(tempfile.gettempdir(), 'radar_infer.sock')
_INFER_DAEMON_LOCK = threading.Lock()


def _infer_daemon_alive():
    """守护进程是否就绪：socket 存在且能连上（连不上视为未就绪）。"""
    if not os.path.exists(INFER_SOCK):
        return False
    try:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(2)
        probe.connect(INFER_SOCK)
        probe.close()
        return True
    except OSError:
        return False


def ensure_infer_daemon():
    """确保常驻推理守护进程在跑；不在则后台拉起（幂等，带锁防并发重复拉起）。

    模型权重在守护进程里只加载一次，之后所有诊断复用，不再「用一次加载一次」。
    拉起是异步的：加载权重的数十秒内本次诊断自动走一次性子进程兜底，用户无感。
    """
    with _INFER_DAEMON_LOCK:
        if _infer_daemon_alive():
            return True
        use_gpu = bool(read_settings().get('use_gpu', True))
        cmd = [sys.executable, os.path.join(current_dir, 'infer_daemon.py'),
               '--socket', INFER_SOCK]
        if use_gpu:
            cmd.append('--use-gpu')
        log_path = os.path.join(tempfile.gettempdir(), 'radar_infer_daemon.log')
        try:
            logf = open(log_path, 'a', encoding='utf-8')
            subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                             cwd=current_dir, start_new_session=True)
            app.logger.info('[infer-daemon] 拉起常驻推理进程 (use_gpu=%s)', use_gpu)
            return True
        except OSError as exc:
            app.logger.warning('[infer-daemon] 拉起失败，本次走一次性子进程: %s', exc)
            return False


def run_inference_daemon(nifti_path, overlay_path=None, timeout=1800):
    """向常驻守护进程提交一次推理；返回与 run_inference_subprocess 相同的三元组。

    任何连接/协议异常都返回错误串，由调用方降级到一次性子进程，保证可用性。
    """
    try:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.settimeout(timeout)
        conn.connect(INFER_SOCK)
    except OSError as exc:
        return None, [], f'daemon 连接失败: {exc}'
    try:
        with conn:
            conn.sendall((json.dumps(
                {'nifti': nifti_path, 'overlay': overlay_path},
                ensure_ascii=False) + '\n').encode('utf-8'))
            buf = b''
            while b'\n' not in buf:
                chunk = conn.recv(1 << 20)
                if not chunk:
                    return None, [], 'daemon 连接提前关闭'
                buf += chunk
        payload = json.loads(buf.split(b'\n', 1)[0].decode('utf-8'))
    except (OSError, socket.timeout, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, [], f'daemon 通信失败: {exc}'
    organs = payload.get('organs') or []
    if organs:
        global _ORGAN_NAMES
        _ORGAN_NAMES = list(organs)
    if not payload.get('success'):
        return None, organs, payload.get('error', '推理失败')
    return payload.get('result'), organs, None


def run_inference(nifti_path, overlay_path=None, timeout=1800):
    """推理统一入口：优先常驻守护进程，失败自动降级一次性子进程并重新拉起守护。

    守护进程崩溃（OOM 等）时本次诊断不失败：先用短命子进程把结果给用户，
    同时后台重新拉起守护，下一次诊断恢复常驻模式。
    """
    if _infer_daemon_alive():
        result, organs, err = run_inference_daemon(nifti_path, overlay_path, timeout)
        if err is None:
            return result, organs, None
        app.logger.warning('[infer] 常驻进程异常（%s），降级一次性子进程并重新拉起守护',
                           err[:80])
        threading.Thread(target=ensure_infer_daemon, daemon=True).start()
        return run_inference_subprocess(nifti_path, overlay_path, timeout)
    # 守护不在：本次走子进程兜底，同时后台拉起守护供后续诊断复用
    threading.Thread(target=ensure_infer_daemon, daemon=True).start()
    return run_inference_subprocess(nifti_path, overlay_path, timeout)


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
        err = payload.get('error', '推理失败')
        # GPU 内存守卫拦截 → 自动降级 CPU 推理重试一次。本机提交余量常年在
        # 10~13 GB 晃动、贴着 GPU 守卫阈值（~10.9 GB），多次出现差 0.1 GB 被拦；
        # CPU 路径阈值仅 ~6.4 GB（按物理可用内存判），慢但能出结果，避免用户
        # 反复重试撞墙。CPU 守卫也拦时返回错误不再重试。
        if use_gpu and '影像过大' in err and 'GPU' in err:
            app.logger.warning('[infer] GPU 内存守卫拦截（%s），自动降级 CPU 推理重试',
                               err[:60])
            os.remove(result_path)
            cmd_cpu = [c for c in cmd if c != '--use-gpu']
            proc = subprocess.run(
                cmd_cpu, capture_output=True, text=True, encoding='utf-8',
                errors='replace', timeout=timeout, cwd=current_dir)
            if proc.stderr:
                for line in proc.stderr.strip().splitlines():
                    app.logger.info('[infer-worker-cpu] %s', line)
            if os.path.exists(result_path):
                with open(result_path, encoding='utf-8') as f:
                    payload = json.load(f)
                organs = payload.get('organs') or organs
                if organs:
                    _ORGAN_NAMES = list(organs)
                if payload.get('success'):
                    return payload.get('result'), organs, None
                err = payload.get('error', err)
        return None, organs, err
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
        # 非压缩包文件的扩展名校验仅在多文件场景做「明确不支持扩展名」的
        # 预拒（避免把上千个噪声文件塞进 FormData）。单文件不看扩展名：
        # PACS/光盘导出的单张切片常用 .ima/.dic/无扩展名/纯 UID 命名，
        # 扩展名白名单会把真切片误拒（多文件路径本就靠落盘后文件头嗅探）。
        if len(files) > 1:
            for item in files:
                if is_archive_name(item.filename):
                    continue
                if not allowed_file(item.filename):
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
        filepath = resolve_file_path(file_id)
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
        filepath = resolve_file_path(file_id)
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
    TotalSegmentator 原始分辨率精细分割，3D 表面最光滑。

    提取管线（实验见 .dbg/exp_mesh_quality2.py）：overlay 的 spacing 各向异性
    （如 [5,1,1]mm），在体素空间做高斯平滑会让 z 向等效 σ 达 7.5mm、表面呈阶梯
    块面。故先重采样到 2mm 各向同性物理网格，用物理 σ=2mm 平滑后提取等值面，
    再在物理 mm 空间做 Taubin 平滑（λ 收缩/μ 膨胀交替，去噪且不缩体积），
    顶点换算回原始体素坐标返回（前端仍按 spacing 映射渲染坐标）。实测粗糙度
    8.7°→5.5°、网格-体素体积误差 12.6%→7.6%。另从上传影像的 CT 灰度提取平滑
    的体表轮廓（id=0）作为半透明解剖背景。响应 gzip 压缩并缓存。
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

            meshes = []
            iso = 2.0           # 各向同性重采样步长（mm）
            sigma_mm = 2.0      # 物理空间平滑 σ（mm）
            taubin_iters = 10   # Taubin 平滑迭代数（λ/μ 各算一步）
            sp = np.asarray(spacing, dtype=np.float64)

            def _taubin_mm(verts_mm, faces):
                """物理 mm 空间 Taubin 平滑：λ 收缩 + μ 膨胀交替，去噪且不缩体积。"""
                e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
                e = np.vstack([e, e[:, ::-1]])
                n = len(verts_mm)
                deg = np.bincount(e[:, 0], minlength=n).astype(np.float32)
                deg[deg == 0] = 1.0
                inv = 1.0 / deg[:, None]
                v = verts_mm.copy()
                for _ in range(taubin_iters):
                    for k in (0.5, -0.53):
                        acc = np.zeros_like(v)
                        np.add.at(acc, e[:, 0], v[e[:, 1]])
                        v = v + k * (acc * inv - v)
                return v

            def _extract(vol_full):
                """各向同性重采样 → 物理 σ 平滑 → marching cubes → Taubin。

                返回 (verts, faces)，verts 为原始体素坐标（前端按 spacing 映射）。
                """
                small = ndimage.zoom(vol_full.astype(np.float32), sp / iso, order=1)
                smooth = ndimage.gaussian_filter(small, sigma_mm / iso,
                                                 mode='constant')
                verts, faces, _n, _v = measure.marching_cubes(smooth, level=0.5)
                verts_mm = _taubin_mm(verts * iso, faces)
                return verts_mm / sp, faces

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
    # 影像库注册的原始 study：目录是 <study>/<series>/*.dcm 两层结构，由
    # dicom_files 递归展开。这样 OHIF 能切换平扫/动脉期/静脉期/冠矢状位等
    # 全部真实序列，并显示真实采集参数，而不是上传路径重写出的单个合成序列。
    lib_dir = library.study_dir(study_uid)
    if lib_dir:
        return Path(lib_dir)
    folder_uid, _ = find_folder_study()
    if folder_uid == study_uid:
        return Path(current_dir).parent / 'data' / 'test_ct_dicom_series'
    return None


def dicom_files(study_dir):
    """列出 study 目录下全部 DICOM 文件（有序）。

    上传导出的 study 是平铺的 <n>.dcm，院内原始数据是 <study>/<series>/*.dcm
    两层，故用 rglob 兼容两种布局。文件名排序不影响解剖顺序——原始切片以
    SOPInstanceUID 命名，真正的层序由 pydicom 的 SliceLocation 排序键决定。
    """
    if not study_dir or not study_dir.is_dir():
        return []
    return sorted(path for path in study_dir.rglob('*.dcm') if path.is_file())


def dicom_json(dataset):
    return dataset.to_json_dict()


# 院内原始 study 的指纹缓存。这些目录是 root 属主的只读快照，内容不会变；
# 而一个腹部 CT study 可能有上千个 .dcm 分布在十几个序列子目录里，若在
# DICOMweb 热路径上每次请求都递归 walk 一遍，光是 stat 就要几十毫秒，OHIF
# 并发预取上百帧时会明显拖慢出片。故只算一次并长期缓存。
_LIB_STUDY_STAMP_CACHE = {}
_LIB_STUDY_STAMP_LOCK = threading.Lock()


def _walk_dicom_stamp(study_dir):
    """递归统计 .dcm 文件数与最新 mtime，用 os.scandir 复用目录项的 stat。"""
    count = 0
    newest = 0.0
    stack = [str(study_dir)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=True):
                            stack.append(entry.path)
                            continue
                        if not entry.name.endswith('.dcm'):
                            continue
                        count += 1
                        mtime = entry.stat(follow_symlinks=True).st_mtime
                        if mtime > newest:
                            newest = mtime
                    except OSError:
                        continue
        except OSError:
            return None
    if count == 0:
        return None
    return (count, newest)


def _study_dir_stamp(study_dir):
    """目录指纹：文件数 + 最新 mtime。

    重新上传（新目录）或 prune_ohif_studies 回收后指纹会变，缓存自然失效，
    不需要显式清理，也不会读到被删目录的旧数据。

    上传导出的 study 是平铺目录，用 scandir 单层扫描即可；院内原始 study
    是只读快照，走上面的长期缓存，避免在 /frames/1 热路径上反复递归。
    """
    key = str(study_dir)
    if library.study_dir(Path(study_dir).name) == os.path.realpath(key):
        with _LIB_STUDY_STAMP_LOCK:
            cached = _LIB_STUDY_STAMP_CACHE.get(key)
        if cached is not None:
            return cached
        stamp = _walk_dicom_stamp(study_dir)
        if stamp is not None:
            with _LIB_STUDY_STAMP_LOCK:
                _LIB_STUDY_STAMP_CACHE[key] = stamp
        return stamp

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
    if not study_uid:
        candidate_uids.extend(library.registered_study_uids())
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
        # pydicom 2.4 从空 dict 构造 FileDataset 时 is_implicit_VR 会落到隐式 VR，
        # 而 file_meta 声明的 TransferSyntaxUID 是 ExplicitVRLittleEndian，文件头
        # 声明与数据集编码不一致：服务端 pydicom 读取能自动回退（只告警），但
        # 浏览器 dicom-parser 严格按声明的传输语法解析，直接抛
        # "parseDicomDataSetExplicit: buffer overrun"，OHIF 视口永远灰屏。
        # 这里显式钉死编码，保证声明与实体一致。
        ds.is_implicit_VR = False
        ds.is_little_endian = True
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


# ==================== 院内影像库 ====================
#
# 服务器本地已有数十 GB 的脱敏 CT，而 /api/upload 只接受浏览器 multipart 上传，
# 等于要求用户先把数据下载到自己机器再传回来。下面这组接口让前端能直接浏览
# 快照目录、按序列粒度选用已有数据，零拷贝接入现有的 OHIF + 诊断链路。

def _library_error(message, status=400):
    return jsonify({'success': False, 'message': message}), status


@app.route('/api/library/status', methods=['GET'])
def library_status():
    """影像库根目录装载状态（便于排查「列表为空」是配置问题还是数据问题）。"""
    library.refresh()
    return jsonify({'success': True, 'roots': library.status(),
                    'registry_entries': len(library._registry)})


@app.route('/api/library/refresh', methods=['POST'])
def library_refresh():
    """重新读取快照目录（导出脚本重跑后调用），并按 settings 重载根目录配置。"""
    library.configure(resolve_library_roots(), LIBRARY_REGISTRY_FILE)
    library.refresh()
    return jsonify({'success': True, 'roots': library.status()})


@app.route('/api/library/tree', methods=['GET'])
def library_tree():
    """三级分类树（系统 → 病种 → 具体病名）+ 各级病例数。"""
    library.refresh()
    return jsonify({'success': True, 'data': library.tree()})


@app.route('/api/library/cases', methods=['GET'])
def library_cases():
    """病例列表：支持分类/关键词/模态/有无影像/有无报告/发布状态筛选 + 分页。"""
    library.refresh()
    args = request.args

    def bool_or_none(name):
        raw = (args.get(name) or '').strip().lower()
        if raw in ('1', 'true', 'yes'):
            return True
        if raw in ('0', 'false', 'no'):
            return False
        return None

    try:
        result = library.query_cases(
            root=args.get('root') or None,
            category=args.get('category') or None,
            keyword=args.get('keyword') or None,
            modality=args.get('modality') or None,
            has_dicom=bool_or_none('has_dicom'),
            has_report=bool_or_none('has_report'),
            status=args.get('status') or None,
            page=args.get('page', 1),
            page_size=args.get('page_size', 20),
        )
    except ValueError as exc:
        return _library_error(f'查询参数不合法: {exc}')
    return jsonify({'success': True, **result})


@app.route('/api/library/case/<case_id>', methods=['GET'])
def library_case_detail(case_id):
    """病例详情：检查列表（含序列概要）+ 报告全文。"""
    library.refresh()
    detail = library.case_detail(case_id)
    if detail is None:
        return _library_error('病例不存在', 404)
    return jsonify({'success': True, 'data': detail})


@app.route('/api/library/examination/<examination_id>/series', methods=['GET'])
def library_exam_series(examination_id):
    """某检查的全部序列 + RADAR 适用性评估 + 推荐项。

    评估必须显式给出「不可用原因」：把定位像、剂量报告或冠状位重建喂进
    RADAR 不会报错，只会静默产出解剖错乱的诊断结果，比直接失败更危险。
    """
    library.refresh()
    force = request.args.get('refresh', '').lower() in ('1', 'true')
    data = library.series_of_exam(examination_id, force=force)
    if data is None:
        return _library_error('检查不存在', 404)
    return jsonify({'success': True, 'data': data})


@app.route('/api/library/import', methods=['POST'])
def library_import():
    """把库内某个序列登记为 file_id，返回与 /api/upload 同构的 info。

    零拷贝：不复制 DICOM，只把 lib_<hash> → 原始序列目录写进注册表；同时把
    整个原始 study 目录登记给 DICOMweb，使 OHIF 能加载全部真实序列。
    """
    library.refresh()
    payload = request.get_json(silent=True) or {}
    series_id = (payload.get('series_id') or '').strip()
    if not series_id:
        return _library_error('缺少 series_id')

    hit, root = library.find_series(series_id)
    if hit is None:
        return _library_error(f'序列不存在: {series_id}', 404)
    series, exam = hit['series'], hit['exam']

    series_dir = root._series_host_dir(exam, series)
    if not series_dir or not os.path.isdir(series_dir):
        return _library_error(
            f'序列目录在服务器上不可读: {series_dir or "(未解析)"}', 404)

    study_dir = exam.get('host_storage_path')
    if not study_dir or not os.path.isdir(study_dir):
        # study 目录不可用时退化为「只挂这个序列」，OHIF 仍可查看单序列
        study_dir = series_dir
    study_uid = str(exam.get('study_instance_uid') or '')

    if not study_uid:
        return _library_error('该检查缺少 StudyInstanceUID，无法交给 OHIF 阅片')

    file_id = library.register_series(series_dir, study_dir, {
        'study_instance_uid': study_uid,
        'series_instance_uid': series.get('series_instance_uid'),
        'examination_id': exam.get('examination_id'),
        'case_id': hit['case_id'],
        'accession_number': exam.get('accession_number'),
        'modality': exam.get('modality'),
        'series_description': series.get('series_description'),
        'series_number': series.get('series_number'),
        'source': 'library',
        'library_root': root.name,
    })

    # 只读文件头拼装 info：库内导入不需要像上传那样把像素全部解码，
    # 体数据留给 /api/image/slice 与 /api/diagnose 按需加载。
    paths = dicom_files(Path(series_dir))
    if not paths:
        return _library_error('序列目录内没有 .dcm 文件', 404)
    try:
        first = pydicom.dcmread(str(paths[0]), stop_before_pixels=True)
    except Exception as exc:  # noqa: BLE001
        return _library_error(f'无法读取 DICOM 头: {exc}', 500)

    metadata = extract_dicom_metadata([first])
    metadata['num_instances'] = len(paths)
    rows = int(getattr(first, 'Rows', 0) or 0)
    cols = int(getattr(first, 'Columns', 0) or 0)
    pixel_spacing = getattr(first, 'PixelSpacing', None) or [1.0, 1.0]
    thickness = _safe_float(getattr(first, 'SliceThickness', None), 1.0)

    display_name = ' / '.join(filter(None, [
        str(metadata.get('patient_id') or exam.get('accession_number') or ''),
        (series.get('series_description') or '').strip(),
        exam.get('study_description') or '',
    ])) or os.path.basename(series_dir)

    session['current_file_id'] = file_id
    session['current_filepath'] = series_dir
    return jsonify({
        'success': True,
        'file_id': file_id,
        'message': f'已从影像库载入 {len(paths)} 张切片',
        'info': {
            'filename': display_name,
            'file_type': 'dicom-series',
            'upload_mode': 'library',
            'source': 'library',
            'library_root': root.name,
            'uploaded_files': len(paths),
            'slice_count': len(paths),
            'shape': [len(paths), rows, cols],
            'dtype': 'int16',
            'size_mb': round(sum(_file_size(p) for p in paths) / 1024 / 1024, 2),
            'spacing': {
                'row': _safe_float(pixel_spacing[0], 1.0),
                'column': _safe_float(pixel_spacing[1], 1.0),
                'slice': thickness,
            },
            'metadata': metadata,
            'ohif_study_uid': study_uid,
            'series_dir': series_dir,
            'study_dir': study_dir,
            'case': {
                'case_id': hit['case_id'],
                'examination_id': exam.get('examination_id'),
                'accession_number': exam.get('accession_number'),
                'modality': exam.get('modality'),
                'series_instance_uid': series.get('series_instance_uid'),
                'series_description': series.get('series_description'),
                'series_number': series.get('series_number'),
            },
        },
    })


def _file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


@app.route('/api/library/reports/<case_id>', methods=['GET'])
def library_reports(case_id):
    """病例关联报告全文（阅片时右侧对照原始报告用）。"""
    library.refresh()
    detail = library.case_detail(case_id)
    if detail is None:
        return _library_error('病例不存在', 404)
    return jsonify({'success': True, 'data': detail['reports']})


@app.route('/api/library/scan', methods=['POST'])
def library_scan():
    """扫描服务器上的任意 DICOM 目录（外部数据的服务器侧导入入口）。

    与浏览器的 /api/upload 互补：数据已经在服务器上时，不必先下载到本地
    再上传回去。只列目录、不复制文件，返回可选的序列清单。
    """
    payload = request.get_json(silent=True) or {}
    target = (payload.get('path') or '').strip()
    if not target:
        return _library_error('缺少 path')
    if not os.path.isdir(target):
        return _library_error(f'目录不存在: {target}', 404)
    try:
        found = library.scan_directory(target)
    except Exception as exc:  # noqa: BLE001
        return _library_error(f'扫描失败: {exc}', 500)
    return jsonify({'success': True, 'data': found})


@app.route('/api/library/import_path', methods=['POST'])
def library_import_path():
    """登记 /api/library/scan 列出的某个序列目录，走与库内导入相同的链路。"""
    payload = request.get_json(silent=True) or {}
    series_dir = (payload.get('series_dir') or '').strip()
    study_dir = (payload.get('study_dir') or '').strip() or series_dir
    if not series_dir or not os.path.isdir(series_dir):
        return _library_error(f'序列目录不存在: {series_dir}', 404)
    paths = dicom_files(Path(series_dir))
    if not paths:
        return _library_error('该目录下没有 .dcm 文件', 404)
    try:
        first = pydicom.dcmread(str(paths[0]), stop_before_pixels=True)
    except Exception as exc:  # noqa: BLE001
        return _library_error(f'无法读取 DICOM 头: {exc}', 500)

    study_uid = str(getattr(first, 'StudyInstanceUID', '') or '')
    if not study_uid:
        return _library_error('DICOM 缺少 StudyInstanceUID，无法交给 OHIF 阅片')
    if not os.path.isdir(study_dir):
        study_dir = series_dir

    file_id = library.register_series(series_dir, study_dir, {
        'study_instance_uid': study_uid,
        'series_instance_uid': str(getattr(first, 'SeriesInstanceUID', '') or ''),
        'source': 'external_path',
        'series_description': str(getattr(first, 'SeriesDescription', '') or ''),
        'series_number': getattr(first, 'SeriesNumber', None),
    })
    metadata = extract_dicom_metadata([first])
    metadata['num_instances'] = len(paths)
    rows = int(getattr(first, 'Rows', 0) or 0)
    cols = int(getattr(first, 'Columns', 0) or 0)
    pixel_spacing = getattr(first, 'PixelSpacing', None) or [1.0, 1.0]
    thickness = _safe_float(getattr(first, 'SliceThickness', None), 1.0)

    session['current_file_id'] = file_id
    session['current_filepath'] = series_dir
    return jsonify({
        'success': True,
        'file_id': file_id,
        'message': f'已载入服务器目录 {len(paths)} 张切片',
        'info': {
            'filename': os.path.basename(series_dir.rstrip('/')) or series_dir,
            'file_type': 'dicom-series',
            'upload_mode': 'library',
            'source': 'external_path',
            'uploaded_files': len(paths),
            'slice_count': len(paths),
            'shape': [len(paths), rows, cols],
            'dtype': 'int16',
            'size_mb': round(sum(_file_size(p) for p in paths) / 1024 / 1024, 2),
            'spacing': {
                'row': _safe_float(pixel_spacing[0], 1.0),
                'column': _safe_float(pixel_spacing[1], 1.0),
                'slice': thickness,
            },
            'metadata': metadata,
            'ohif_study_uid': study_uid,
            'series_dir': series_dir,
            'study_dir': study_dir,
        },
    })


def _build_library_context(file_id, result):
    """为院内影像库导入的病例组装溯源快照，随历史记录一起落库。

    诊断结果本身不含「这是谁、哪次检查、哪个序列、原始报告怎么写」这些信息，
    而院外报告复核恰恰需要它们。这里一次性抓齐并固化，历史记录即便日后影像库
    被重新导出/移动，仍能回溯当时的病例与人工报告做对照。

    返回 None 表示非库内病例（外部上传），不写 library_json。
    """
    entry = library.registry_entry(file_id)
    if not entry:
        return None

    ctx = {
        'library_root': entry.get('library_root'),
        'case_id': entry.get('case_id'),
        'examination_id': entry.get('examination_id'),
        'series_instance_uid': entry.get('series_instance_uid'),
        'accession_number': entry.get('accession_number'),
        'series_description': entry.get('series_description'),
        'series_number': entry.get('series_number'),
        'study_instance_uid': entry.get('study_instance_uid'),
        'series_dir': entry.get('series_dir'),
        'ohif_study_uid': result.get('ohif_study_uid') or entry.get('study_instance_uid'),
    }

    # 病例级信息与原始报告全文：报告是 AI 对照的基准，必须固化
    detail = None
    try:
        detail = library.case_detail(ctx['case_id'])
    except Exception:  # noqa: BLE001 - 溯源失败不该阻断诊断落库
        detail = None
    if detail:
        ctx['patient'] = {
            'patient_name': detail.get('patient_name'),
            'medical_record_no': detail.get('medical_record_no'),
            'gender': detail.get('gender'),
            'age': detail.get('age'),
            'level1': detail.get('level1'),
            'level2': detail.get('level2'),
            'level3': detail.get('level3'),
            'disease_name': detail.get('disease_name'),
        }
        ctx['reports'] = detail.get('reports') or []
        for exam in detail.get('examinations') or []:
            if exam.get('examination_id') != ctx['examination_id']:
                continue
            ctx['examination'] = {
                'modality': exam.get('modality'),
                'study_description': exam.get('study_description'),
                'body_part_examined': exam.get('body_part_examined'),
                'examination_date': exam.get('examination_date'),
                'series_count': exam.get('series_count'),
                'image_count': exam.get('image_count'),
                'size_gb': exam.get('size_gb'),
                'host_storage_path': exam.get('host_storage_path'),
            }
            # 只固化本次实际导入的那个序列，避免把整检查的序列清单都写进历史
            for s in exam.get('series') or []:
                if s.get('series_instance_uid') == ctx['series_instance_uid']:
                    ctx['series'] = s
                    break
    return ctx


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


# 推理串行化：Flask threaded=True 下并发诊断会同时拉起多个推理子进程，
# CPU/内存互相抢占导致单次推理耗时翻倍，前端/代理等待超时（504）。
# 诊断本身是重计算任务，串行执行更符合预期。
_DIAGNOSE_LOCK = threading.Lock()


def _assess_input_quality(image_data, geometry, file_type, filepath, overlay_path=None):
    """评估输入影像是否满足 RADAR 的适用条件，返回警告列表。

    RADAR 在「增强、胸腹大范围 CT」上训练。两类输入会系统性产生高置信度
    假阳性（实测：局灶裁剪序列报肺转移瘤/胃溃疡，平扫序列报胰腺癌）：
      1) 扫描范围过短：Z 向覆盖远小于模型视野，重采样后大部分体为填充，
         模型把填充边界与局部解剖误读为病变；
      2) 平扫（无对比剂）：实质/血管强化模式与训练分布不符。
    警告不阻断诊断，仅随结果返回并由前端提示。推理后调用：平扫判定依赖
    overlay 的肝掩码平均 HU（增强门脉期肝实质约 90-110 HU，平扫约 45-60）。
    """
    warnings = []
    geometry = geometry or {}
    slice_spacing = geometry.get('slice_spacing') or 1.0
    slice_count = geometry.get('slice_count') or int(np.asarray(image_data).shape[0])
    z_extent = float(slice_spacing) * max(int(slice_count) - 1, 0)
    if 0 < z_extent < 250.0:
        warnings.append(
            f'扫描范围过短：Z 向覆盖约 {z_extent:.0f} mm（{slice_count} 层）。'
            'RADAR 训练于胸腹大范围 CT，局灶/裁剪序列超出其适用分布，'
            '诊断结果假阳性风险极高，仅供演示参考。'
        )

    if file_type in ('dicom-series', 'dicom'):
        agent = ''
        try:
            if os.path.isdir(filepath):
                files = sorted(str(p) for p in Path(filepath).iterdir() if p.is_file())
            else:
                files = [filepath]
            if files:
                ds = pydicom.dcmread(files[len(files) // 2], stop_before_pixels=True)
                agent = str(getattr(ds, 'ContrastBolusAgent', '') or '').strip()
                # 部分设备把缺失值写成字面量 "None"/"N/A"
                if agent.lower() in ('none', 'n/a', 'na'):
                    agent = ''
        except Exception:
            agent = ''
        if not agent and overlay_path and os.path.exists(overlay_path):
            # 头字段缺失/为空时用肝区平均 HU 判定：overlay 的 ct 为
            # HU[-300,400] 归一化 uint8，肝掩码标签 id=21（器官表第 21 位）。
            try:
                ov = np.load(overlay_path)
                ct = ov['ct'].astype(np.float32) / 255.0 * 700.0 - 300.0
                liver = ov['mask'] == 21
                if int(liver.sum()) > 500 and float(ct[liver].mean()) < 70.0:
                    warnings.append(
                        f'疑似平扫（无对比剂）序列：肝区平均 '
                        f'{float(ct[liver].mean()):.0f} HU。RADAR 训练于增强 CT，'
                        '平扫输入的强化特征缺失，诊断置信度不可靠。'
                    )
            except Exception:
                pass
    return warnings


@app.route('/api/diagnose/<file_id>', methods=['POST'])
def diagnose(file_id):
    """
    执行AI诊断
    """
    try:
        filepath = resolve_file_path(file_id)
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
        # 推理走统一入口：优先常驻守护进程（权重只加载一次），守护不在或
        # 崩溃时自动降级一次性子进程并后台重新拉起守护。串行化：并发推理
        # 互相抢占 CPU/内存，单次耗时翻倍触发 504。
        with _DIAGNOSE_LOCK:
            result, _organs, err = run_inference(nifti_path, overlay_path)
        if err:
            status = 503 if '过大' in err or '内存' in err or '显存' in err else 500
            return jsonify({'success': False, 'message': f'诊断失败: {err}'}), status

        # 输入适用性评估：扫描范围/对比剂不满足训练分布时随结果给出警告。
        # 平扫判定依赖推理产出的 overlay 肝掩码，故在推理后执行。
        input_warnings = _assess_input_quality(
            image_data, geometry, file_type, filepath, overlay_path
        )

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
        result['warnings'] = input_warnings

        # 患者ID：优先用 DICOM 真实 PatientID，缺失时回退到时间戳生成的占位ID
        patient_id = metadata.get('patient_id') or f"P{datetime.now().strftime('%Y%m%d%H%M%S')}"
        # 引擎内部使用的是临时NIfTI路径，这里统一替换为面向用户的患者ID
        result['patient_id'] = patient_id
        # 库内导入的 filepath 是 <series_uid> 目录，直接取 basename 会在历史记录里
        # 显示成一串 UID（如 1.2.840.113619...），不可读。用注册表里的检查号/序列名
        # 拼一个有意义的名字，与上传病例的 file_name 展示保持一致的可读性。
        display_name = os.path.basename(filepath)
        lib_entry = library.registry_entry(file_id)
        if lib_entry:
            parts = [
                lib_entry.get('accession_number'),
                (lib_entry.get('series_description') or '').strip(),
            ]
            display_name = ' · '.join(p for p in parts if p) or display_name
        # 库内导入附带溯源快照（病例/检查/序列 + 原始报告全文），外部上传为 None
        lib_context = _build_library_context(file_id, result)
        case_id = history_manager.add_case(
            patient_id=patient_id,
            file_name=display_name,
            file_path=filepath,
            diagnosis_result=result,
            metadata={'image_shape': list(image_data.shape), 'file_type': file_type},
            file_id=file_id,
            source='library' if lib_entry else 'upload',
            library_context=lib_context,
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


@app.route('/api/client_diag', methods=['POST'])
def client_diag():
    """接收前端浏览器环境诊断并写入服务日志。

    阅片视口卡 Loading 这类问题只发生在用户浏览器里（WebGL2 不可用、
    扩展拦截等），服务端日志看不到任何异常。前端打开查看器时把环境
    快照 POST 上来，出问题时直接查 flask.log 即可定位，不必让用户
    手动翻控制台。只记日志、不落库，内容全部来自前端自报。
    """
    payload = request.get_json(silent=True) or {}
    safe = {k: payload.get(k) for k in
            ('webgl2', 'webgl1', 'user_agent', 'study_uid', 'canvas_count',
             'viewport_loading', 'cross_origin_isolated', 'shared_array_buffer')}
    app.logger.info('[client-diag] %s', json.dumps(safe, ensure_ascii=False))
    return jsonify({'success': True})


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
        if 'library_roots' in payload:
            roots = payload['library_roots']
            if not isinstance(roots, list):
                raise ValueError('library_roots 必须是数组')
            cleaned = []
            for item in roots:
                entry = item if isinstance(item, dict) else {'path': item}
                path = str(entry.get('path') or '').strip()
                if not path:
                    continue
                if not os.path.isdir(path):
                    raise ValueError(f'影像库目录不存在: {path}')
                cleaned.append({'name': entry.get('name') or os.path.basename(path),
                                'path': path})
            if not cleaned:
                raise ValueError('至少需要一个有效的影像库目录')
            settings['library_roots'] = cleaned
        write_settings(settings)
        # 库目录变了必须重新装载，否则本次进程仍用旧索引（要重启才生效）
        library.configure(resolve_library_roots(), LIBRARY_REGISTRY_FILE)
        library.refresh()
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


def _load_attention(file_id):
    """加载 overlay.npz 内的器官空间注意力 (36,D,H,W) uint8，无则 None。

    注意力为模型 MultiheadAttention 的器官级空间权重散回全体积（每器官
    归一化 0-255），报告红圈据此把病灶约束到器官内模型高关注区。
    """
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'radar_infer', file_id,
                        'overlay.npz')
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as npz:
            if 'attn' not in npz.files:
                return None
            return npz['attn']
    except Exception:  # noqa: BLE001
        return None


def _render_evidence_slice(ct, mask, stats, organ_names, palette,
                           organ_conf, slice_idx, mode, attn=None):
    """把单张切片渲染成 PNG 字节。

    mode='circle': 纯 CT 灰度 + 红圈标出受累器官病灶，不做任何填充，保留
                   原始影像纹理供医师复核。定位优先级：模型空间注意力高
                   关注区（attn，器官内前 50% 分位）∩ 原始分割 → 最大连通
                   域外接圆；无 attn 时退化为原始分割最大连通域外接圆
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

    if mode == 'circle':
        from PIL import ImageDraw
        img = PILImage.fromarray(rgb.astype(np.uint8), mode='RGB')
        draw = ImageDraw.Draw(img)
        for lab in labels:
            vol = seg == lab
            if not vol.any():
                continue
            region = vol
            # 注意力高关注区（器官内前 50% 分位）∩ 原始分割：把圈约束到
            # 模型做该诊断时真正关注的器官子区，避免大器官整器官画圈
            if attn is not None and lab - 1 < attn.shape[0]:
                a = attn[lab - 1][slice_idx]
                pos = a[vol]
                if pos.size and pos.max() > 0:
                    thr = np.quantile(pos[pos > 0], 0.5)
                    hi = (a >= thr) & (a > 0) & vol
                    if hi.sum() >= 20:
                        region = hi
            comps, n = ndimage.label(region)
            if not n:
                continue
            sizes = ndimage.sum(region, comps, index=list(range(1, n + 1)))
            big = int(np.argmax(sizes)) + 1
            ys, xs = np.where(comps == big)
            cy, cx = float(ys.mean()), float(xs.mean())
            # 外接圆半径取包围盒半长边并留 25% 余量，保证圈住整个病灶区
            r = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2.0 * 1.25 + 4
            draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                         outline=(255, 40, 40), width=3)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
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


def _compose_evidence_pair(png_seg, png_circle):
    """左右拼接对照图：分割标记图 | 原图红圈，中间留白分隔。"""
    from PIL import Image as PILImage
    left = PILImage.open(io.BytesIO(png_seg)).convert('RGB')
    right = PILImage.open(io.BytesIO(png_circle)).convert('RGB')
    h = max(left.height, right.height)
    gap = 8
    canvas = PILImage.new('RGB', (left.width + gap + right.width, h),
                          (255, 255, 255))
    canvas.paste(left, (0, (h - left.height) // 2))
    canvas.paste(right, (left.width + gap, (h - right.height) // 2))
    buf = io.BytesIO()
    canvas.save(buf, format='PNG')
    return buf.getvalue()


def _build_report_evidence(file_id, diagnosis_result, limit=4):
    """生成 PDF「影像证据」截图：左右对照（模型原始输出分割 | 原图红圈）。

    每张图选在器官内注意力总量最大的层面（证据最充分），标题带器官名、
    发现、置信度与层号，医师可据此在原始序列上定位复核。渲染失败不影响
    报告生成。

    红圈定位基于模型原始输出（mask_raw）：后处理掩码的「体内最近邻补全」
    会把器官标签拉成楔形区（如肾标签延伸到侧腹壁、肺标签铺满胸腔），
    外接圆随之偏移/放大；原始输出只含模型真实预测的器官区域，圈更贴合。
    """
    if not file_id:
        return []
    try:
        loaded = _load_overlay(file_id)
        if loaded is None:
            return []
        ct, mask, stats, _spacing = loaded
        # 旧 overlay 无 mask_raw 时 _load_overlay 自动回退后处理掩码
        raw_loaded = _load_overlay(file_id, mode='raw')
        mask_loc = raw_loaded[1] if raw_loaded is not None else mask
        attn = _load_attention(file_id)
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
            vol_lab = mask_loc == lab
            per_z = vol_lab.sum(axis=(1, 2))
            a_per_z = None
            if attn is not None and lab - 1 < attn.shape[0]:
                a_per_z = np.where(vol_lab, attn[lab - 1].astype(np.float64),
                                   0.0).sum(axis=(1, 2))
            if a_per_z is not None and a_per_z.max() > 0:
                # 证据层取「器官内注意力总量」最大的层：模型关注最集中、
                # 病灶证据最充分的层面，而非单纯器官面积最大的层
                zs = np.where(a_per_z > 0)[0]
                score = a_per_z
            else:
                # 该器官无注意力（旧 overlay 或补推理未产出）时回退体素选层
                zs = np.where(per_z > 0)[0]
                score = per_z
            if not len(zs):
                continue
            z = int(zs[np.argmax(score[zs])])
            d = best_diag.get(organ, {})
            items.append((organ, lab, z, d, s.get('volume', 0)))
            if len(items) >= max(1, limit):
                break

        evidence = []
        for organ, lab, z, d, volume in items:
            title = (f"{organ} · {d.get('finding', '异常发现')}"
                     f"　置信度 {float(d.get('confidence') or 0):.1%}"
                     f"　第 {z + 1}/{ct.shape[0]} 层"
                     f"（左：模型原始输出分割｜右：原图红圈病灶定位）")
            try:
                # 对照组：左=仅病灶所在器官的模型原始输出分割填充（标记
                # 受累范围，不画无关器官），右=纯原图+红圈（可看纹理）
                png_seg = _render_evidence_slice(
                    ct, mask_loc, {lab: stats[lab]}, organ_names, palette,
                    organ_conf, z, 'seg')
                png_circle = _render_evidence_slice(
                    ct, mask_loc, {lab: stats[lab]}, organ_names, palette,
                    organ_conf, z, 'circle', attn=attn)
                evidence.append({
                    'title': title,
                    'png': _compose_evidence_pair(png_seg, png_circle),
                })
            except Exception as exc:  # noqa: BLE001 - 单图失败不影响报告
                app.logger.warning('[report] 证据图渲染失败 %s: %s', organ, exc)
        return evidence[:limit]
    except Exception as exc:  # noqa: BLE001
        app.logger.warning('[report] 生成影像证据失败: %s', exc)
        return []


@app.route('/api/report/generate', methods=['POST'])
def generate_report():
    """生成PDF诊断报告（含患者信息与CT影像证据截图）。

    两种取数方式：
    1. 直接传 diagnosis_result（诊断刚完成，前端手上有完整结果）
    2. 只传 case_id（历史记录复核）——从库里读回完整诊断结果，
       这样升级后保存的完整记录才能真正用于事后出报告。
    """
    try:
        data = request.get_json()
        diagnosis_result = data.get('diagnosis_result')
        patient_info = data.get('patient_info', {})
        file_id = data.get('file_id')
        case_id = data.get('case_id')

        if not diagnosis_result and case_id:
            detail = history_manager.get_full_diagnosis(int(case_id)) or {}
            diagnosis_result = detail.get('diagnosis')
            # 库内病例带上原始报告溯源，报告里可标注「院内教学库对照」
            if not patient_info and detail.get('library_context'):
                patient_info = detail['library_context'].get('patient') or {}
            if not file_id:
                file_id = detail.get('file_id')

        if not diagnosis_result:
            return jsonify({'success': False,
                            'message': '缺少诊断结果（该历史记录未保存完整诊断数据，无法重新生成报告）'}), 400

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
    获取历史记录列表（只返回小体积摘要，完整结果走 /api/history/<case_id>）
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        patient_id = request.args.get('patient_id')

        cases = history_manager.list_cases(patient_id=patient_id, limit=limit, offset=offset)

        history_list = []
        for case in cases:
            summary = case.get('summary') or {}
            history_list.append({
                'id': case['id'],
                'case_id': case['id'],
                'patient_id': case['patient_id'],
                'file_name': case['file_name'],
                'file_id': case.get('file_id'),
                # 老记录没有存过来源（NULL），原样返回让前端显示「来源未记录」，
                # 不要兜底成 'upload'——那会把库内导入的老病例误标成外部导入。
                'source': case.get('source'),
                'status': case.get('status'),
                'timestamp': case.get('upload_time'),
                'risk_level': case.get('risk_level') or summary.get('risk_level'),
                'processing_time': case.get('processing_time') or summary.get('processing_time'),
                'findings_count': summary.get('findings_count'),
                'organs_detected_count': summary.get('organs_detected_count'),
                'warnings': summary.get('warnings') or [],
                'diagnosis': {
                    # 前端展示的“诊断结论”摘要
                    'diagnosis': summary.get('diagnosis') or '未发现达到阈值的异常',
                    'diagnoses': summary.get('diagnoses') or [],
                }
            })

        return jsonify({
            'success': True,
            'records': history_list,
            'total': len(history_list)
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f'获取历史记录失败: {str(e)}'}), 500


@app.route('/api/history/<int:case_id>', methods=['GET'])
def get_history_detail(case_id):
    """获取单条历史记录的完整诊断结果。

    列表页只带摘要以控制传输与内存；详情、报告复核、重新出 PDF 都走这里。
    库内病例额外返回 library_context（病例溯源 + 原始报告全文），前端可直接
    复原「AI 结果 vs 人工报告」对照，无需影像库仍在线。
    """
    try:
        case = history_manager.get_case(case_id)
        if not case:
            return jsonify({'success': False, 'message': '记录不存在'}), 404

        summary = case.get('summary') or {}
        full = case.get('diagnosis')
        return jsonify({
            'success': True,
            'record': {
                'id': case['id'],
                'case_id': case['id'],
                'patient_id': case['patient_id'],
                'file_name': case['file_name'],
                'file_id': case.get('file_id'),
                'source': case.get('source'),
                'status': case.get('status'),
                'timestamp': case.get('upload_time'),
                'notes': case.get('notes'),
                'processing_time': case.get('processing_time'),
                'summary': summary,
                'organs_detected': case.get('organs_detected') or [],
                'metadata': case.get('metadata') or {},
                'library_context': case.get('library_context'),
                # 完整诊断结果；升级前的老记录没有存过，返回 None 由前端降级提示
                'diagnosis': full,
                'has_full_result': full is not None,
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'获取历史记录详情失败: {str(e)}'}), 500


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
    # 首次启动即拉起常驻推理守护进程：模型权重只加载一次，后续诊断全部复用，
    # 不再「用一次重新加载一次」。加载期间到达的诊断自动走一次性子进程兜底。
    threading.Thread(target=ensure_infer_daemon, name='radar-infer-daemon',
                     daemon=True).start()
    # 必须开多线程：Werkzeug 开发服务器默认 threaded=False（单线程串行处理），
    # 而 OHIF viewer 打开一个 study 会并发预取 25~100 个 /frames/<n> 请求。
    # 单线程下这些请求排队，实测 20 张并发要 37.6s，后面的请求被浏览器判定超时/
    # 中断，dicomweb-client 直接抛 "Error: request failed"，viewer 卡在加载态。
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False, threaded=True)
