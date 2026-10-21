// RADAR医学影像诊断平台 - 前端交互逻辑

// OHIF 查看器端口。默认 3001，需与后端 app.py 的 OHIF_PORT 及
// start_radar_system.ps1 的 $OhifPort 保持一致（3000 已被本机 Docker 的 attu 占用）。
const OHIF_PORT = 3001;

// 全局状态
const AppState = {
    currentFileId: null,
    currentFileInfo: null,
    currentDiagnosis: null,
    currentTool: 'pan',
    imageData: null,
    viewerState: {
        scale: 1,
        offsetX: 0,
        offsetY: 0,
        windowCenter: 40,
        windowWidth: 400,
        sliceIndex: 50
    }
};

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initUpload();
    initViewer();
    initDiagnose();
    initSettings();
    loadSettings();
    loadHistory();
});

// ============ 导航功能 ============
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const pages = document.querySelectorAll('.page');
    
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const targetPage = item.dataset.page;
            
            // 更新导航状态
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
            
            // 切换页面
            pages.forEach(page => page.classList.remove('active'));
            document.getElementById(`${targetPage}-page`).classList.add('active');
            
            // 加载页面数据
            if (targetPage === 'history') {
                loadHistory();
            }
        });
    });
}

// ============ 文件上传 ============

// 文件夹里必然混有非影像文件，先在前端粗筛掉，避免把上千个无用条目塞进
// FormData 上传（后端仍会用文件头嗅探 + SeriesInstanceUID 分组做最终判定）。
// 注意不能按扩展名白名单筛：PACS 导出的 DICOM 常常没有扩展名（IM0001、纯 UID）。
const NOISE_FILE_RE = /\.(xml|txt|csv|json|pdf|html?|rtf|docx?|xlsx?|pptx?|md|ini|dll|exe|lnk|bmp|mp4|avi|mov)$/i;
const NOISE_NAME_RE = /^(dicomdir|thumbs\.db|desktop\.ini|\.ds_store|readme)/i;

function isPlausibleSliceFile(file) {
    const name = file.name || '';
    if (!name || name.startsWith('._')) return false;
    if (NOISE_NAME_RE.test(name)) return false;
    if (NOISE_FILE_RE.test(name)) return false;
    return true;
}

// 递归展开拖入的文件夹。dataTransfer.files 只含顶层文件，拖文件夹时为空，
// 必须走 items -> webkitGetAsEntry 才能拿到目录树。
async function collectFilesFromDataTransfer(dataTransfer) {
    const items = dataTransfer.items;
    if (!items || !items.length || typeof items[0].webkitGetAsEntry !== 'function') {
        return Array.from(dataTransfer.files || []);
    }

    const entries = [];
    for (let i = 0; i < items.length; i++) {
        const entry = items[i].webkitGetAsEntry();
        if (entry) entries.push(entry);
    }
    if (!entries.length) return Array.from(dataTransfer.files || []);

    const out = [];
    const walk = (entry) => new Promise((resolve) => {
        if (entry.isFile) {
            entry.file(
                (f) => { out.push(f); resolve(); },
                () => resolve()
            );
            return;
        }
        if (!entry.isDirectory) { resolve(); return; }
        const reader = entry.createReader();
        const readBatch = () => {
            reader.readEntries(
                async (children) => {
                    if (!children.length) { resolve(); return; }
                    for (const child of children) await walk(child);
                    // readEntries 单次最多返回 100 条，必须反复读到空为止，
                    // 否则大文件夹会静默漏掉后面的切片
                    readBatch();
                },
                () => resolve()
            );
        };
        readBatch();
    });

    for (const entry of entries) await walk(entry);
    return out;
}

function initUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const folderInput = document.getElementById('folderInput');

    // 拖拽上传（支持直接拖文件夹）
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', async (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const files = await collectFilesFromDataTransfer(e.dataTransfer);
        if (!files.length) {
            showToast('未能读取拖入的内容', 'error');
            return;
        }
        handleFileUpload(files);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files);
        }
        // 清空以便同一批文件可重复选择（否则第二次 change 不触发）
        e.target.value = '';
    });

    if (folderInput) {
        folderInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileUpload(e.target.files);
            }
            e.target.value = '';
        });
    }
}

async function handleFileUpload(fileList) {
    let files = Array.from(fileList);

    // 压缩包整包上传，不参与切片筛选（.zip/.tar.gz 本身就是要交给后端解的）
    const hasArchive = files.some(f => /\.(zip|tar|tgz|tbz2|tar\.gz|tar\.bz2)$/i.test(f.name || ''));
    if (!hasArchive && files.length > 1) {
        const before = files.length;
        files = files.filter(isPlausibleSliceFile);
        const dropped = before - files.length;
        if (!files.length) {
            showToast('所选文件里没有可识别的影像切片', 'error');
            return;
        }
        if (dropped > 0) {
            console.info(`[upload] 前端已忽略 ${dropped} 个非影像文件`);
        }
    }

    const label = files.length > 1 ? `正在上传${files.length}个文件...` : '正在上传文件...';
    showLoading(label);

    // 预检：Chrome 受 Windows MAX_PATH(260 字符) 限制，路径过长的文件能出现在
    // 选择列表里，但上传时读不出内容，请求会被中止并报误导性的
    // "Failed to fetch"（net::ERR_ALPN_NEGOTIATION_FAILED）。这里整文件读一次
    // （与实际上传相同的读取方式）提前暴露问题，给出明确指引。
    if (!hasArchive && files.length) {
        try {
            await files[0].arrayBuffer();
        } catch (err) {
            hideLoading();
            console.error('[upload] 文件内容读取失败:', files[0].name, err);
            showToast('无法读取文件内容：多半是文件路径超过 Windows 260 字符限制。' +
                '请把 DICOM 文件夹复制到较短路径（如 D:\\dcm\\）后重新选择上传', 'error');
            return;
        }
    }

    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            AppState.currentFileId = data.file_id;
            AppState.currentFileInfo = data.info;
            
            displayFileInfo(data.info);
            showViewerSection();
            openOhifViewer(data.info.ohif_study_uid);
            setWorkflowStep('workflowViewerStep');
            document.getElementById('viewerStudyName').textContent = `${data.info.filename} · ${data.info.shape.join(' × ')}`;
            
            showToast(data.message || '上传成功，已打开专业影像查看器', 'success');
        } else {
            showToast(data.message || '上传失败', 'error');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showToast('上传失败: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

function setWorkflowStep(stepId) {
    const step = document.getElementById(stepId);
    if (step) step.classList.add('active', 'done');
}

function openOhifViewer(studyUid) {
    if (!studyUid) {
        showToast('未生成OHIF研究数据', 'error');
        return;
    }
    const viewerUrl = `http://127.0.0.1:${OHIF_PORT}/viewer/local5000?StudyInstanceUIDs=${encodeURIComponent(studyUid)}`;
    document.getElementById('ohifViewerFrame').src = viewerUrl;
    document.getElementById('openOhifNewTab').href = viewerUrl;
}

function displayFileInfo(info) {
    document.getElementById('fileInfo').style.display = 'block';
    document.getElementById('fileName').textContent = info.filename;
    document.getElementById('fileType').textContent = info.file_type.toUpperCase();
    document.getElementById('fileShape').textContent = info.shape.join(' × ');
    document.getElementById('fileSize').textContent = info.size_mb + ' MB';
    document.getElementById('fileSlices').textContent = `${info.slice_count} 张`;
    const spacing = info.spacing;
    document.getElementById('fileSpacing').textContent = spacing
        ? `${spacing.row.toFixed(3)} × ${spacing.column.toFixed(3)} × ${spacing.slice.toFixed(3)} mm`
        : '未提供';

    // 患者与检查信息（DICOM / NIfTI 头）
    renderMetaCard(info.metadata, {
        card: 'fileMeta',
        grid: 'fileMetaGrid',
        source: 'fileMetaSource',
        title: 'fileMetaTitle',
    });
}

// 元数据中不参与逐字段展示的控制键
const META_CONTROL_KEYS = new Set(['label_map', 'source', 'num_instances']);

// 将后端提取的元数据渲染到指定卡片。DICOM 来源标注实例数，NIfTI 标注格式。
function renderMetaCard(metadata, ids) {
    const card = document.getElementById(ids.card);
    const grid = document.getElementById(ids.grid);
    if (!card || !grid) return;

    const fields = metadata
        ? Object.keys(metadata).filter(k => !META_CONTROL_KEYS.has(k) && metadata[k])
        : [];
    if (!fields.length) {
        card.style.display = 'none';
        grid.innerHTML = '';
        return;
    }

    card.style.display = '';
    const labelMap = metadata.label_map || {};
    // 按后端 label_map 的顺序（即 _DICOM_META_FIELDS 顺序）展示，未知键排在后面
    const order = Object.keys(labelMap);
    fields.sort((a, b) => {
        const ia = order.indexOf(a), ib = order.indexOf(b);
        return (ia === -1 ? order.length : ia) - (ib === -1 ? order.length : ib);
    });

    grid.innerHTML = '';
    fields.forEach(key => {
        const item = el('div', 'info-item');
        item.appendChild(el('span', 'info-label', labelMap[key] || key));
        item.appendChild(el('span', 'info-value', String(metadata[key])));
        grid.appendChild(item);
    });

    if (ids.source) {
        const srcEl = document.getElementById(ids.source);
        const src = metadata.source === 'dicom'
            ? `DICOM${metadata.num_instances ? ` · ${metadata.num_instances} 实例` : ''}`
            : (metadata.source === 'nifti' ? 'NIfTI 头' : '元数据');
        srcEl.textContent = src;
    }
    if (ids.title) {
        document.getElementById(ids.title).textContent =
            metadata.source === 'dicom' ? '患者与检查信息' : '影像信息';
    }
}

function showViewerSection() {
    document.getElementById('viewerSection').style.display = 'block';
    document.getElementById('diagnoseSection').style.display = 'block';
}

// ============ 影像查看器 ============
function initViewer() {
    // 主流程使用OHIF Viewer，旧Canvas逻辑不再初始化。
    return;
}
function initCanvasInteraction() {
    const canvas = document.getElementById('imageCanvas');
    let isDragging = false;
    let startX, startY;
    
    canvas.addEventListener('mousedown', (e) => {
        isDragging = true;
        startX = e.clientX - AppState.viewerState.offsetX;
        startY = e.clientY - AppState.viewerState.offsetY;
    });
    
    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = Math.floor(e.clientX - rect.left);
        const y = Math.floor(e.clientY - rect.top);
        
        document.getElementById('mousePos').textContent = `(${x}, ${y})`;
        
        if (isDragging && AppState.currentTool === 'pan') {
            AppState.viewerState.offsetX = e.clientX - startX;
            AppState.viewerState.offsetY = e.clientY - startY;
            drawImage();
        }
    });
    
    canvas.addEventListener('mouseup', () => {
        isDragging = false;
    });
    
    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        if (AppState.currentTool === 'zoom') {
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            AppState.viewerState.scale = Math.max(0.1, Math.min(10, AppState.viewerState.scale * delta));
            drawImage();
        }
    });
}

async function loadImageSlice(sliceIdx) {
    if (!AppState.currentFileId) return;
    
    const viewMode = document.getElementById('viewMode').value;
    const windowCenter = AppState.viewerState.windowCenter;
    const windowWidth = AppState.viewerState.windowWidth;
    
    try {
        const response = await fetch(
            `/api/image/slice/${AppState.currentFileId}/${sliceIdx}?view=${viewMode}&window_center=${windowCenter}&window_width=${windowWidth}`
        );
        
        const data = await response.json();
        
        if (data.success) {
            const img = new Image();
            img.onload = () => {
                AppState.imageData = img;
                drawImage();
                
                // 更新信息
                const shape = AppState.currentFileInfo.shape;
                const maxSlice = viewMode === 'axial' ? shape[0] : 
                               viewMode === 'sagittal' ? shape[2] : shape[1];
                document.getElementById('sliceInfo').textContent = `${sliceIdx + 1} / ${maxSlice}`;
                document.getElementById('viewInfo').textContent = viewMode.toUpperCase();
                document.getElementById('sliceSlider').max = maxSlice - 1;
            };
            img.src = 'data:image/png;base64,' + data.image;
        }
    } catch (error) {
        console.error('Load slice error:', error);
    }
}

function drawImage() {
    if (!AppState.imageData) return;
    
    const canvas = document.getElementById('imageCanvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = AppState.imageData.width;
    canvas.height = AppState.imageData.height;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    
    ctx.translate(canvas.width / 2 + AppState.viewerState.offsetX, 
                  canvas.height / 2 + AppState.viewerState.offsetY);
    ctx.scale(AppState.viewerState.scale, AppState.viewerState.scale);
    ctx.translate(-canvas.width / 2, -canvas.height / 2);
    
    ctx.drawImage(AppState.imageData, 0, 0);
    ctx.restore();
}

function updateImage() {
    loadImageSlice(AppState.viewerState.sliceIndex);
}

function resetViewer() {
    AppState.viewerState.scale = 1;
    AppState.viewerState.offsetX = 0;
    AppState.viewerState.offsetY = 0;
    drawImage();
}

function updateCursor(tool) {
    const canvas = document.getElementById('imageCanvas');
    const cursors = {
        'pan': 'move',
        'zoom': 'zoom-in',
        'window': 'crosshair'
    };
    canvas.style.cursor = cursors[tool] || 'default';
}

async function loadMultiviewImages() {
    if (!AppState.currentFileId) return;
    
    showLoading('加载三视图...');
    
    try {
        const response = await fetch(`/api/image/multiview/${AppState.currentFileId}`);
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('axialImage').src = 'data:image/png;base64,' + data.views.axial.image;
            document.getElementById('sagittalImage').src = 'data:image/png;base64,' + data.views.sagittal.image;
            document.getElementById('coronalImage').src = 'data:image/png;base64,' + data.views.coronal.image;
        }
    } catch (error) {
        console.error('Load multiview error:', error);
        showToast('加载三视图失败', 'error');
    } finally {
        hideLoading();
    }
}

function initSettings() {
    const threshold = document.getElementById('confidenceSetting');
    threshold.addEventListener('input', () => {
        document.getElementById('confidenceSettingValue').textContent = `${threshold.value}%`;
    });
    document.getElementById('saveSettingsBtn').addEventListener('click', saveSettings);
}

async function loadSettings() {
    const response = await fetch('/api/settings');
    const data = await response.json();
    if (!data.success) return;
    document.getElementById('themeSetting').value = data.settings.theme;
    document.getElementById('languageSetting').value = data.settings.language;
    document.getElementById('confidenceSetting').value = Math.round(data.settings.confidence_threshold * 100);
    document.getElementById('confidenceSettingValue').textContent = `${Math.round(data.settings.confidence_threshold * 100)}%`;
    document.getElementById('gpuSetting').checked = data.settings.use_gpu;
    applyTheme(data.settings.theme);
}

async function saveSettings() {
    const payload = {
        theme: document.getElementById('themeSetting').value,
        language: document.getElementById('languageSetting').value,
        confidence_threshold: Number(document.getElementById('confidenceSetting').value) / 100,
        use_gpu: document.getElementById('gpuSetting').checked
    };
    const response = await fetch('/api/settings', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (data.success) {
        applyTheme(data.settings.theme);
        showToast('设置已保存并生效', 'success');
    } else {
        showToast(data.message || '设置保存失败', 'error');
    }
}

function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
}


function initDiagnose() {
    const diagnoseBtn = document.getElementById('diagnoseBtn');
    const generateReportBtn = document.getElementById('generateReportBtn');
    
    diagnoseBtn.addEventListener('click', performDiagnosis);
    generateReportBtn.addEventListener('click', generateReport);

    // 发现列表 / 证据列表 的展开-收起
    const findingsBtn = document.getElementById('toggleFindingsBtn');
    if (findingsBtn) {
        findingsBtn.addEventListener('click', () => {
            const grid = document.getElementById('findingsList');
            const collapsed = grid.classList.toggle('collapsed');
            findingsBtn.textContent = collapsed ? '展开全部' : '收起';
        });
    }
    const explBtn = document.getElementById('toggleExplanationBtn');
    if (explBtn) {
        explBtn.addEventListener('click', () => {
            const list = document.getElementById('evidenceList');
            const collapsed = list.classList.toggle('collapsed');
            explBtn.textContent = collapsed ? '展开全部' : '收起';
        });
    }
}

async function performDiagnosis() {
    if (!AppState.currentFileId) {
        showToast('请先上传影像文件', 'warning');
        return;
    }
    
    showLoading('AI正在分析影像...');
    
    try {
        const response = await fetch(`/api/diagnose/${AppState.currentFileId}`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            AppState.currentDiagnosis = data.result;
            displayDiagnosisResult(data.result);
            setWorkflowStep('workflowDiagnosisStep');
            setWorkflowStep('workflowReportStep');
            showToast('诊断完成，结果已生成', 'success');
        } else {
            showToast(data.message || '诊断失败', 'error');
        }
    } catch (error) {
        console.error('Diagnosis error:', error);
        showToast('诊断失败: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// ---- 诊断结果可视化辅助 ----
const SEVERITY_META = {
    high: { label: '高风险', cls: 'sev-high' },
    medium: { label: '中风险', cls: 'sev-medium' },
    low: { label: '低风险', cls: 'sev-low' },
};

function sevClass(sev) {
    return (SEVERITY_META[sev] || SEVERITY_META.low).cls;
}

function sevLabel(sev) {
    return (SEVERITY_META[sev] || SEVERITY_META.low).label;
}

// 总体风险分级：取所有阳性项中最高严重度；无阳性项则为「未见异常」
function overallSeverity(diagnoses) {
    if (!diagnoses.length) return null;
    if (diagnoses.some(d => d.severity === 'high')) return 'high';
    if (diagnoses.some(d => d.severity === 'medium')) return 'medium';
    return 'low';
}

function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
}

function pct(v) {
    return (Number(v || 0) * 100).toFixed(1) + '%';
}

function displayDiagnosisResult(result) {
    const section = document.getElementById('resultSection');
    section.style.display = 'block';

    const diagnoses = result.diagnoses || [];
    const explanation = result.explanation || {};
    const threshold = result.settings?.confidence_threshold;

    // 引擎标识
    const engineEl = document.getElementById('resultEngine');
    if (engineEl) {
        const dev = explanation.device || result.raw_output?.device || '';
        engineEl.textContent = `${result.engine || 'RADAR'}${dev ? ' · ' + dev : ''}`;
    }

    // 患者与检查信息（随诊断结果返回，DICOM 时含真实患者ID）
    renderMetaCard(result.metadata, {
        card: 'resultMetaCard',
        grid: 'resultMetaGrid',
        source: 'resultMetaSource',
    });

    // ===== 总体印象横幅 =====
    const overall = overallSeverity(diagnoses);
    const banner = document.getElementById('impressionBanner');
    banner.className = 'impression-banner card ' + (overall ? sevClass(overall) : 'sev-low');
    document.getElementById('overallRiskValue').textContent = overall ? sevLabel(overall) : '未见异常';

    const organs = result.organs_detected || [];
    const headlineEl = document.getElementById('impressionHeadline');
    const detailEl = document.getElementById('impressionDetail');
    if (!diagnoses.length) {
        headlineEl.textContent = '未发现达到阈值的明显异常';
        detailEl.textContent = `本次共评估 ${explanation.total_items ?? '—'} 个诊断项，均在置信度阈值${threshold != null ? `（${pct(threshold)}）` : ''}以下。RADAR 分割识别到 ${organs.length} 个器官，建议结合临床常规随访。`;
    } else {
        const topOrgans = [...new Set(diagnoses.map(d => d.organ))];
        headlineEl.textContent = `检出 ${diagnoses.length} 项阳性发现，累及 ${topOrgans.length} 个器官`;
        const top3 = diagnoses.slice(0, 3).map(d => `${d.organ}·${d.finding}（${pct(d.confidence)}）`);
        detailEl.textContent = `主要为：${top3.join('；')}${diagnoses.length > 3 ? ' 等' : ''}。`;
    }

    // ===== 关键指标 =====
    const maxConf = diagnoses.length ? Math.max(...diagnoses.map(d => Number(d.confidence) || 0)) : 0;
    const topFinding = diagnoses.length ? diagnoses[0] : null;
    document.getElementById('lesionCount').textContent = diagnoses.length;
    document.getElementById('lesionCountHint').textContent = threshold != null
        ? `置信度 ≥ ${pct(threshold)}` : '超过置信度阈值';
    document.getElementById('confidenceValue').textContent = diagnoses.length ? pct(maxConf) : '—';
    document.getElementById('confidenceHint').textContent = topFinding
        ? `${topFinding.organ}·${topFinding.finding}` : '无阳性项';
    const sevEl = document.getElementById('severityValue');
    sevEl.textContent = overall ? sevLabel(overall) : '正常';
    sevEl.className = 'metric-value metric-value-sm ' + (overall ? sevClass(overall) : 'is-normal');
    document.getElementById('organCount').textContent = organs.length;

    // ===== 影像学发现（卡片网格）=====
    const grid = document.getElementById('findingsList');
    grid.innerHTML = '';
    grid.className = 'findings-grid' + (diagnoses.length > 6 ? ' collapsed' : '');
    const findingsBtn = document.getElementById('toggleFindingsBtn');
    if (findingsBtn) {
        findingsBtn.style.display = diagnoses.length > 6 ? '' : 'none';
        findingsBtn.textContent = '展开全部';
    }
    if (!diagnoses.length) {
        grid.appendChild(el('div', 'empty-state', '未见达到阈值的影像学发现。'));
    } else {
        diagnoses.forEach(d => {
            const card = el('div', 'finding-card ' + sevClass(d.severity));
            const top = el('div', 'finding-top');
            top.appendChild(el('span', 'finding-organ', d.organ));
            top.appendChild(el('span', 'finding-conf', pct(d.confidence)));
            card.appendChild(top);

            const nameWrap = el('div');
            nameWrap.appendChild(el('div', 'finding-name', d.finding));
            if (d.finding_en) nameWrap.appendChild(el('div', 'finding-name-en', d.finding_en));
            card.appendChild(nameWrap);

            const bar = el('div', 'finding-bar');
            const fill = el('span');
            fill.style.width = Math.max(2, Math.round((Number(d.confidence) || 0) * 100)) + '%';
            bar.appendChild(fill);
            card.appendChild(bar);

            const foot = el('div', 'finding-foot');
            foot.appendChild(el('span', 'severity-tag', sevLabel(d.severity)));
            foot.appendChild(el('span', 'finding-rec', d.recommendation || '建议专科进一步评估。'));
            card.appendChild(foot);

            grid.appendChild(card);
        });
    }

    // ===== 器官分割 =====
    const chips = document.getElementById('organChips');
    chips.innerHTML = '';
    const organNote = document.getElementById('organNote');
    const affectedOrgans = new Set(diagnoses.map(d => d.organ));
    if (!organs.length) {
        chips.appendChild(el('div', 'empty-state', '未获取到器官分割结果。'));
        if (organNote) organNote.textContent = '';
    } else {
        organs.forEach(o => {
            chips.appendChild(el('span', 'organ-chip' + (affectedOrgans.has(o) ? ' is-target' : ''), o));
        });
        if (organNote) organNote.textContent = `共 ${organs.length} 个器官 · 蓝色为受累器官`;
    }

    // ===== Top-N 得分条形图 =====
    const chart = document.getElementById('topScoresChart');
    chart.innerHTML = '';
    const topScores = explanation.top_scores || [];
    if (!topScores.length) {
        chart.appendChild(el('div', 'empty-state', '暂无得分排行数据。'));
    } else {
        topScores.forEach(s => {
            const positive = Number(s.confidence) >= (threshold ?? 0.5);
            const row = el('div', 'score-row' + (positive ? ' is-positive sev-medium' : ''));
            row.appendChild(el('span', 'score-label', s.item));
            row.appendChild(el('span', 'score-val', pct(s.confidence)));
            const track = el('div', 'score-track');
            const fill = el('span');
            fill.style.width = Math.max(2, Math.round((Number(s.confidence) || 0) * 100)) + '%';
            track.appendChild(fill);
            row.appendChild(track);
            chart.appendChild(row);
        });
    }

    // ===== 建议处理 =====
    const recList = document.getElementById('recommendationsList');
    recList.innerHTML = '';
    (diagnoses.length ? diagnoses : [{ recommendation: '未见明显异常，建议结合临床情况进行常规随访。' }]).forEach(d => {
        recList.appendChild(el('li', null, d.recommendation || '建议由专科医生进一步评估。'));
    });

    // ===== 可解释性与证据 =====
    document.getElementById('explanationMethod').textContent = explanation.method || '';
    document.getElementById('explanationSummary').textContent = explanation.summary || '';

    const evidenceList = document.getElementById('evidenceList');
    evidenceList.innerHTML = '';
    evidenceList.className = 'evidence-list' + (diagnoses.length > 3 ? ' collapsed' : '');
    const explBtn = document.getElementById('toggleExplanationBtn');
    if (explBtn) {
        explBtn.style.display = diagnoses.length > 3 ? '' : 'none';
        explBtn.textContent = '展开全部';
    }
    if (!diagnoses.length) {
        evidenceList.appendChild(el('div', 'empty-state', '无阳性项证据。'));
    } else {
        diagnoses.forEach(d => {
            const item = el('div', 'evidence-item');
            const strong = el('strong', null, `${d.organ}·${d.finding}：`);
            item.appendChild(strong);
            item.appendChild(document.createTextNode(d.evidence || '未提供具体影像证据'));
            evidenceList.appendChild(item);
        });
    }

    const limitsBox = document.getElementById('limitationsBox');
    limitsBox.innerHTML = '';
    const limits = explanation.limitations || [];
    if (limits.length) {
        limitsBox.appendChild(el('div', 'limitations-title', '使用限制'));
        const ul = document.createElement('ul');
        limits.forEach(t => ul.appendChild(el('li', null, t)));
        limitsBox.appendChild(ul);
        limitsBox.style.display = '';
    } else {
        limitsBox.style.display = 'none';
    }

    const meta = document.getElementById('inferenceMeta');
    meta.innerHTML = '';
    const procTime = explanation.processing_time_sec ?? result.raw_output?.processing_time;
    const metaItems = [
        procTime != null ? `推理耗时 ${procTime} s` : '',
        explanation.device ? `设备 ${explanation.device}` : '',
        explanation.total_items != null ? `评估诊断项 ${explanation.total_items}` : '',
        result.patient_id ? `患者 ${result.patient_id}` : '',
    ].filter(Boolean);
    metaItems.forEach(t => meta.appendChild(el('span', null, t)));

    // 分割叠加交互看片（依赖诊断时导出的 overlay.npz）
    initOverlayViewer(result);

    section.scrollIntoView({ behavior: 'smooth' });
}

// ============ 分割叠加交互看片 ============
// 叠加数据（预处理 CT 灰度 + 器官标签）由推理引擎在诊断时导出为 overlay.npz，
// 与 stitched_mask 逐体素对齐，故前端按切片直接叠加即可，无需坐标逆变换。
const OverlayState = {
    fileId: null,
    meta: null,
    legend: [],
    affectedIds: new Set(),
    // 器官标签 -> 该器官最高置信度（注意力热力模式用）
    organConfidence: new Map(),
    // 器官名 -> 该器官最高置信度的诊断项（3D 病灶标注用）
    organFinding: new Map(),
    visible: new Set(),
    slice: 0,
    opacity: 0.45,
    // 2D 渲染模式：fill 填充 / contour 轮廓 / heat 注意力热力 / ct 隐藏叠加
    renderMode: 'fill',
    // 分割来源：full 后处理完整分割 / raw 模型原始输出 / fine TotalSegmentator
    segSource: 'full',
    // 3D 场景当前网格对应的分割来源（切换来源时需重建）
    scene3dSource: null,
    // 视图模式：2d 切片 / 3d 重建
    mode: '2d',
    current: null,
    cache: new Map(),
    cacheOrder: [],
    playTimer: null,
    loadSeq: 0,
    bound: false,
    scene3d: null,
    threeLoading: null,
};

function setOverlayLoading(on) {
    const l = document.getElementById('overlayLoading');
    if (l) l.classList.toggle('is-visible', !!on);
}

async function initOverlayViewer(result) {
    const card = document.getElementById('overlayCard');
    const fileId = AppState.currentFileId;
    if (!fileId || !result || !result.overlay) {
        if (card) card.style.display = 'none';
        return;
    }

    stopOverlayPlay();
    OverlayState.fileId = fileId;
    OverlayState.cache.clear();
    OverlayState.cacheOrder = [];
    OverlayState.current = null;
    OverlayState.slice = 0;
    OverlayState.loadSeq = 0;
    // 新诊断的网格与旧文件不同：标记 3D 场景需重建，避免沿用旧器官网格
    OverlayState.scene3dSource = null;

    // 器官 -> 最高置信度：注意力热力模式按它给受累器官着色
    OverlayState.organConfidence = new Map();
    // 器官 -> 最高置信度的诊断项：3D 病灶标注显示发现名与置信度
    OverlayState.organFinding = new Map();
    (result.diagnoses || []).forEach(d => {
        const c = Number(d.confidence) || 0;
        const prev = OverlayState.organConfidence.get(d.organ);
        if (prev == null || c > prev) {
            OverlayState.organConfidence.set(d.organ, c);
            OverlayState.organFinding.set(d.organ, d);
        }
    });

    const affectedNames = new Set((result.diagnoses || []).map(d => d.organ));

    card.style.display = '';
    bindOverlayEvents();

    setOverlayLoading(true);
    try {
        // 按当前选择的分割来源拉 meta；来源不可用（如 fine 未生成）回退 full
        let resp = await fetch(
            `/api/overlay/${encodeURIComponent(fileId)}/meta?mode=${encodeURIComponent(OverlayState.segSource)}`
        );
        if (!resp.ok && OverlayState.segSource !== 'full') {
            OverlayState.segSource = 'full';
            resp = await fetch(`/api/overlay/${encodeURIComponent(fileId)}/meta`);
        }
        const data = await resp.json();
        if (!data.success) { card.style.display = 'none'; return; }

        // 分割来源下拉：精细分割仅在后台生成完成后出现
        const segSel = document.getElementById('overlaySegSource');
        if (segSel) {
            const fineOpt = segSel.querySelector('option[value="fine"]');
            if (fineOpt) fineOpt.hidden = !data.has_fine;
            if (segSel.value !== OverlayState.segSource) segSel.value = OverlayState.segSource;
        }

        OverlayState.meta = data;
        OverlayState.legend = data.organ_legend || [];
        OverlayState.affectedIds = new Set(
            OverlayState.legend.filter(o => affectedNames.has(o.name)).map(o => o.id)
        );
        OverlayState.visible = new Set(OverlayState.legend.map(o => o.id));

        buildOverlayLegend();
        renderOverlayFindings(result.diagnoses || []);
        const sliceInput = document.getElementById('overlaySlice');
        sliceInput.max = String(Math.max(0, data.num_slices - 1));
        document.getElementById('overlaySize').textContent =
            `${data.width} × ${data.height} × ${data.num_slices}`;

        // 初始定位到器官标注最丰富的切片，而非几何中层（边缘层常近乎空白）
        const best = Number.isInteger(data.best_slice) ? data.best_slice : Math.floor(data.num_slices / 2);
        OverlayState.slice = Math.min(Math.max(0, best), data.num_slices - 1);
        sliceInput.value = String(OverlayState.slice);
        await loadOverlaySlice(OverlayState.slice);
    } catch (e) {
        console.error('Overlay init error:', e);
        card.style.display = 'none';
    } finally {
        setOverlayLoading(false);
    }
}

function buildOverlayLegend() {
    const box = document.getElementById('overlayLegend');
    box.innerHTML = '';
    const items = OverlayState.legend;
    document.getElementById('overlayLegendCount').textContent = `${items.length} 个`;
    items.forEach(o => {
        const label = el('label', 'overlay-legend-item');
        label.title = `层 ${o.z_min + 1}–${o.z_max + 1} · ${o.volume} 体素`;
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = OverlayState.visible.has(o.id);
        cb.addEventListener('change', () => {
            if (cb.checked) OverlayState.visible.add(o.id);
            else OverlayState.visible.delete(o.id);
            label.classList.toggle('is-off', !cb.checked);
            renderOverlay();
            syncScene3dVisibility();
        });
        const sw = el('span', 'overlay-swatch');
        sw.style.background = `rgb(${o.color.join(',')})`;
        label.appendChild(cb);
        label.appendChild(sw);
        label.appendChild(el('span', 'overlay-legend-name', o.name));
        if (OverlayState.affectedIds.has(o.id)) {
            label.appendChild(el('span', 'overlay-legend-tag', '受累'));
        }
        // 跳转到该器官正中层面
        const jump = document.createElement('button');
        jump.type = 'button';
        jump.className = 'overlay-legend-jump';
        jump.textContent = '跳转';
        jump.title = `跳到${o.name}正中层面`;
        jump.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            const target = Math.round((o.z_min + o.z_max) / 2);
            document.getElementById('overlaySlice').value = String(target);
            stopOverlayPlay();
            loadOverlaySlice(target);
        });
        label.appendChild(jump);
        box.appendChild(label);
    });
}

// 看片右侧「影像学发现」速览列：诊断结果与切片同屏对照，无需上下翻动页面。
// 点击条目跳到该器官正中层面，并在热力模式下高亮对应器官。
function renderOverlayFindings(diagnoses) {
    const list = document.getElementById('overlayFindingsList');
    const countEl = document.getElementById('overlayFindingsCount');
    if (!list) return;
    list.innerHTML = '';
    if (countEl) countEl.textContent = diagnoses.length ? `${diagnoses.length} 项` : '';

    if (!diagnoses.length) {
        list.appendChild(el('div', 'ovf-empty', '未见达到阈值的影像学发现'));
        return;
    }

    // 器官名 -> Z 范围（图例里有），用于「定位」跳转
    const zRange = {};
    OverlayState.legend.forEach(o => { zRange[o.name] = o; });

    diagnoses.forEach((d, i) => {
        const item = el('div', 'ovf-item ' + sevClass(d.severity));
        item.dataset.organ = d.organ;

        const top = el('div', 'ovf-top');
        top.appendChild(el('span', 'ovf-organ', d.organ));
        top.appendChild(el('span', 'ovf-conf', pct(d.confidence)));
        item.appendChild(top);

        item.appendChild(el('div', 'ovf-name', d.finding));

        const bar = el('div', 'ovf-bar');
        const fill = el('span');
        fill.style.width = Math.max(2, Math.round((Number(d.confidence) || 0) * 100)) + '%';
        bar.appendChild(fill);
        item.appendChild(bar);

        const meta = el('div', 'ovf-meta');
        meta.appendChild(el('span', null, sevLabel(d.severity)));
        const o = zRange[d.organ];
        if (o) {
            const jump = el('button', 'ovf-jump', '定位');
            jump.type = 'button';
            jump.title = `跳到${d.organ}正中层面`;
            jump.addEventListener('click', (e) => {
                e.stopPropagation();
                const target = Math.round((o.z_min + o.z_max) / 2);
                document.getElementById('overlaySlice').value = String(target);
                stopOverlayPlay();
                loadOverlaySlice(target);
            });
            meta.appendChild(jump);
        }
        item.appendChild(meta);

        // 点击整条：热力模式下单独高亮该器官
        item.addEventListener('click', () => {
            const on = !item.classList.contains('is-highlight');
            list.querySelectorAll('.ovf-item').forEach(n => n.classList.remove('is-highlight'));
            if (!on) {
                // 再次点击取消高亮 -> 恢复显示全部受累器官
                setOverlayOrgans('affected');
                return;
            }
            item.classList.add('is-highlight');
            const ids = new Set(OverlayState.legend.filter(x => x.name === d.organ).map(x => x.id));
            if (ids.size) {
                OverlayState.visible = ids;
                syncLegendCheckboxes();
                renderOverlay();
                syncScene3dVisibility();
            }
        });

        list.appendChild(item);
    });
}

// 把 OverlayState.visible 同步回图例复选框（程序化改变可见性后调用）
function syncLegendCheckboxes() {
    const box = document.getElementById('overlayLegend');
    if (!box) return;
    box.querySelectorAll('.overlay-legend-item').forEach((label, i) => {
        const o = OverlayState.legend[i];
        if (!o) return;
        const on = OverlayState.visible.has(o.id);
        const cb = label.querySelector('input');
        if (cb) cb.checked = on;
        label.classList.toggle('is-off', !on);
    });
}

function decodePng(b64) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = 'data:image/png;base64,' + b64;
    });
}

function decodeMask(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
}

function putOverlayCache(idx, entry) {
    OverlayState.cache.set(idx, entry);
    OverlayState.cacheOrder.push(idx);
    while (OverlayState.cacheOrder.length > 8) {
        OverlayState.cache.delete(OverlayState.cacheOrder.shift());
    }
}

async function loadOverlaySlice(idx) {
    const meta = OverlayState.meta;
    if (!meta) return;
    OverlayState.slice = idx;
    document.getElementById('overlaySliceVal').textContent = `${idx + 1} / ${meta.num_slices}`;

    if (OverlayState.cache.has(idx)) {
        OverlayState.current = OverlayState.cache.get(idx);
        renderOverlay();
        return;
    }

    const token = ++OverlayState.loadSeq;
    setOverlayLoading(true);
    try {
        const resp = await fetch(
            `/api/overlay/${encodeURIComponent(OverlayState.fileId)}/slice/${idx}`
            + `?mode=${encodeURIComponent(OverlayState.segSource)}`
        );
        const data = await resp.json();
        if (!data.success) return;
        const [ct, mask] = await Promise.all([
            decodePng(data.ct),
            Promise.resolve(decodeMask(data.mask)),
        ]);
        const entry = { ct, mask, width: data.width, height: data.height };
        putOverlayCache(idx, entry);
        if (token !== OverlayState.loadSeq) return; // 已有更新的请求，丢弃本次渲染
        OverlayState.current = entry;
        renderOverlay();
    } catch (e) {
        console.error('loadOverlaySlice error:', e);
    } finally {
        if (token === OverlayState.loadSeq) setOverlayLoading(false);
    }
}

function isBoundary(mask, p, x, y, w, h) {
    const lab = mask[p];
    if (x > 0 && mask[p - 1] !== lab) return true;
    if (x < w - 1 && mask[p + 1] !== lab) return true;
    if (y > 0 && mask[p - w] !== lab) return true;
    if (y < h - 1 && mask[p + w] !== lab) return true;
    return false;
}

// 注意力热力配色：置信度 0→1 走 蓝→青→绿→黄→红（与图例渐变一致）
function heatColor(t) {
    const c = Math.min(1, Math.max(0, t));
    const stops = [
        [0.0, 0, 64, 255],
        [0.25, 0, 220, 255],
        [0.5, 80, 255, 80],
        [0.75, 255, 230, 0],
        [1.0, 255, 40, 0],
    ];
    for (let i = 1; i < stops.length; i++) {
        if (c <= stops[i][0]) {
            const [p0, r0, g0, b0] = stops[i - 1];
            const [p1, r1, g1, b1] = stops[i];
            const k = (c - p0) / (p1 - p0);
            return [
                r0 + (r1 - r0) * k,
                g0 + (g1 - g0) * k,
                b0 + (b1 - b0) * k,
            ];
        }
    }
    return [255, 40, 0];
}

function renderOverlay() {
    const canvas = document.getElementById('overlayCanvas');
    const cur = OverlayState.current;
    if (!canvas || !cur) return;
    const { ct, mask, width, height } = cur;

    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(ct, 0, 0, width, height);

    const rmode = OverlayState.renderMode;
    if (rmode === 'ct') return;

    const base = ctx.getImageData(0, 0, width, height);
    const out = ctx.createImageData(width, height);
    out.data.set(base.data);
    const d = out.data;

    const palette = {};
    OverlayState.legend.forEach(o => { palette[o.id] = o.color; });
    // 热力模式：器官标签 -> 该器官最高置信度（无诊断项的器官不上色）
    const heatConf = {};
    OverlayState.legend.forEach(o => {
        const c = OverlayState.organConfidence.get(o.name);
        if (c != null) heatConf[o.id] = c;
    });
    const alpha = OverlayState.opacity;
    const contour = rmode === 'contour';
    const heat = rmode === 'heat';

    // 热力模式：一次性算出「每个像素到器官内部核心的距离」，中心浓、
    // 边缘淡（羽化），避免色块生硬切割 CT 纹理。按切片缓存。
    let distMap = null;
    if (heat && Object.keys(heatConf).length) {
        distMap = getOverlayDistMap();
    }

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const p = y * width + x;
            const lab = mask[p];
            if (lab === 0 || !OverlayState.visible.has(lab)) continue;
            if (contour && !isBoundary(mask, p, x, y, width, height)) continue;
            const i = p * 4;
            if (heat) {
                // 只渲染有阳性诊断项的器官
                const conf = heatConf[lab];
                if (conf == null) continue;
                const falloff = Math.min(1, distMap[p] / 6);
                const c = heatColor(conf);
                const a = alpha * (0.45 + 0.55 * falloff);
                d[i]     = d[i]     * (1 - a) + c[0] * a;
                d[i + 1] = d[i + 1] * (1 - a) + c[1] * a;
                d[i + 2] = d[i + 2] * (1 - a) + c[2] * a;
            } else if (contour) {
                const c = palette[lab] || [255, 0, 255];
                d[i] = c[0]; d[i + 1] = c[1]; d[i + 2] = c[2]; d[i + 3] = 255;
            } else {
                const c = palette[lab] || [255, 0, 255];
                d[i]     = d[i]     * (1 - alpha) + c[0] * alpha;
                d[i + 1] = d[i + 1] * (1 - alpha) + c[1] * alpha;
                d[i + 2] = d[i + 2] * (1 - alpha) + c[2] * alpha;
            }
        }
    }
    ctx.putImageData(out, 0, 0);
}

// 当前切片的羽化距离图（Uint8，值=到异标签的切比雪夫距离，封顶 8）。
// 两遍扫描（左上→右下、右下→左上）近似距离变换，O(W*H)，随切片缓存。
function getOverlayDistMap() {
    const cur = OverlayState.current;
    if (cur.distMap) return cur.distMap;
    const { mask, width: w, height: h } = cur;
    const CAP = 8;
    const dist = new Uint8Array(w * h);
    // 第一遍：边界像素置 0，其余先置 CAP
    for (let p = 0; p < w * h; p++) {
        dist[p] = isBoundary(mask, p, p % w, Math.floor(p / w), w, h) ? 0 : CAP;
    }
    // 左上→右下
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const p = y * w + x;
            let v = dist[p];
            if (x > 0) v = Math.min(v, dist[p - 1] + 1);
            if (y > 0) v = Math.min(v, dist[p - w] + 1);
            dist[p] = v;
        }
    }
    // 右下→左上
    for (let y = h - 1; y >= 0; y--) {
        for (let x = w - 1; x >= 0; x--) {
            const p = y * w + x;
            let v = dist[p];
            if (x < w - 1) v = Math.min(v, dist[p + 1] + 1);
            if (y < h - 1) v = Math.min(v, dist[p + w] + 1);
            dist[p] = v > CAP ? CAP : v;
        }
    }
    cur.distMap = dist;
    return dist;
}

function handleOverlayHover(e) {
    const cur = OverlayState.current;
    const tip = document.getElementById('overlayCursorTip');
    if (!cur || !tip) return;
    const canvas = document.getElementById('overlayCanvas');
    const rect = canvas.getBoundingClientRect();
    const px = Math.floor((e.clientX - rect.left) * (canvas.width / rect.width));
    const py = Math.floor((e.clientY - rect.top) * (canvas.height / rect.height));
    if (px < 0 || py < 0 || px >= cur.width || py >= cur.height) {
        tip.style.display = 'none';
        return;
    }
    const lab = cur.mask[py * cur.width + px];
    const organ = OverlayState.legend.find(o => o.id === lab);
    const name = lab === 0 ? '背景' : (organ ? organ.name : `器官${lab}`);
    tip.textContent = `${name} · 层 ${OverlayState.slice + 1} · (${px}, ${py})`;
    tip.style.display = 'block';
    const stageRect = document.getElementById('overlayStage').getBoundingClientRect();
    tip.style.left = (e.clientX - stageRect.left + 14) + 'px';
    tip.style.top = (e.clientY - stageRect.top + 14) + 'px';
}

function toggleOverlayPlay() {
    const btn = document.getElementById('overlayPlayBtn');
    if (OverlayState.playTimer) {
        stopOverlayPlay();
    } else {
        btn.textContent = '⏸ 暂停';
        OverlayState.playTimer = setInterval(() => {
            const n = OverlayState.meta.num_slices;
            const next = (OverlayState.slice + 1) % n;
            document.getElementById('overlaySlice').value = String(next);
            loadOverlaySlice(next);
        }, 180);
    }
}

function stopOverlayPlay() {
    if (OverlayState.playTimer) {
        clearInterval(OverlayState.playTimer);
        OverlayState.playTimer = null;
        const btn = document.getElementById('overlayPlayBtn');
        if (btn) btn.textContent = '▶ 播放';
    }
}

function setOverlayOrgans(which) {
    if (which === 'affected' && OverlayState.affectedIds.size > 0) {
        OverlayState.visible = new Set(OverlayState.affectedIds);
    } else {
        OverlayState.visible = new Set(OverlayState.legend.map(o => o.id));
    }
    // 可见性被整体重置后，发现列的单器官高亮态不再成立，一并清除
    document.querySelectorAll('#overlayFindingsList .ovf-item.is-highlight')
        .forEach(n => n.classList.remove('is-highlight'));
    syncLegendCheckboxes();
    renderOverlay();
    syncScene3dVisibility();
}

// ------------------------------------------------------------------ //
// 3D 重建模式：three.js 表面网格渲染（marching cubes 网格来自后端）
// ------------------------------------------------------------------ //

function loadThree() {
    if (window.THREE) return Promise.resolve(window.THREE);
    if (OverlayState.threeLoading) return OverlayState.threeLoading;
    OverlayState.threeLoading = new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js';
        s.onload = () => resolve(window.THREE);
        s.onerror = () => {
            OverlayState.threeLoading = null;
            reject(new Error('three.js 加载失败（需联网）'));
        };
        document.head.appendChild(s);
    });
    return OverlayState.threeLoading;
}

function initScene3d() {
    const THREE = window.THREE;
    const host = document.getElementById('overlayStage3d');
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05070c);
    const camera = new THREE.PerspectiveCamera(45, 1, 1, 5000);
    scene.add(new THREE.AmbientLight(0xffffff, 1.1));
    const dir = new THREE.DirectionalLight(0xffffff, 1.6);
    dir.position.set(1, 1.4, 1);
    scene.add(dir);
    const dir2 = new THREE.DirectionalLight(0x8899ff, 0.5);
    dir2.position.set(-1, -0.6, -1);
    scene.add(dir2);

    const group = new THREE.Group();
    scene.add(group);

    const state = {
        THREE, renderer, scene, camera, group,
        meshes: new Map(),       // organId -> THREE.Mesh
        yaw: 0.6, pitch: 0.35, dist: 0, target: new THREE.Vector3(),
        panX: 0, panY: 0, built: false,
    };

    // 手工轨道控制：左键旋转、右键平移、滚轮缩放
    const dom = renderer.domElement;
    let drag = null;
    dom.addEventListener('contextmenu', e => e.preventDefault());
    dom.addEventListener('pointerdown', e => {
        drag = { x: e.clientX, y: e.clientY, btn: e.button };
        dom.setPointerCapture(e.pointerId);
    });
    dom.addEventListener('pointermove', e => {
        if (!drag) return;
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        drag.x = e.clientX; drag.y = e.clientY;
        if (drag.btn === 2) {
            state.panX -= dx * state.dist * 0.0012;
            state.panY += dy * state.dist * 0.0012;
        } else {
            state.yaw -= dx * 0.006;
            state.pitch = Math.min(1.45, Math.max(-1.45, state.pitch + dy * 0.006));
        }
        updateCamera3d();
    });
    dom.addEventListener('pointerup', () => { drag = null; });
    dom.addEventListener('wheel', e => {
        e.preventDefault();
        state.dist *= e.deltaY > 0 ? 1.1 : 0.9;
        state.dist = Math.min(3000, Math.max(60, state.dist));
        updateCamera3d();
    }, { passive: false });

    function updateCamera3d() {
        const t = state.target;
        camera.position.set(
            t.x + state.panX + state.dist * Math.cos(state.pitch) * Math.sin(state.yaw),
            t.y + state.panY + state.dist * Math.sin(state.pitch),
            t.z + state.dist * Math.cos(state.pitch) * Math.cos(state.yaw),
        );
        camera.lookAt(t.x + state.panX, t.y + state.panY, t.z);
    }
    state.updateCamera = updateCamera3d;

    function resize() {
        const w = host.clientWidth, h = host.clientHeight;
        if (!w || !h) return;
        renderer.setSize(w, h);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
    }
    state.resize = resize;
    new ResizeObserver(resize).observe(host);
    resize();
    updateCamera3d();

    (function loop() {
        requestAnimationFrame(loop);
        if (host.hidden) return;
        renderer.render(scene, camera);
    }());

    OverlayState.scene3d = state;
    return state;
}

// 用 canvas 画一块带文字的气泡，做成 three.js 精灵，作为 3D 病灶标注。
// depthTest:false + 高 renderOrder，保证标签始终浮在器官表面之上、不被遮挡。
function makeLabelSprite(THREE, lines, opts = {}) {
    const pad = 18, lineH = 46, fontSize = 40;
    const font = `bold ${fontSize}px "Microsoft YaHei","PingFang SC",sans-serif`;
    const probe = document.createElement('canvas').getContext('2d');
    probe.font = font;
    let textW = 0;
    lines.forEach(l => { textW = Math.max(textW, probe.measureText(l).width); });
    const w = Math.ceil(textW + pad * 2 + 8);
    const h = Math.ceil(lines.length * lineH + pad * 2);

    const cv = document.createElement('canvas');
    cv.width = w; cv.height = h;
    const ctx = cv.getContext('2d');
    // 圆角背景
    const r = 16;
    ctx.beginPath();
    ctx.moveTo(r, 0); ctx.arcTo(w, 0, w, h, r); ctx.arcTo(w, h, 0, h, r);
    ctx.arcTo(0, h, 0, 0, r); ctx.arcTo(0, 0, w, 0, r); ctx.closePath();
    ctx.fillStyle = opts.bg || 'rgba(9,17,30,0.86)';
    ctx.fill();
    // 左侧强调色条
    if (opts.accent) { ctx.fillStyle = opts.accent; ctx.fillRect(0, 0, 7, h); }
    ctx.font = font; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
    lines.forEach((l, i) => {
        ctx.fillStyle = i === 0 ? '#ffffff' : (opts.subColor || '#ffcf4d');
        ctx.fillText(l, pad + 4, pad + lineH * i + lineH / 2);
    });

    const tex = new THREE.CanvasTexture(cv);
    tex.minFilter = THREE.LinearFilter;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
        map: tex, depthTest: false, transparent: true,
    }));
    const hh = opts.height || 34;   // 标签世界高度
    sprite.scale.set(hh * w / h, hh, 1);
    sprite.renderOrder = 999;
    return sprite;
}

async function buildScene3d() {
    const loading = document.getElementById('stage3dLoading');
    const errBox = document.getElementById('stage3dError');
    const src = OverlayState.segSource;
    // 已构建且来源未变则只同步可见性，避免重复拉网格
    if (OverlayState.scene3d && OverlayState.scene3d.built
        && OverlayState.scene3dSource === src) {
        syncScene3dVisibility();
        return;
    }
    errBox.hidden = true;
    loading.hidden = false;
    try {
        // 必须先加载 three.js 再初始化场景：initScene3d 依赖 window.THREE，
        // 旧代码在其之前调用会因 THREE 未定义抛错，导致 3D 永远卡在加载态。
        await loadThree();
        const state = OverlayState.scene3d || initScene3d();
        // 切换分割来源时清空旧网格，重新拉取对应来源的网格
        if (state.built) {
            state.meshes.forEach(mesh => {
                mesh.children.forEach(c => {
                    if (c.material) {
                        if (c.material.map) c.material.map.dispose();
                        c.material.dispose();
                    }
                });
                mesh.geometry.dispose();
                mesh.material.dispose();
            });
            state.meshes.clear();
            state.group.clear();
            state.built = false;
        }
        const resp = await fetch(
            `/api/overlay/${encodeURIComponent(OverlayState.fileId)}/mesh?mode=${encodeURIComponent(src)}`
        );
        if (!resp.ok) {
            const detail = await resp.json().catch(() => ({}));
            throw new Error(detail.message || `网格获取失败（HTTP ${resp.status}）`);
        }
        const data = await resp.json();
        if (!data.success) throw new Error(data.message || '网格获取失败');
        const THREE = state.THREE;
        const [sz, sy, sx] = data.spacing;
        let count = 0;
        data.meshes.forEach(m => {
            const v = m.verts, f = m.faces;
            const pos = new Float32Array(v.length);
            for (let i = 0; i < v.length; i += 3) {
                // 顶点列序 (z,y,x) → 渲染坐标 (x, -z, y)，z 间距 5mm 保持物理比例
                pos[i] = v[i + 2] * sx;
                pos[i + 1] = -v[i] * sz;
                pos[i + 2] = v[i + 1] * sy;
            }
            const geo = new THREE.BufferGeometry();
            geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
            geo.setIndex(f);
            geo.computeVertexNormals();
            geo.computeBoundingBox();

            // 病灶标注：受累器官按诊断置信度走热力色并加悬浮标签，
            // 其余器官降透明度作为解剖背景，突出模型发现的病灶。
            // id=0 为后端从 CT 灰度提取的体表轮廓，常驻极低透明度作解剖参照。
            const isBody = m.id === 0;
            const isLesion = !isBody && OverlayState.affectedIds.has(m.id);
            const conf = OverlayState.organConfidence.get(m.name);
            let color, opacity;
            if (isBody) {
                color = new THREE.Color(`rgb(${m.color.join(',')})`);
                opacity = 0.10;
            } else if (isLesion && conf != null) {
                const rgb = heatColor(conf);
                color = new THREE.Color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
                opacity = 0.97;
            } else {
                color = new THREE.Color(`rgb(${m.color.join(',')})`);
                opacity = 0.22;
            }
            const mesh = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({
                color, transparent: true, opacity, side: THREE.DoubleSide,
                shininess: isLesion ? 42 : 18,
                specular: new THREE.Color(isLesion ? 0x303030 : 0x141414),
                depthWrite: !isLesion && !isBody,  // 半透明层不写深度，避免遮挡标签
            }));
            mesh.name = m.name;
            mesh.userData.lesion = isLesion;

            if (isLesion) {
                const d = OverlayState.organFinding.get(m.name) || {};
                const rgb = heatColor(conf == null ? 0.5 : conf);
                const accent = `rgb(${rgb.map(x => Math.round(x)).join(',')})`;
                const label = makeLabelSprite(THREE, [
                    m.name,
                    `${d.finding || '异常发现'} ${conf != null ? pct(conf) : ''}`.trim(),
                ], { accent });
                // 定位到该器官包围盒顶部中心
                const bb = geo.boundingBox;
                label.position.set(
                    (bb.min.x + bb.max.x) / 2,
                    bb.max.y + 12,
                    (bb.min.z + bb.max.z) / 2,
                );
                mesh.add(label);   // 作为器官子节点：器官隐藏时标签一并隐藏
            }

            state.group.add(mesh);
            state.meshes.set(m.id, mesh);
            count += 1;
        });
        if (!count) throw new Error('没有可重建的器官网格');
        const box = new THREE.Box3().setFromObject(state.group);
        const center = new THREE.Vector3();
        box.getCenter(center);
        state.target.copy(center);
        state.dist = box.getSize(new THREE.Vector3()).length() * 1.35;
        state.updateCamera();
        state.built = true;
        OverlayState.scene3dSource = src;
        syncScene3dVisibility();
    } catch (e) {
        console.error('3D 重建失败:', e);
        errBox.textContent = (e && e.message) ? e.message : String(e);
        errBox.hidden = false;
    } finally {
        loading.hidden = true;
    }
}

function syncScene3dVisibility() {
    const state = OverlayState.scene3d;
    if (!state || !state.built) return;
    state.meshes.forEach((mesh, id) => {
        // id=0 体表轮廓常驻显示，不受器官图例勾选影响
        mesh.visible = id === 0 ? true : OverlayState.visible.has(id);
    });
}

async function setOverlayMode(mode) {
    if (OverlayState.mode === mode) return;
    OverlayState.mode = mode;
    document.querySelectorAll('#overlayModeToggle .mode-btn').forEach(b => {
        b.classList.toggle('is-active', b.dataset.mode === mode);
    });
    const stage2d = document.getElementById('overlayStage');
    const stage3d = document.getElementById('overlayStage3d');
    const hint = document.getElementById('overlayHint');
    if (mode === '3d') {
        stage2d.hidden = true;
        stage3d.hidden = false;
        hint.textContent = '拖拽旋转 · 滚轮缩放 · 右键平移';
        stopOverlayPlay();
        await buildScene3d();
    } else {
        stage3d.hidden = true;
        stage2d.hidden = false;
        hint.textContent = '滚轮/↑↓翻层 · 悬停查看器官 · 图例可跳转';
        renderOverlay();
    }
}

function bindOverlayEvents() {
    if (OverlayState.bound) return;
    OverlayState.bound = true;

    const sliceInput = document.getElementById('overlaySlice');
    sliceInput.addEventListener('input', () => {
        stopOverlayPlay();
        loadOverlaySlice(parseInt(sliceInput.value, 10));
    });

    const opacity = document.getElementById('overlayOpacity');
    opacity.addEventListener('input', () => {
        OverlayState.opacity = parseInt(opacity.value, 10) / 100;
        document.getElementById('overlayOpacityVal').textContent = opacity.value + '%';
        renderOverlay();
    });

    const segSource = document.getElementById('overlaySegSource');
    if (segSource) {
        segSource.addEventListener('change', async () => {
            OverlayState.segSource = segSource.value;
            OverlayState.cache.clear();
            OverlayState.cacheOrder = [];
            OverlayState.current = null;
            await loadOverlaySlice(OverlayState.slice);
            // 3D 已构建时按新来源重建网格
            if (OverlayState.mode === '3d') await buildScene3d();
        });
    }

    const mode = document.getElementById('overlayMode');
    mode.addEventListener('change', () => {
        OverlayState.renderMode = mode.value;
        const heatLegend = document.getElementById('overlayHeatLegend');
        if (heatLegend) heatLegend.hidden = mode.value !== 'heat';
        // 热力模式默认只看受累器官，突出模型关注区域
        if (mode.value === 'heat' && OverlayState.affectedIds.size > 0) {
            setOverlayOrgans('affected');
        } else {
            renderOverlay();
        }
    });

    const stage = document.getElementById('overlayStage');
    const canvas = document.getElementById('overlayCanvas');
    stage.addEventListener('wheel', (e) => {
        if (!OverlayState.meta) return;
        e.preventDefault();
        stopOverlayPlay();
        const dir = e.deltaY > 0 ? 1 : -1;
        const next = Math.min(OverlayState.meta.num_slices - 1, Math.max(0, OverlayState.slice + dir));
        if (next !== OverlayState.slice) {
            sliceInput.value = String(next);
            loadOverlaySlice(next);
        }
    }, { passive: false });

    canvas.addEventListener('mousemove', handleOverlayHover);
    canvas.addEventListener('mouseleave', () => {
        const tip = document.getElementById('overlayCursorTip');
        if (tip) tip.style.display = 'none';
    });

    // 键盘翻层：看片卡可见时 ↑/↓ 或 W/S 切换切片（焦点在输入控件上时不拦截）
    document.addEventListener('keydown', (e) => {
        const card = document.getElementById('overlayCard');
        if (!OverlayState.meta || !card || card.style.display === 'none') return;
        const tag = (document.activeElement || {}).tagName || '';
        if (['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(tag)) return;
        let dir = 0;
        if (e.key === 'ArrowUp' || e.key === 'w' || e.key === 'W') dir = -1;
        else if (e.key === 'ArrowDown' || e.key === 's' || e.key === 'S') dir = 1;
        if (!dir) return;
        e.preventDefault();
        stopOverlayPlay();
        const next = Math.min(OverlayState.meta.num_slices - 1, Math.max(0, OverlayState.slice + dir));
        if (next !== OverlayState.slice) {
            sliceInput.value = String(next);
            loadOverlaySlice(next);
        }
    });

    document.getElementById('overlayPlayBtn').addEventListener('click', toggleOverlayPlay);
    document.getElementById('overlayAllOrgansBtn').addEventListener('click', () => setOverlayOrgans('all'));
    document.getElementById('overlayAffectedBtn').addEventListener('click', () => setOverlayOrgans('affected'));
    document.querySelectorAll('#overlayModeToggle .mode-btn').forEach(btn => {
        btn.addEventListener('click', () => setOverlayMode(btn.dataset.mode));
    });
}

async function generateReport() {
    if (!AppState.currentDiagnosis) {
        showToast('请先进行AI诊断', 'warning');
        return;
    }
    
    showLoading('生成PDF报告...');
    
    try {
        const response = await fetch('/api/report/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                diagnosis_result: AppState.currentDiagnosis,
                file_id: AppState.currentFileId,
                patient_info: {
                    patient_id: (AppState.currentDiagnosis.metadata || {}).patient_id
                        || ('P' + new Date().getTime()),
                    study_date: new Date().toISOString().split('T')[0],
                    modality: 'CT'
                }
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // 下载PDF
            const link = document.createElement('a');
            link.href = 'data:application/pdf;base64,' + data.pdf_data;
            link.download = data.filename;
            link.click();
            
            showToast('PDF报告已生成', 'success');
        } else {
            showToast(data.message || '生成报告失败', 'error');
        }
    } catch (error) {
        console.error('Generate report error:', error);
        showToast('生成报告失败: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// ============ 历史记录 ============
async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        
        if (data.success) {
            displayHistory(data.records);
        }
    } catch (error) {
        console.error('Load history error:', error);
    }
}

function displayHistory(records) {
    const historyList = document.getElementById('historyList');
    historyList.innerHTML = '';
    
    if (records.length === 0) {
        historyList.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">暂无历史记录</p>';
        return;
    }
    
    records.forEach(record => {
        const item = document.createElement('div');
        item.className = 'history-item';
        item.innerHTML = `
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <strong>患者ID: ${record.patient_id}</strong>
                <span style="color: var(--text-secondary);">${record.timestamp}</span>
            </div>
            <div style="color: var(--text-secondary);">
                诊断结论: ${record.diagnosis.diagnosis || '无'}
            </div>
        `;
        historyList.appendChild(item);
    });
}

// ============ 批量处理 ============
document.getElementById('batchFileInput')?.addEventListener('change', async (e) => {
    const files = e.target.files;
    if (files.length === 0) return;
    
    showLoading('批量上传中...');
    
    const formData = new FormData();
    for (let file of files) {
        formData.append('files', file);
    }
    
    try {
        const response = await fetch('/api/batch/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayBatchResults(data.results);
            showToast(`批量上传完成: ${data.success_count}/${data.total}`, 'success');
        }
    } catch (error) {
        console.error('Batch upload error:', error);
        showToast('批量上传失败', 'error');
    } finally {
        hideLoading();
    }
});

function displayBatchResults(results) {
    const container = document.getElementById('batchResults');
    container.innerHTML = '<h4 style="margin: 1rem 0;">批量处理结果</h4>';
    
    results.forEach(result => {
        const item = document.createElement('div');
        item.className = 'history-item';
        item.style.borderLeftColor = result.success ? 'var(--success)' : 'var(--error)';
        item.innerHTML = `
            <div><strong>${result.filename}</strong></div>
            <div style="color: var(--text-secondary);">
                ${result.success ? 
                    `格式: ${result.file_type} | 尺寸: ${result.shape.join(' × ')}` : 
                    `错误: ${result.message}`
                }
            </div>
        `;
        container.appendChild(item);
    });
}

// ============ 工具函数 ============
function showLoading(text = '加载中...') {
    const overlay = document.getElementById('loadingOverlay');
    const loadingText = document.getElementById('loadingText');
    loadingText.textContent = text;
    overlay.style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}
