// 院内影像库 —— 浏览服务器已有数据、按序列粒度选用、零拷贝进入阅片诊断。
//
// 与上传流程的关键差异：库内导入不产生新的 file_id 目录，而是把 lib_<hash>
// 注册到原始 DICOM 序列路径。因此导入成功后走的是与上传完全相同的下游链路
// （OHIF + /api/diagnose + /api/image/slice），只是 file_id 的来源不同。

const LibState = {
    roots: [],
    tree: [],
    category: '',          // 当前选中的分类 key，如 '腹部影像/消化系统'
    filters: { keyword: '', modality: '', has_dicom: true, has_report: false, status: '' },
    page: 1,
    pageSize: 20,
    total: 0,
    items: [],
    currentCase: null,     // 详情抽屉里的病例
    seriesCache: {},       // examination_id -> series 数据
    imported: null,        // 最近一次导入的 {file_id, info, case_id}
};

// app.js 的导航回调据此决定「首次进入影像库」是否要拉索引。定义在本文件顶部，
// 因为 library.js 先于 app.js 加载。
let LibStateLoaded = false;

function initLibrary() {
    const on = (id, ev, fn) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener(ev, fn);
    };
    on('libRefreshBtn', 'click', () => refreshLibrary(true));
    on('libSearchBtn', 'click', () => { LibState.page = 1; loadLibraryCases(); });
    on('libResetBtn', 'click', resetLibraryFilters);
    on('libKeyword', 'keydown', (e) => { if (e.key === 'Enter') { LibState.page = 1; loadLibraryCases(); } });
    on('libScanBtn', 'click', openServerScanDialog);

    const drawer = document.getElementById('libDrawer');
    if (drawer) {
        drawer.addEventListener('click', (e) => {
            if (e.target.closest('[data-close]')) closeLibraryDrawer();
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !drawer.hidden) closeLibraryDrawer();
        });
    }

    // 诊断页「院内影像库」Tab 与入口按钮
    document.querySelectorAll('[data-upload-tab]').forEach((btn) => {
        btn.addEventListener('click', () => switchUploadTab(btn.dataset.uploadTab));
    });
    const entryBtn = document.getElementById('libraryEntryBtn');
    if (entryBtn) entryBtn.addEventListener('click', () => gotoPage('library'));

    // 统计数字：让诊断页的入口不至于显示 '-'
    loadLibraryStat();
}

function switchUploadTab(tab) {
    document.querySelectorAll('[data-upload-tab]').forEach((b) => {
        b.classList.toggle('active', b.dataset.uploadTab === tab);
    });
    const uploadArea = document.getElementById('uploadArea');
    const entry = document.getElementById('libraryEntry');
    const isLib = tab === 'library';
    if (uploadArea) uploadArea.style.display = isLib ? 'none' : '';
    if (entry) {
        entry.hidden = !isLib;
        if (isLib && !entry.dataset.loaded) { entry.dataset.loaded = '1'; loadLibraryStat(); }
    }
}

function gotoPage(name) {
    const nav = document.querySelector(`.nav-item[data-page="${name}"]`);
    if (nav) nav.click();
}

async function refreshLibrary(manual) {
    try {
        if (manual) {
            showLoading('重新读取影像库索引…');
            await fetch('/api/library/refresh', { method: 'POST' });
        }
        await loadLibraryStat();
        await loadLibraryTree();
        LibState.page = 1;
        await loadLibraryCases();
        if (manual) showToast('索引已重载', 'success');
    } catch (err) {
        console.error('[library] refresh failed', err);
        if (manual) showToast('重载索引失败: ' + err.message, 'error');
    } finally {
        if (manual) hideLoading();
    }
}

async function loadLibraryStat() {
    try {
        const res = await fetch('/api/library/status');
        const data = await res.json();
        if (!data.success) return;
        LibState.roots = data.roots || [];
        const ok = LibState.roots.filter((r) => !r.error);
        const cases = ok.reduce((s, r) => s + (r.cases || 0), 0);
        const exams = ok.reduce((s, r) => s + (r.examinations || 0), 0);
        const series = ok.reduce((s, r) => s + (r.series || 0), 0);
        const stat = ok.length
            ? `${ok.length} 个数据源 · ${cases} 病例 · ${exams} 检查 · ${series} 序列`
            : '未配置可用数据源';
        setLibStat(stat);
        const failed = LibState.roots.filter((r) => r.error);
        if (failed.length) {
            setLibStat(stat + ` · ${failed.length} 个目录读取失败`);
            document.getElementById('libTreeNote').textContent = failed[0].error;
        }
        const countEl = document.getElementById('libraryEntryCount');
        if (countEl) countEl.textContent = String(cases);
    } catch (err) {
        console.error('[library] status failed', err);
        setLibStat('影像库不可用');
    }
}

function setLibStat(text) {
    const el = document.getElementById('libStat');
    if (el) el.textContent = text;
}

async function loadLibraryTree() {
    try {
        const res = await fetch('/api/library/tree');
        const data = await res.json();
        if (!data.success) return;
        LibState.tree = (data.data || []).flatMap((r) => r.tree || []);
        renderLibraryTree();
    } catch (err) {
        console.error('[library] tree failed', err);
    }
}

function renderLibraryTree() {
    const box = document.getElementById('libTree');
    if (!box) return;
    box.innerHTML = '';
    if (!LibState.tree.length) {
        box.innerHTML = '<div class="lib-empty">分类树为空，请检查影像库配置</div>';
        return;
    }
    box.appendChild(buildTreeNode({ title: '全部分类', key: '', count: null, children: LibState.tree, level: 0 }));
}

function buildTreeNode(node) {
    const wrap = document.createElement('div');
    wrap.className = 'tree-node';

    const row = document.createElement('div');
    row.className = 'tree-row level-' + (node.level || 1);
    if (LibState.category === node.key) row.classList.add('selected');

    const kids = node.children || [];
    const caret = document.createElement('span');
    caret.className = 'tree-caret' + (kids.length ? '' : ' is-leaf');
    caret.textContent = kids.length ? '▸' : '·';
    let expanded = node.level === 0 || node.level === 1;
    if (expanded && kids.length) caret.classList.add('open');

    const label = document.createElement('span');
    label.className = 'tree-label';
    label.textContent = node.title;

    const count = document.createElement('span');
    count.className = 'tree-count';
    count.textContent = node.count === null
        ? String(kids.reduce((s, k) => s + (k.count || 0), 0))
        : String(node.count || 0);

    row.append(caret, label, count);
    wrap.appendChild(row);

    const childBox = document.createElement('div');
    childBox.className = 'tree-children';
    childBox.hidden = !expanded;
    kids.forEach((k) => childBox.appendChild(buildTreeNode(k)));
    wrap.appendChild(childBox);

    caret.addEventListener('click', (e) => {
        e.stopPropagation();
        if (!kids.length) return;
        expanded = !expanded;
        childBox.hidden = !expanded;
        caret.classList.toggle('open', expanded);
    });
    label.addEventListener('click', () => {
        LibState.category = node.key;
        LibState.page = 1;
        document.querySelectorAll('#libTree .tree-row.selected').forEach((r) => r.classList.remove('selected'));
        row.classList.add('selected');
        renderCrumb();
        loadLibraryCases();
    });
    row.addEventListener('click', () => label.click());
    return wrap;
}

function renderCrumb() {
    const el = document.getElementById('libCrumb');
    if (!el) return;
    const parts = LibState.category ? LibState.category.split('/') : [];
    el.innerHTML = '';
    const all = document.createElement('button');
    all.type = 'button';
    all.className = 'crumb-link';
    all.textContent = '全部分类';
    all.addEventListener('click', () => {
        LibState.category = '';
        LibState.page = 1;
        document.querySelectorAll('#libTree .tree-row.selected').forEach((r) => r.classList.remove('selected'));
        renderCrumb();
        loadLibraryCases();
    });
    el.appendChild(all);
    parts.forEach((p, i) => {
        const sep = document.createElement('span');
        sep.className = 'crumb-sep';
        sep.textContent = '/';
        el.appendChild(sep);
        const span = document.createElement('span');
        span.className = i === parts.length - 1 ? 'crumb-current' : 'crumb-text';
        span.textContent = p;
        el.appendChild(span);
    });
}

function resetLibraryFilters() {
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
    const chk = (id, v) => { const el = document.getElementById(id); if (el) el.checked = v; };
    set('libKeyword', ''); set('libModality', ''); set('libStatus', '');
    chk('libHasDicom', true); chk('libHasReport', false);
    LibState.filters = { keyword: '', modality: '', has_dicom: true, has_report: false, status: '' };
    LibState.page = 1;
    loadLibraryCases();
}

async function loadLibraryCases() {
    const body = document.getElementById('libTableBody');
    if (body) body.innerHTML = '<tr><td colspan="8" class="lib-empty">加载中…</td></tr>';
    LibState.filters = {
        keyword: (document.getElementById('libKeyword') || {}).value || '',
        modality: (document.getElementById('libModality') || {}).value || '',
        has_dicom: !!(document.getElementById('libHasDicom') || {}).checked,
        has_report: !!(document.getElementById('libHasReport') || {}).checked,
        status: (document.getElementById('libStatus') || {}).value || '',
    };

    const params = new URLSearchParams({ page: LibState.page, page_size: LibState.pageSize });
    if (LibState.category) params.set('category', LibState.category);
    Object.entries(LibState.filters).forEach(([k, v]) => {
        if (v === '' || v === false || v === null) return;
        params.set(k, v === true ? '1' : String(v));
    });

    try {
        const res = await fetch('/api/library/cases?' + params.toString());
        const data = await res.json();
        if (!data.success) {
            if (body) body.innerHTML = `<tr><td colspan="8" class="lib-empty">${escapeHtml(data.message || '查询失败')}</td></tr>`;
            return;
        }
        LibState.items = data.items || [];
        LibState.total = data.total || 0;
        renderLibraryTable();
        renderLibraryPager();
    } catch (err) {
        console.error('[library] cases failed', err);
        if (body) body.innerHTML = '<tr><td colspan="8" class="lib-empty">请求失败，请确认后端服务在运行</td></tr>';
    }
}

function renderLibraryTable() {
    const body = document.getElementById('libTableBody');
    if (!body) return;
    body.innerHTML = '';
    if (!LibState.items.length) {
        body.innerHTML = '<tr><td colspan="8" class="lib-empty">没有符合条件的病例</td></tr>';
        return;
    }
    LibState.items.forEach((c) => {
        const tr = document.createElement('tr');
        const cat = [c.level2, c.level3].filter(Boolean).join(' / ') || c.level1 || '-';
        tr.innerHTML = `
            <td class="mono">${escapeHtml(c.medical_record_no || '-')}</td>
            <td>${escapeHtml(c.patient_name || '-')}</td>
            <td class="muted">${escapeHtml((c.gender || '-') + ' / ' + (c.age || '-'))}</td>
            <td class="muted">${escapeHtml(cat)}</td>
            <td>${escapeHtml(c.disease_name || c.level3 || '-')}</td>
            <td>${c.has_dicom
                ? `<span class="tag tag-ok">CT×${c.exam_count || 0}</span> <span class="muted">${c.dicom_gb || 0} GB</span>`
                : '<span class="tag tag-off">无影像</span>'}</td>
            <td>${c.report_count
                ? `<span class="tag tag-ok">${c.report_count} 份</span>`
                : '<span class="tag tag-off">-</span>'}</td>
            <td class="col-action"></td>`;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-primary btn-sm';
        btn.textContent = c.has_dicom ? '选序列阅片' : '查看';
        btn.disabled = !c.has_dicom && !c.report_count;
        btn.addEventListener('click', () => openCaseDrawer(c.case_id));
        tr.lastElementChild.appendChild(btn);
        body.appendChild(tr);
    });
}

function renderLibraryPager() {
    const box = document.getElementById('libPager');
    if (!box) return;
    const pages = Math.max(1, Math.ceil(LibState.total / LibState.pageSize));
    box.innerHTML = '';
    const info = document.createElement('span');
    info.className = 'pager-info';
    info.textContent = `共 ${LibState.total} 例 · 第 ${LibState.page}/${pages} 页`;
    box.appendChild(info);

    const mk = (label, page, disabled) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'pager-btn';
        b.textContent = label;
        b.disabled = !!disabled;
        b.addEventListener('click', () => { LibState.page = page; loadLibraryCases(); });
        return b;
    };
    box.append(
        mk('«', 1, LibState.page <= 1),
        mk('‹', LibState.page - 1, LibState.page <= 1),
        mk('›', LibState.page + 1, LibState.page >= pages),
        mk('»', pages, LibState.page >= pages),
    );
}

// ==================== 病例详情抽屉 ====================

async function openCaseDrawer(caseId) {
    const drawer = document.getElementById('libDrawer');
    if (!drawer) return;
    drawer.hidden = false;
    document.body.classList.add('drawer-open');
    document.getElementById('libExamList').innerHTML = '<div class="lib-empty">加载中…</div>';
    document.getElementById('libReportList').innerHTML = '<div class="lib-empty">加载中…</div>';

    try {
        const res = await fetch('/api/library/case/' + encodeURIComponent(caseId));
        const data = await res.json();
        if (!data.success) throw new Error(data.message || '病例不存在');
        const c = data.data;
        LibState.currentCase = c;
        document.getElementById('libDrawerTitle').textContent =
            `${c.patient_name || '匿名'} · ${c.medical_record_no || c.case_id}`;
        document.getElementById('libDrawerSub').textContent = [
            c.gender, c.age, c.level2, c.level3, c.examination_part, c.examination_method,
            c.publication_status === 'published' ? '已发布' : '',
        ].filter(Boolean).join(' · ');
        // 检查列表与报告列表各自兜底：报告渲染出错不该让检查列表一起消失，
        // 否则用户看到的是「没有影像也没有报告」，而实际数据是好的。
        try {
            renderExamList(c.examinations || []);
        } catch (err) {
            console.error('[library] exam list failed', err);
            document.getElementById('libExamList').innerHTML =
                `<div class="lib-empty">检查列表渲染失败：${escapeHtml(err.message)}</div>`;
        }
        try {
            renderReportList(c.reports || []);
        } catch (err) {
            console.error('[library] report list failed', err);
            document.getElementById('libReportList').innerHTML =
                `<div class="lib-empty">报告渲染失败：${escapeHtml(err.message)}</div>`;
        }
    } catch (err) {
        console.error('[library] case detail failed', err);
        document.getElementById('libExamList').innerHTML =
            `<div class="lib-empty">${escapeHtml(err.message)}</div>`;
        document.getElementById('libReportList').innerHTML =
            `<div class="lib-empty">${escapeHtml(err.message)}</div>`;
    }
}

// 抽屉右栏的原始报告。与诊断页「原始报告对照」共用 renderReportItems，
// 保证两处对同一份报告的字段取舍与排版一致。
function renderReportList(reports) {
    const box = document.getElementById('libReportList');
    if (!box) return;
    box.innerHTML = '';
    if (!reports.length) {
        box.innerHTML = '<div class="lib-empty">该病例没有关联报告</div>';
        return;
    }
    const note = document.getElementById('libReportNote');
    if (note) note.textContent = `${reports.length} 份`;
    box.innerHTML = renderReportItems(reports);
}

function closeLibraryDrawer() {
    const drawer = document.getElementById('libDrawer');
    if (drawer) drawer.hidden = true;
    document.body.classList.remove('drawer-open');
}

function renderExamList(exams) {
    const box = document.getElementById('libExamList');
    box.innerHTML = '';
    if (!exams.length) {
        box.innerHTML = '<div class="lib-empty">该病例没有关联影像</div>';
        return;
    }
    exams.forEach((ex, i) => {
        const card = document.createElement('div');
        card.className = 'exam-card';
        card.innerHTML = `
            <div class="exam-head">
                <span class="tag tag-modality">${escapeHtml(ex.modality || '?')}</span>
                <strong>${escapeHtml(ex.study_description || '未命名检查')}</strong>
                <span class="muted">${escapeHtml(ex.examination_date || '')}</span>
            </div>
            <div class="exam-meta">
                检查号 ${escapeHtml(ex.accession_number || '-')} ·
                ${ex.series_count || 0} 个序列 · ${ex.image_count || 0} 张 ·
                ${ex.size_gb || 0} GB
                ${ex.body_part_examined ? ' · ' + escapeHtml(ex.body_part_examined) : ''}
            </div>
            <div class="series-box" data-exam="${escapeHtml(ex.examination_id)}"></div>`;
        box.appendChild(card);

        const seriesBox = card.querySelector('.series-box');
        // 首个有影像的检查自动展开序列（多数病例只有一个检查，省一次点击）
        if (i === 0 || exams.length === 1) {
            loadExamSeries(ex.examination_id, seriesBox);
        } else {
            seriesBox.innerHTML = '<button class="btn-text series-toggle" type="button">展开序列清单 ▾</button>';
            seriesBox.querySelector('.series-toggle').addEventListener('click', function once() {
                this.remove();
                loadExamSeries(ex.examination_id, seriesBox);
            });
        }
    });
}

async function loadExamSeries(examId, box) {
    box.innerHTML = '<div class="lib-empty">正在评估序列适用性…</div>';
    if (!LibState.seriesCache[examId]) {
        try {
            const res = await fetch(`/api/library/examination/${encodeURIComponent(examId)}/series`);
            const data = await res.json();
            if (!data.success) throw new Error(data.message || '读取失败');
            LibState.seriesCache[examId] = data.data;
        } catch (err) {
            box.innerHTML = `<div class="lib-empty">${escapeHtml(err.message)}</div>`;
            return;
        }
    }
    renderSeriesList(LibState.seriesCache[examId], box);
}

function renderSeriesList(data, box) {
    box.innerHTML = '';
    const list = data.series || [];
    if (!list.length) {
        box.innerHTML = '<div class="lib-empty">该检查没有序列记录</div>';
        return;
    }
    if (!data.study_dir_exists) {
        box.innerHTML += '<div class="lib-warn">服务器上的原始影像目录不可读，无法阅片</div>';
    }
    list.forEach((s) => {
        const row = document.createElement('div');
        row.className = 'series-row verdict-' + s.verdict;
        const badge = s.verdict === 'recommended' ? '★ 推荐'
            : s.verdict === 'usable' ? '可用' : '✗ 不适用';
        const desc = s.series_description || '(无序列名)';
        const thick = s.thickness ? `${Number(s.thickness).toFixed(1)}mm` : '层厚未知';
        row.innerHTML = `
            <span class="series-badge">${badge}</span>
            <span class="series-no">#${s.series_number ?? '-'}</span>
            <span class="series-desc" title="${escapeHtml(s.series_instance_uid || '')}">${escapeHtml(desc)}</span>
            <span class="series-spec muted">${s.instance_count || 0} 层 · ${thick}${s.phase ? ' · ' + escapeHtml(s.phase) : ''}</span>
            <span class="series-note">${escapeHtml([...(s.reasons || []), ...(s.warnings || [])].join('；'))}</span>`;

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-sm ' + (s.verdict === 'unsuitable' ? 'btn-secondary' : 'btn-primary');
        btn.textContent = '导入阅片';
        btn.disabled = !s.dir_exists || s.verdict === 'unsuitable';
        if (s.verdict === 'unsuitable') {
            btn.title = 'RADAR 只支持横断位 CT 断层序列：' + (s.reasons || []).join('；');
        }
        btn.addEventListener('click', () => importSeries(s, data));
        row.appendChild(btn);
        box.appendChild(row);
    });
}

async function importSeries(series, examData) {
    const caseId = (LibState.currentCase || {}).case_id;
    showLoading('正在从影像库载入序列（零拷贝）…');
    try {
        const res = await fetch('/api/library/import', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ series_id: series.series_id }),
        });
        const data = await res.json();
        if (!data.success) throw new Error(data.message || '导入失败');

        AppState.currentFileId = data.file_id;
        AppState.currentFileInfo = data.info;
        AppState.currentDiagnosis = null;
        LibState.imported = { file_id: data.file_id, info: data.info, case_id: caseId };

        displayFileInfo(data.info);
        showViewerSection();
        openOhifViewer(data.info.ohif_study_uid);
        setWorkflowStep('workflowViewerStep');
        const nameEl = document.getElementById('viewerStudyName');
        if (nameEl) {
            nameEl.textContent = `${data.info.filename} · ${data.info.shape.join(' × ')}`;
        }
        await loadOriginalReports(caseId, series, examData);

        closeLibraryDrawer();
        gotoPage('diagnosis');
        switchUploadTab('library');
        showToast(data.message || '已载入库内序列', 'success');
    } catch (err) {
        console.error('[library] import failed', err);
        showToast('导入失败: ' + err.message, 'error');
    } finally {
        hideLoading();
    }
}

// ==================== 原始报告对照 ====================

// 把一组报告渲染成 HTML。抽屉右栏与诊断页对照卡共用，避免两处各写一遍
// 字段取舍逻辑（金标准报告常只有 report_content，没有 findings/impression）。
function renderReportItems(reports) {
    const section = (title, text) => `
        <div class="report-section">
            <div class="report-section-title">${title}</div>
            <div class="report-section-body">${escapeHtml(text).replace(/\n/g, '<br>')}</div>
        </div>`;
    return reports.map((r) => {
        const parts = [
            r.clinical_history && section('临床病史', r.clinical_history),
            r.findings && section('影像所见', r.findings),
            r.impression && section('诊断意见', r.impression),
            // 结构化字段缺失时退回整段报告全文（金标准报告常是这种）
            (!r.findings && !r.impression && r.report_content) && section('报告全文', r.report_content),
        ].filter(Boolean);
        return `<div class="report-item">
            <div class="report-item-head">${escapeHtml(reportTypeLabel(r.report_type))}
                ${r.source_accession_number ? ' · ' + escapeHtml(r.source_accession_number) : ''}
                ${r.match_status && r.match_status !== 'matched' ? ` · <span class="muted">${escapeHtml(r.match_status)}</span>` : ''}
            </div>
            ${parts.length ? parts.join('') : '<div class="muted">（无正文）</div>'}
        </div>`;
    }).join('');
}

async function loadOriginalReports(caseId, series, examData) {
    const card = document.getElementById('originalReportCard');
    if (!card) return;
    if (!caseId) { card.style.display = 'none'; return; }
    try {
        const res = await fetch('/api/library/reports/' + encodeURIComponent(caseId));
        const data = await res.json();
        const reports = (data.success && data.data) || [];
        if (!reports.length) { card.style.display = 'none'; return; }
        card.style.display = '';
        const note = document.getElementById('originalReportNote');
        note.textContent = `${reports.length} 份 · ${reports.map((r) => reportTypeLabel(r.report_type)).join('、')}`;

        document.getElementById('originalReportBody').innerHTML = renderReportItems(reports);

        // AI 侧先占位，诊断完成后由 renderAiSummaryIntoCompare 填充
        document.getElementById('aiReportBody').innerHTML =
            '<div class="muted">尚未执行 AI 诊断。点击「开始AI诊断」后此处显示 RADAR 的影像学发现与印象。</div>';
    } catch (err) {
        console.error('[library] reports failed', err);
        card.style.display = 'none';
    }
}

function reportTypeLabel(t) {
    return ({ original: '原始报告', gold_standard: '金标准报告', teaching: '教学报告' })[t] || (t || '报告');
}

// 诊断结果出来后，把 AI 发现汇总到对照卡的右侧。由 app.js 在渲染结果时调用。
// result.diagnoses 每项形如 {organ, finding, confidence, severity, description}。
function renderAiSummaryIntoCompare(result, diagnoses) {
    const box = document.getElementById('aiReportBody');
    if (!box) return;
    const card = document.getElementById('originalReportCard');
    if (!card || card.style.display === 'none') return;
    const items = diagnoses || (result && result.diagnoses) || [];

    const lines = items.slice(0, 12).map((d) => `
        <div class="ai-finding">
            <div class="ai-finding-name">${escapeHtml(d.organ || '')} · ${escapeHtml(d.finding || '')}
                <span class="muted">${d.confidence != null ? ' ' + (d.confidence * 100).toFixed(0) + '%' : ''}</span>
            </div>
            ${d.description ? `<div class="ai-finding-desc">${escapeHtml(String(d.description))}</div>` : ''}
        </div>`).join('');

    const threshold = result && result.settings && result.settings.confidence_threshold;
    const summary = items.length
        ? `检出 ${items.length} 项阳性发现，累及 ${new Set(items.map((d) => d.organ)).size} 个器官。`
        : `未检出超过置信度阈值${threshold != null ? `（${(threshold * 100).toFixed(0)}%）` : ''}的异常。`;

    box.innerHTML = `
        <div class="report-section">
            <div class="report-section-title">总体印象</div>
            <div class="report-section-body">${escapeHtml(summary)}</div>
        </div>
        ${lines || '<div class="muted">无阳性项</div>'}`;
}

// ==================== 服务器目录导入（外部数据的另一条路） ====================

async function openServerScanDialog() {
    const path = prompt('输入服务器上的 DICOM 目录绝对路径\n（例如 /home/rimagai/data2/some-study）');
    if (!path) return;
    showLoading('正在扫描目录并评估序列…');
    try {
        const res = await fetch('/api/library/scan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path }),
        });
        const data = await res.json();
        if (!data.success) throw new Error(data.message || '扫描失败');
        const found = data.data;
        if (!found.series || !found.series.length) {
            showToast('该目录下没有找到 .dcm 序列', 'error');
            return;
        }
        showScanResults(found);
    } catch (err) {
        console.error('[library] scan failed', err);
        showToast('扫描失败: ' + err.message, 'error');
    } finally {
        hideLoading();
    }
}

function showScanResults(found) {
    const drawer = document.getElementById('libDrawer');
    drawer.hidden = false;
    document.body.classList.add('drawer-open');
    document.getElementById('libDrawerTitle').textContent = '服务器目录扫描结果';
    document.getElementById('libDrawerSub').textContent =
        `${found.root} · ${found.total_files} 个 DICOM 文件 · ${found.series_count} 个序列`;
    document.getElementById('libReportList').innerHTML =
        '<div class="lib-empty">目录导入不含院内报告</div>';
    const box = document.getElementById('libExamList');
    box.innerHTML = '';
    const card = document.createElement('div');
    card.className = 'exam-card';
    card.innerHTML = `<div class="exam-head"><strong>目录内序列</strong></div>
        <div class="exam-meta">按 RADAR 适用性排序，★ 为推荐主诊断序列</div>
        <div class="series-box"></div>`;
    box.appendChild(card);
    const seriesBox = card.querySelector('.series-box');
    renderScanSeries(found.series, seriesBox);
}

function renderScanSeries(list, box) {
    box.innerHTML = '';
    list.forEach((s) => {
        const row = document.createElement('div');
        row.className = 'series-row verdict-' + s.verdict;
        const badge = s.verdict === 'recommended' ? '★ 推荐'
            : s.verdict === 'usable' ? '可用' : '✗ 不适用';
        row.innerHTML = `
            <span class="series-badge">${badge}</span>
            <span class="series-no">#${s.series_number ?? '-'}</span>
            <span class="series-desc" title="${escapeHtml(s.series_dir || '')}">${escapeHtml(s.series_description || '(无序列名)')}</span>
            <span class="series-spec muted">${s.instance_count || 0} 层${s.thickness ? ' · ' + Number(s.thickness).toFixed(1) + 'mm' : ''}</span>
            <span class="series-note">${escapeHtml([...(s.reasons || []), ...(s.warnings || [])].join('；'))}</span>`;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-sm ' + (s.verdict === 'unsuitable' ? 'btn-secondary' : 'btn-primary');
        btn.textContent = '导入阅片';
        btn.disabled = s.verdict === 'unsuitable';
        btn.addEventListener('click', () => importScannedSeries(s));
        row.appendChild(btn);
        box.appendChild(row);
    });
}

async function importScannedSeries(s) {
    showLoading('正在载入服务器目录…');
    try {
        const res = await fetch('/api/library/import_path', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ series_dir: s.series_dir }),
        });
        const data = await res.json();
        if (!data.success) throw new Error(data.message || '导入失败');
        AppState.currentFileId = data.file_id;
        AppState.currentFileInfo = data.info;
        AppState.currentDiagnosis = null;
        displayFileInfo(data.info);
        showViewerSection();
        openOhifViewer(data.info.ohif_study_uid);
        setWorkflowStep('workflowViewerStep');
        const nameEl = document.getElementById('viewerStudyName');
        if (nameEl) nameEl.textContent = `${data.info.filename} · ${data.info.shape.join(' × ')}`;
        const card = document.getElementById('originalReportCard');
        if (card) card.style.display = 'none';
        closeLibraryDrawer();
        gotoPage('diagnosis');
        showToast(data.message || '已载入', 'success');
    } catch (err) {
        console.error('[library] import_path failed', err);
        showToast('导入失败: ' + err.message, 'error');
    } finally {
        hideLoading();
    }
}

// ==================== 工具 ====================

function escapeHtml(text) {
    return String(text ?? '').replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[ch]));
}
