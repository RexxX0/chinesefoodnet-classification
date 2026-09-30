// ChineseFoodNet frontend
// Talks to the FastAPI backend at /api/*
(() => {
  const API_BASE = '';  // same origin
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const preview = document.getElementById('preview');
  const predictBtn = document.getElementById('predict-btn');
  const resetBtn = document.getElementById('reset-btn');
  const topkInput = document.getElementById('topk');
  const resultsEl = document.getElementById('results');
  const placeholder = document.getElementById('placeholder');
  const elapsedEl = document.getElementById('elapsed');
  const statusDot = document.getElementById('status-dot');
  const statusText = document.getElementById('status-text');

  let currentFile = null;

  // -------- Status check --------
  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      statusDot.classList.add('ok');
      statusText.textContent = `${data.model_name} \u00b7 ${data.device}`;
      if (data.head_warning) {
        statusDot.classList.remove('ok');
        statusDot.classList.add('warn');
        statusText.title = data.head_warning;
        statusText.textContent += ' (warn)';
      }
    } catch (err) {
      statusDot.classList.add('err');
      statusText.textContent = 'API offline';
      statusText.title = String(err);
    }
  }

  // -------- File handling --------
  function setFile(file) {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      showError('Please upload an image file.');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showError('File is too large (max 10 MB).');
      return;
    }
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      preview.src = e.target.result;
      dropzone.classList.add('has-image');
      predictBtn.disabled = false;
      resetBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function reset() {
    currentFile = null;
    preview.src = '';
    fileInput.value = '';
    dropzone.classList.remove('has-image');
    predictBtn.disabled = true;
    resetBtn.disabled = true;
    resultsEl.innerHTML = '';
    placeholder.style.display = 'flex';
    elapsedEl.textContent = '';
  }

  function showError(msg) {
    placeholder.style.display = 'none';
    resultsEl.innerHTML = `<div class="error-msg">${escapeHtml(msg)}</div>`;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
  }

  // -------- Predict --------
  async function predict() {
    if (!currentFile) return;
    predictBtn.disabled = true;
    predictBtn.classList.add('loading');
    predictBtn.innerHTML = '<span class="spinner"></span>Identifying...';
    placeholder.style.display = 'none';
    resultsEl.innerHTML = '';

    const fd = new FormData();
    fd.append('file', currentFile);
    const topK = Math.max(1, Math.min(20, parseInt(topkInput.value, 10) || 5));

    try {
      const res = await fetch(`${API_BASE}/api/predict?top_k=${topK}`, {
        method: 'POST',
        body: fd,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      renderResults(data);
    } catch (err) {
      showError(`Prediction failed: ${err.message}`);
    } finally {
      predictBtn.disabled = false;
      predictBtn.classList.remove('loading');
      predictBtn.textContent = 'Identify Dish';
    }
  }

  function renderResults(data) {
    elapsedEl.textContent = `${data.elapsed_ms.toFixed(1)} ms`;
    if (!data.predictions || data.predictions.length === 0) {
      resultsEl.innerHTML = '<div class="error-msg">No predictions returned.</div>';
      return;
    }

    resultsEl.innerHTML = data.predictions.map((p, i) => {
      const pct = (p.confidence * 100).toFixed(2);
      const top1Cls = i === 0 ? 'top1' : '';
      // English primary (large), Chinese secondary (small).
      // Fall back to the Chinese label if no English translation is available.
      const enLabel = p.label_en || p.label_zh || '(unknown)';
      const zhLabel = (p.label_en && p.label_zh) ? p.label_zh : '';
      return `
        <div class="result-row ${top1Cls}">
          <div class="result-bar" style="transform: scaleX(${p.confidence});"></div>
          <div class="result-content">
            <div class="result-label">
              <span class="en">${escapeHtml(enLabel)}</span>
              ${zhLabel ? `<span class="zh">${escapeHtml(zhLabel)}</span>` : ''}
              <span class="id">id=${p.class_id}</span>
            </div>
            <div class="result-pct">${pct}%</div>
          </div>
        </div>
      `;
    }).join('');
  }

  // -------- Drag & drop --------
  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener('change', (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) setFile(f);
  });
  ['dragenter', 'dragover'].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });
  ['dragleave', 'drop'].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });
  dropzone.addEventListener('drop', (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) setFile(f);
  });

  // Paste from clipboard
  window.addEventListener('paste', (e) => {
    const items = e.clipboardData && e.clipboardData.items;
    if (!items) return;
    for (const item of items) {
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        setFile(item.getAsFile());
        break;
      }
    }
  });

  predictBtn.addEventListener('click', predict);
  resetBtn.addEventListener('click', reset);

  // Init
  checkHealth();
})();
