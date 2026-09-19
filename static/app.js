'use strict';

const $ = (id) => document.getElementById(id);
const state = {
  specs: null, file: null, results: [],
  colors: new Set(), sizes: new Set(), maxKb: 0, quality: 'high',
  originalUrl: null,     // source image, for the before/after comparison
};

/* ---------------------------------------------------------------- setup -- */

async function boot() {
  try {
    state.specs = await (await fetch('/api/specs')).json();
  } catch {
    showError('无法连接到服务，请确认电脑上的程序还在运行');
    return;
  }
  // pre-select only what the server marks as default (one size, one colour)
  state.specs.colors.filter((c) => c.default).forEach((c) => state.colors.add(c.key));
  state.specs.sizes.filter((s) => s.default).forEach((s) => state.sizes.add(s.key));
  state.maxKb = state.specs.default_max_kb || 0;
  state.quality = state.specs.default_quality || 'high';
  renderChips();
}

function renderChips() {
  $('colors').replaceChildren(...state.specs.colors.map((c) => {
    const el = chipEl(c.label, state.colors.has(c.key));
    el.insertBefore(Object.assign(document.createElement('i'), {
      className: 'dot',
      style: `background:rgb(${c.rgb.join(',')})`,
    }), el.firstChild);
    el.onclick = () => { toggle(state.colors, c.key); renderChips(); };
    return el;
  }));

  $('sizes').replaceChildren(...state.specs.sizes.map((s) => {
    const el = chipEl(`${s.label} ${s.w}×${s.h}`, state.sizes.has(s.key));
    el.onclick = () => { toggle(state.sizes, s.key); renderChips(); };
    return el;
  }));

  // single-choice: a photo has one size budget
  $('maxkb').replaceChildren(...(state.specs.max_kb || []).map((o) => {
    const el = chipEl(o.label, state.maxKb === o.key);
    el.onclick = () => { state.maxKb = o.key; renderChips(); };
    return el;
  }));

  $('quality').replaceChildren(...(state.specs.quality || []).map((o) => {
    const el = chipEl(`${o.label} ${o.secs}`, state.quality === o.key);
    el.onclick = () => { state.quality = o.key; renderChips(); };
    return el;
  }));
}

function chipEl(text, on) {
  const el = document.createElement('label');
  el.className = 'chip';
  el.dataset.on = on ? '1' : '0';
  el.appendChild(document.createTextNode(text));
  return el;
}

const toggle = (set, key) => (set.has(key) ? set.delete(key) : set.add(key));

/* ---------------------------------------------------------------- input -- */

$('file').addEventListener('change', (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  state.file = file;
  $('preview').src = URL.createObjectURL(file);
  $('step-pick').classList.add('hidden');
  $('step-options').classList.remove('hidden');
  $('step-results').classList.add('hidden');
  hideError();
});

$('reselect').onclick = () => { $('file').click(); };
$('again').onclick = () => {
  $('step-results').classList.add('hidden');
  $('step-options').classList.remove('hidden');
  hideError();
};

/* -------------------------------------------------------------- generate -- */

$('go').addEventListener('click', async () => {
  if (!state.file) return;
  if (!state.colors.size || !state.sizes.size) {
    return showError('请至少选择一种底色和一个规格');
  }

  const body = new FormData();
  body.append('photo', state.file);
  body.append('sizes', [...state.sizes].join(','));
  body.append('colors', [...state.colors].join(','));
  body.append('retouch', $('opt-retouch').checked ? 'true' : 'false');
  if (state.maxKb) body.append('max_kb', String(state.maxKb));
  body.append('quality', state.quality);

  hideError();
  $('busy').classList.remove('hidden');
  $('go').disabled = true;
  try {
    const res = await fetch('/api/process', { method: 'POST', body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `处理失败（${res.status}）`);
    state.results = data.results;
    renderResults(data);
    $('step-options').classList.add('hidden');
    $('step-results').classList.remove('hidden');
  } catch (err) {
    showError(err.message || '生成失败，请重试');
  } finally {
    $('busy').classList.add('hidden');
    $('go').disabled = false;
  }
});

function renderResults(payload) {
  state.originalUrl = payload.original_url || null;
  const first = payload.results[0];

  // before/after pair: the source next to the first result
  if (state.originalUrl && first) {
    $('original-img').src = state.originalUrl;
    $('compare-img').src = first.url;
    $('compare-cap').textContent = first.label;
    $('open-original').onclick = () => openViewer({
      url: state.originalUrl, label: '原图', filename: 'original.jpg',
    });
    $('open-result').onclick = () => openViewer(first);
  }

  // group: one block per size, one thumbnail per colour
  const bySize = new Map();
  for (const r of payload.results) {
    if (!bySize.has(r.size)) bySize.set(r.size, []);
    bySize.get(r.size).push(r);
  }

  const frag = document.createDocumentFragment();
  for (const [, items] of bySize) {
    const block = document.createElement('div');
    block.className = 'size-block';

    const h = document.createElement('h3');
    h.textContent = items[0].size_label;
    h.insertAdjacentHTML('beforeend', `<span>${items[0].w}×${items[0].h} · 300dpi</span>`);
    block.appendChild(h);

    const grid = document.createElement('div');
    grid.className = 'thumbs';
    for (const r of items) {
      const btn = document.createElement('button');
      btn.className = 'thumb';
      btn.type = 'button';
      btn.innerHTML = `<img src="${r.url}" alt="${r.label}" loading="lazy"><span>${r.color_label}</span>`;
      btn.onclick = () => openViewer(r);
      grid.appendChild(btn);
    }
    block.appendChild(grid);
    frag.appendChild(block);
  }
  $('results').replaceChildren(frag);

  if (payload.warnings && payload.warnings.length) {
    const p = document.createElement('p');
    p.className = 'tip';
    p.textContent = payload.warnings.join('；');
    $('results').appendChild(p);
  }
}

/* ---------------------------------------------------------------- viewer -- */

let current = null;
let showingOriginal = false;

// Flip the full-screen view between the result and the untouched source, so the
// comparison can be made at full size rather than on a thumbnail.
function paintViewer() {
  if (!current) return;
  const toggle = $('viewer-toggle');
  const canCompare = Boolean(state.originalUrl) && current.filename !== 'original.jpg';
  toggle.hidden = !canCompare;
  toggle.textContent = showingOriginal ? '看成品' : '看原图';
  $('viewer-img').src = showingOriginal ? state.originalUrl : current.url;
}

$('viewer-toggle').addEventListener('click', () => {
  showingOriginal = !showingOriginal;
  paintViewer();
});

function openViewer(item) {
  current = item;
  showingOriginal = false;
  paintViewer();
  $('viewer-label').textContent = item.label;
  const dl = $('viewer-download');
  dl.href = item.url;
  dl.download = item.filename;
  // Offer the share sheet only where it actually exists. Over plain http on a
  // LAN address this is not a secure context, so on iOS it is simply absent and
  // the long-press instruction below is the working path.
  if (navigator.canShare) {
    dl.textContent = '存储 / 下载';
  }
  $('viewer').classList.remove('hidden');
  document.body.style.overflow = 'hidden';
}

$('viewer-close').onclick = closeViewer;

$('viewer-download').addEventListener('click', async (e) => {
  if (!current || !navigator.canShare) return;   // let the plain download proceed
  e.preventDefault();
  try {
    const blob = await (await fetch(current.url)).blob();
    const file = new File([blob], current.filename, { type: 'image/jpeg' });
    if (navigator.canShare({ files: [file] })) {
      await navigator.share({ files: [file] });
      return;
    }
  } catch { /* fall through to the download link */ }
  window.location.href = current.url;
});

function closeViewer() {
  $('viewer').classList.add('hidden');
  $('viewer-img').removeAttribute('src');
  document.body.style.overflow = '';
  current = null;
}

/* ----------------------------------------------------------------- misc -- */

const showError = (msg) => {
  $('error').textContent = msg;
  $('error').classList.remove('hidden');
};
const hideError = () => $('error').classList.add('hidden');

boot();
