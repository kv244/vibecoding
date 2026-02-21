const fxConfig = {
    gain: { name: 'Gain', desc: "Linearly scales the signal amplitude.", params: [{ id: 'p1', label: 'Multiplier', min: 0, max: 5, step: 0.1, default: 1.0 }] },
    pan: { name: 'Pan', desc: "Adjusts left/right stereo balance.", params: [{ id: 'p1', label: 'L/R Balance', min: -1, max: 1, step: 0.05, default: 0.0 }] },
    eq: { name: 'EQ', desc: "Simple one-band parametric equalizer.", params: [{ id: 'p1', label: 'Center Freq', min: 0, max: 1, step: 0.05, default: 0.5 }, { id: 'p2', label: 'Gain', min: 0, max: 5, step: 0.1, default: 2.0 }] },
    lowpass: { name: 'Lowpass Filter', desc: "Attenuates high frequencies.", params: [{ id: 'p1', label: 'Strength', min: 0, max: 1, step: 0.05, default: 0.5 }] },
    distortion: { name: 'Distortion', desc: "Applies soft-clipping overdrive.", params: [{ id: 'p1', label: 'Drive', min: 1, max: 20, step: 0.5, default: 2.0 }] },
    bitcrush: { name: 'Bitcrusher', desc: "Reduces bit depth for lo-fi digital grit.", params: [{ id: 'p1', label: 'Bits', min: 1, max: 16, step: 1, default: 8 }] },
    compress: { name: 'Compressor', desc: "Reduces dynamic range of loud signals.", params: [{ id: 'p1', label: 'Threshold', min: 0, max: 1, step: 0.01, default: 0.5 }, { id: 'p2', label: 'Ratio', min: 1, max: 20, step: 1, default: 4 }] },
    gate: { name: 'Noise Gate', desc: "Silences signal below threshold.", params: [{ id: 'p1', label: 'Threshold', min: 0, max: 1, step: 0.01, default: 0.1 }, { id: 'p2', label: 'Reduction', min: 0, max: 1, step: 0.05, default: 0.0 }] },
    autowah: { name: 'Auto-Wah', desc: "A modulated sweeping envelope filter.", params: [] },
    chorus: { name: 'Chorus', desc: "Thickens sound with modulated delays.", params: [] },
    flange: { name: 'Flanger', desc: "Comb filtering with a sweeping LFO.", params: [{ id: 'p1', label: 'Depth', min: 0, max: 1, step: 0.05, default: 0.5 }, { id: 'p2', label: 'Feedback', min: 0, max: 1, step: 0.05, default: 0.7 }] },
    phase: { name: 'Phaser', desc: "Sweeping phase cancellation effect.", params: [{ id: 'p1', label: 'Depth', min: 0, max: 1, step: 0.05, default: 0.5 }, { id: 'p2', label: 'Rate', min: 0, max: 1, step: 0.05, default: 0.2 }] },
    tremolo: { name: 'Tremolo', desc: "Rhythmic volume/amplitude modulation.", params: [{ id: 'p1', label: 'Freq (Hz)', min: 0, max: 20, step: 0.5, default: 5.0 }, { id: 'p2', label: 'Depth', min: 0, max: 1, step: 0.05, default: 0.8 }] },
    widening: { name: 'Stereo Widening', desc: "Enhances the stereo field.", params: [{ id: 'p1', label: 'Width', min: 1, max: 3, step: 0.1, default: 1.5 }] },
    ringmod: { name: 'Ring Modulator', desc: "Multiplies signal with an oscillator.", params: [{ id: 'p1', label: 'Freq (Hz)', min: 20, max: 5000, step: 10, default: 440 }] },
    pitch: { name: 'Pitch Shift', desc: "Changes pitch without changing speed.", params: [{ id: 'p1', label: 'Ratio', min: 0.5, max: 2.0, step: 0.05, default: 1.0 }] },
    echo: { name: 'Echo / Delay', desc: "Repeats the signal over time.", params: [{ id: 'p1', label: 'Delay (smp)', min: 100, max: 40000, step: 100, default: 4410 }, { id: 'p2', label: 'Decay', min: 0, max: 1, step: 0.05, default: 0.5 }] },
    pingpong: { name: 'Ping-Pong Delay', desc: "Stereo delay bouncing left and right.", params: [{ id: 'p1', label: 'Delay (smp)', min: 100, max: 40000, step: 100, default: 8820 }, { id: 'p2', label: 'Decay', min: 0, max: 1, step: 0.05, default: 0.5 }] },
    reverb: { name: 'Reverb (Alg)', desc: "Simulates acoustic space via algorithms.", params: [{ id: 'p1', label: 'Size', min: 0, max: 1, step: 0.05, default: 0.6 }, { id: 'p2', label: 'Mix', min: 0, max: 1, step: 0.05, default: 0.5 }] },
    convolve: { name: 'Convolution Reverb', desc: "Applies an impulse response file (FFT based).", params: [{ id: 'ir', label: 'IR File', type: 'text', default: 'ir.wav' }] },
    freeze: { name: 'Spectral Freeze', desc: "Smears transients via random phases.", params: [] }
};

let effectChain = [];
let draggedIndex = null;

const chainContainer = document.getElementById('chain-container');
const fxSelect = document.getElementById('fxSelect');
const processBtn = document.getElementById('processBtn');
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('fileInput');
const selectedFileName = document.getElementById('selectedFileName');
const uploadHint = document.getElementById('upload-hint');

const osInfoSpan = document.getElementById('os-info');
const engineInfoSpan = document.getElementById('engine-info');
const engineVerSpan = document.getElementById('engine-ver');
const guiVerSpan = document.getElementById('gui-ver');

let currentInputFile = null;

// --- Effect Management ---

fxSelect.addEventListener('change', (e) => {
    const type = e.target.value;
    if (!type) return;

    addEffect(type);
    e.target.value = '';
    showToast(`Added ${fxConfig[type].name}`);
});

function addEffect(type) {
    const id = Date.now();
    const config = fxConfig[type];

    const fx = { id, type, params: {} };
    config.params.forEach(p => fx.params[p.id] = p.default);

    effectChain.push(fx);
    renderChain();
}

function removeEffect(id) {
    effectChain = effectChain.filter(fx => fx.id !== id);
    renderChain();
}

function updateParam(fxId, paramId, value) {
    const fx = effectChain.find(f => f.id === fxId);
    if (fx) {
        // Cast to float for numeric params, keep as string for 'text' params
        const config = fxConfig[fx.type];
        const paramConfig = config.params.find(p => p.id === paramId);
        fx.params[paramId] = paramConfig.type === 'text' ? value : parseFloat(value);

        const valSpan = document.getElementById(`val-${fxId}-${paramId}`);
        if (valSpan) valSpan.innerText = value;
    }
}

function renderChain() {
    if (effectChain.length === 0) {
        chainContainer.innerHTML = '<div class="empty-hint">Your chain is empty. Add an effect to start.</div>';
        return;
    }

    chainContainer.innerHTML = '';
    effectChain.forEach((fx, index) => {
        const config = fxConfig[fx.type];
        const module = document.createElement('div');
        module.className = 'fx-module';
        module.draggable = true;
        module.dataset.index = index;

        let paramsHtml = '';
        config.params.forEach(p => {
            if (p.type === 'text') {
                paramsHtml += `
                    <div class="param-item">
                        <label>${p.label}</label>
                        <input type="text" value="${fx.params[p.id]}" 
                               placeholder="e.g. ir.wav"
                               onchange="updateParam(${fx.id}, '${p.id}', this.value)">
                        <span class="param-hint">Must be a file on server</span>
                    </div>
                `;
            } else {
                paramsHtml += `
                    <div class="param-item">
                        <div class="slider-row">
                            <label>${p.label}</label>
                            <span id="val-${fx.id}-${p.id}">${fx.params[p.id]}</span>
                        </div>
                        <input type="range" min="${p.min}" max="${p.max}" step="${p.step}" value="${fx.params[p.id]}" 
                               oninput="updateParam(${fx.id}, '${p.id}', this.value)">
                    </div>
                `;
            }
        });

        module.innerHTML = `
            <div class="drag-handle">⋮⋮</div>
            <div class="fx-info">
                <h4>
                    ${index + 1}. ${config.name}
                    <span class="info-icon" title="${config.desc || ''}" onclick="alert('${config.desc || ''}')">ℹ️</span>
                </h4>
                ${config.desc ? `<div class="fx-desc">${config.desc}</div>` : ''}
                <div class="params">${paramsHtml}</div>
            </div>
            <button class="remove-btn" onclick="removeEffect(${fx.id})">✕</button>
        `;

        // Reordering Handlers (added after innerHTML to be clearer, though both work)
        module.addEventListener('dragstart', (e) => {
            draggedIndex = index;
            module.classList.add('dragging');
            e.dataTransfer.effectAllowed = 'move';
        });

        module.addEventListener('dragend', () => {
            module.classList.remove('dragging');
            draggedIndex = null;
        });

        module.addEventListener('dragover', (e) => {
            e.preventDefault();
            module.classList.add('drag-over');
        });

        module.addEventListener('dragleave', () => {
            module.classList.remove('drag-over');
        });

        module.addEventListener('drop', (e) => {
            e.preventDefault();
            module.classList.remove('drag-over');
            if (draggedIndex === null || draggedIndex === index) return;

            // Move item in array
            const item = effectChain.splice(draggedIndex, 1)[0];
            effectChain.splice(index, 0, item);
            renderChain();
        });

        chainContainer.appendChild(module);
    });
}

// --- Upload Logic ---

uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFileSelection(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFileSelection(e.target.files[0]);
    }
});

async function handleFileSelection(file) {
    if (!file.name.toLowerCase().endsWith('.wav')) {
        showToast("Only WAV files are supported", 5000);
        return;
    }

    selectedFileName.innerText = `Selected: ${file.name}`;
    uploadHint.innerText = "Uploading...";

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        if (result.success) {
            currentInputFile = result.filename;
            uploadHint.innerText = result.info || 'Upload Complete!';
            selectedFileName.innerText = `Selected: ${file.name}`;
            document.getElementById('convert-hint').classList.add('hidden');
            showToast('✅ File uploaded');
        } else {
            uploadHint.innerText = 'Upload Failed — see hint below';
            selectedFileName.innerText = '';
            // Show conversion hint for bit-depth or compressed format errors
            const isBitDepthError = result.error &&
                (result.error.includes('16-bit') || result.error.includes('compressed') || result.error.includes('non-PCM'));
            if (isBitDepthError) {
                const baseName = file.name.replace(/\.[^.]+$/, '');
                const cmd = `ffmpeg -i "${file.name}" -acodec pcm_s16le -ar 44100 "${baseName}_16bit.wav"`;
                document.getElementById('convert-hint-msg').innerText = result.error;
                document.getElementById('convert-cmd').innerText = cmd;
                document.getElementById('convert-hint').classList.remove('hidden');
            } else {
                showToast(`Upload failed: ${result.error}`, 5000);
            }
        }
    } catch (e) {
        showToast(`Server error during upload: ${e.message}`, 5000);
        uploadHint.innerText = 'Server Error. Try again.';
    }
}

// --- System Info ---

async function fetchSystemInfo() {
    try {
        const response = await fetch('/system-info');
        const data = await response.json();

        osInfoSpan.innerText = data.os || 'Unknown OS';
        guiVerSpan.innerText = data.gui_version || '?';
        engineVerSpan.innerText = data.engine_version || '?';

        if (Array.isArray(data.engine) && data.engine.length > 0) {
            // Find the first device if available
            const device = data.engine.find(line => line.includes('Device')) || data.engine[0];
            engineInfoSpan.innerText = device.replace('Device[0]:', 'GPU:').trim();
            engineInfoSpan.title = data.engine.join('\n'); // Show full info on hover
        } else if (typeof data.engine === 'string' && data.engine.startsWith('Probe failed')) {
            // Engine binary not found or not yet compiled - show graceful message
            engineInfoSpan.innerText = 'Engine: Not Compiled';
            engineInfoSpan.title = data.engine;
        } else {
            engineInfoSpan.innerText = 'Engine: Ready';
        }
    } catch (e) {
        osInfoSpan.innerText = 'Server Offline';
        engineInfoSpan.innerText = 'Cannot reach backend';
        engineVerSpan.innerText = '?';
        guiVerSpan.innerText = '?';
    }
}

// --- Interaction ---

processBtn.addEventListener('click', async () => {
    if (!currentInputFile) {
        showToast("Please upload an input file first");
        return;
    }

    if (effectChain.length === 0) {
        showToast("Add at least one effect to the chain");
        return;
    }

    // Auto-generate output filename
    const uuid = Math.random().toString(36).substring(2, 8);
    const outputName = `processed_${uuid}.wav`;

    showLoader(true);

    const payload = {
        inputFile: currentInputFile,
        outputName,
        mix: 1.0,  // Master mix removed from UI
        effects: effectChain.map(fx => ({
            type: fx.type,
            ...fx.params
        }))
    };

    try {
        const response = await fetch('/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        if (result.success) {
            showToast(`✅ Generated ${result.output}`);

            // Audio preview
            if (result.audio) {
                const audioContainer = document.getElementById('audio-container');
                const audioPlayer = document.getElementById('outputAudio');
                audioPlayer.src = `${result.audio}?t=${Date.now()}`;
                audioPlayer.load();
                audioContainer.classList.remove('hidden');
            }

            // Set up download button
            const downloadBtn = document.getElementById('downloadBtn');
            if (downloadBtn) {
                downloadBtn.onclick = () => {
                    window.location.href = '/download/' + result.output;
                };
            }

            // Canvas waveform visualization
            fetchAndDrawWaveform(result.output);

        } else {
            const errMsg = result.error || result.stderr || 'Unknown engine error. Is clfx.exe compiled?';
            showToast(`Error: ${errMsg}`, 6000);
        }
    } catch (e) {
        showToast(`Server unreachable: ${e.message || 'Check that server.py is running'}`, 5000);
    } finally {
        showLoader(false);
    }
});

function showToast(msg, duration = 3000) {
    const toast = document.getElementById('toast');
    toast.innerText = msg;
    toast.classList.remove('hidden');
    setTimeout(() => toast.classList.add('hidden'), duration);
}

function showLoader(show) {
    document.getElementById('loader').classList.toggle('hidden', !show);
}

// --- Waveform Canvas Renderer ---

async function fetchAndDrawWaveform(outputFilename) {
    try {
        const resp = await fetch('/visualize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file: outputFilename })
        });
        const data = await resp.json();
        if (data.error) { console.warn('Waveform error:', data.error); return; }

        const vizContainer = document.getElementById('viz-container');
        const canvas = document.getElementById('waveformCanvas');
        const meta = document.getElementById('viz-meta');
        meta.innerText = `${data.channels}ch · ${data.sample_rate}Hz · ${data.duration}s`;
        vizContainer.classList.remove('hidden');

        drawChannel(canvas, data.peaks_l, data.peaks_r, data.channels > 1);
    } catch (e) {
        console.warn('Visualization fetch failed:', e);
    }
}

function drawChannel(canvas, peaksL, peaksR, isStereo) {
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.offsetWidth || 900;
    const H = isStereo ? 200 : 120;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.height = `${H}px`;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    ctx.fillStyle = 'rgba(0,0,0,0.2)';
    ctx.fillRect(0, 0, W, H);

    const drawPeaks = (peaks, yOff, rowH, color) => {
        const mid = yOff + rowH / 2;
        const grad = ctx.createLinearGradient(0, yOff, 0, yOff + rowH);
        grad.addColorStop(0, color + 'aa');
        grad.addColorStop(0.5, color);
        grad.addColorStop(1, color + 'aa');
        ctx.fillStyle = grad;
        const bw = W / peaks.length;
        for (let i = 0; i < peaks.length; i++) {
            const h = peaks[i] * rowH * 0.48;
            ctx.fillRect(i * bw, mid - h, Math.max(1, bw - 0.5), h * 2);
        }
        // Centre line
        ctx.strokeStyle = 'rgba(255,255,255,0.08)';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(W, mid); ctx.stroke();
    };

    if (isStereo) {
        drawPeaks(peaksL, 0, H / 2, '#00f2ff');
        drawPeaks(peaksR, H / 2, H / 2, '#bc13fe');
    } else {
        drawPeaks(peaksL, 0, H, '#00f2ff');
    }
}

// --- Startup ---
fetchSystemInfo();
