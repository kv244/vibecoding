const fxConfig = {
    gain: { name: 'Gain', params: [{ id: 'p1', label: 'Multiplier', min: 0, max: 5, step: 0.1, default: 1.0 }] },
    distortion: { name: 'Distortion', params: [{ id: 'p1', label: 'Drive', min: 1, max: 20, step: 0.5, default: 2.0 }] },
    lowpass: { name: 'Lowpass', params: [{ id: 'p1', label: 'Strength', min: 0, max: 1, step: 0.05, default: 0.5 }] },
    reverb: { name: 'Reverb (Alg)', params: [{ id: 'p1', label: 'Size', min: 0, max: 1, step: 0.05, default: 0.6 }, { id: 'p2', label: 'Mix', min: 0, max: 1, step: 0.05, default: 0.5 }] },
    echo: { name: 'Echo', params: [{ id: 'p1', label: 'Delay (smp)', min: 100, max: 20000, step: 100, default: 4410 }, { id: 'p2', label: 'Decay', min: 0, max: 1, step: 0.05, default: 0.5 }] },
    pitch: { name: 'Pitch Shift', params: [{ id: 'p1', label: 'Ratio', min: 0.5, max: 2.0, step: 0.05, default: 1.0 }] },
    flange: { name: 'Flanger', params: [{ id: 'p1', label: 'Depth', min: 0, max: 1, step: 0.05, default: 0.5 }, { id: 'p2', label: 'Feedback', min: 0, max: 1, step: 0.05, default: 0.7 }] },
    phase: { name: 'Phaser', params: [{ id: 'p1', label: 'Depth', min: 0, max: 1, step: 0.05, default: 0.5 }, { id: 'p2', label: 'Rate', min: 0, max: 1, step: 0.05, default: 0.2 }] },
    compress: { name: 'Compressor', params: [{ id: 'p1', label: 'Threshold', min: 0, max: 1, step: 0.01, default: 0.5 }, { id: 'p2', label: 'Ratio', min: 1, max: 20, step: 1, default: 4 }] },
    convolve: { name: 'Convolution', params: [{ id: 'ir', label: 'IR File', type: 'text', default: 'ir.wav' }] }
};

let effectChain = [];
let draggedIndex = null;

const chainContainer = document.getElementById('chain-container');
const fxSelect = document.getElementById('fxSelect');
const processBtn = document.getElementById('processBtn');
const masterMix = document.getElementById('masterMix');
const mixVal = document.getElementById('mixVal');
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
                <h4>${index + 1}. ${config.name}</h4>
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
            uploadHint.innerText = "Upload Complete!";
            showToast("File uploaded successfully");
        } else {
            showToast(`Upload failed: ${result.error}`, 5000);
            uploadHint.innerText = "Upload Failed. Try again.";
        }
    } catch (e) {
        showToast(`Server error during upload: ${e.message}`, 5000);
        uploadHint.innerText = "Server Error. Try again.";
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

masterMix.addEventListener('input', (e) => {
    mixVal.innerText = e.target.value;
});

processBtn.addEventListener('click', async () => {
    const outputName = document.getElementById('outputFile').value;

    if (!currentInputFile) {
        showToast("Please upload an input file first");
        return;
    }

    if (effectChain.length === 0) {
        showToast("Add at least one effect to the chain");
        return;
    }

    if (!outputName || outputName.trim() === "") {
        showToast("Please specify an output filename");
        return;
    }

    if (/[^a-zA-Z0-9._-]/.test(outputName)) {
        showToast("Output filename contains restricted characters");
        return;
    }

    if (!outputName.toLowerCase().endsWith('.wav')) {
        showToast("Output filename must end with .wav");
        return;
    }

    if (!outputName.toLowerCase().endsWith('.wav')) {
        showToast("Output filename must end with .wav");
        return;
    }

    showLoader(true);

    const payload = {
        inputFile: currentInputFile,
        outputName,
        mix: parseFloat(masterMix.value),
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
            showToast(`Success! Generated ${result.output}`);

            // Handle Audio Preview
            if (result.audio) {
                const audioContainer = document.getElementById('audio-container');
                const audioPlayer = document.getElementById('outputAudio');
                audioPlayer.src = `${result.audio}?t=${Date.now()}`;
                audioPlayer.load();
                audioContainer.classList.remove('hidden');
            }

            // Handle Waveform
            if (result.waveform) {
                const vizContainer = document.getElementById('viz-container');
                const vizImg = document.getElementById('waveformImg');
                vizImg.src = `${result.waveform}?t=${Date.now()}`;
                vizContainer.classList.remove('hidden');
            }
        } else {
            showToast(`Error: ${result.error}`, 5000);
        }
    } catch (e) {
        showToast(`Server unreachable: ${e.message}`, 5000);
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
