import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/esm/ort.min.js';

/* ---------- MODEL CONFIG ---------- */
// const MODEL_URL = '/static/model_750_per_label.onnx';
const MODEL_URL = '/static/model.onnx';
const SAMPLE_RATE = 16_000; // Hz expected by the model
const CLIP_LEN = SAMPLE_RATE; // 1‑second window
const WASM_PATH = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

/* ---------- SILENCE CONFIG ---------- */
const SILENCE_THRESH_DB = -25; // dBFS
const MIN_SILENCE_MS = 50; // contiguous silence span to detect trim points
const KEEP_SILENCE_MS = 10;  // keep 10 ms head/tail

const CHUNK_LEN = Math.round(SAMPLE_RATE / 1000 * MIN_SILENCE_MS); // 50 ms in samples
const SILENCE_THRESH_AMP = Math.pow(10, SILENCE_THRESH_DB / 20);
console.log('CHUNK_LEN', CHUNK_LEN);
// const MIN_SILENCE_SAMPLES = Math.round((MIN_SILENCE_MS / 1000) * SAMPLE_RATE);
// const KEEP_SAMPLES = Math.round((KEEP_SILENCE_MS / 1000) * SAMPLE_RATE);

/* ---- GLOBAL STATE ---- */
let session
let ctx
let ring = new Float32Array(CLIP_LEN)
let ringPos = 0
let listening = false

/* ---------- ORT setup ---------- */
ort.env.wasm.wasmPaths = WASM_PATH;
async function loadModel() {
    if (session) return session;
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
    return session;
}

function softmax(arr) {
    return arr.map(function (value, index) {
        return Math.exp(value) / arr.map(
            function (y /*value*/) { return Math.exp(y) }).reduce(
                function (a, b) { return a + b }
            )
    })
}

async function classify(frame, file_name) {
    if (!session) return;
    const input = new ort.Tensor('float32', frame, [1, CLIP_LEN]);
    const output = await session.run({ [session.inputNames[0]]: input });
    const logits_sfmax = softmax(output[session.outputNames[0]].data);

    // get top three logits
    const top3 = Array.from(logits_sfmax)
        .map((value, index) => ({ value, index }))
        .sort((a, b) => b.value - a.value);
    // .slice(0, 2);
    // keep only the indices
    top3.forEach((item, i) => top3[i] = item.index);
    console.log('Processing file:', file_name);
    console.log("Top pick: ", top3[0], logits_sfmax[top3[0]]);
    console.log('Second pick: ', top3[1], logits_sfmax[top3[1]]);

    document.getElementById('result').innerHTML =
        `${file_name}<br/>` +
        `Top pick: ${top3[0]} (${logits_sfmax[top3[0]].toFixed(2)})<br/>` +
        `Second pick: ${top3[1]} (${logits_sfmax[top3[1]].toFixed(2)})<br/>`;
}

function filterSubsequences(arr, thresh) {
    const tempResult = [];
    let temp = [];

    for (let i = 0; i <= arr.length; i++) {
        if (i < arr.length && arr[i] < thresh) {
            temp.push(arr[i]);
        } else {
            if (temp.length >= CHUNK_LEN && temp.every(val => val < thresh)) {
                // Skip silent chunk
            } else {
                tempResult.push(...temp);
            }
            if (i < arr.length) tempResult.push(arr[i]);
            temp = [];
        }
    }

    return new Float32Array(tempResult);
}

/* ---------- File‑upload flow ---------- */
async function processFile(file) {
    await loadModel();
    const arrayBuf = await file.arrayBuffer();
    // const arrayBuf = outputData.buffer;
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuf = await ctx.decodeAudioData(arrayBuf); // browser can decode WAV & AAC (.m4a)
    console.log({
        channels: audioBuf.numberOfChannels,
        length: audioBuf.length,
        sampleRate: audioBuf.sampleRate,
        duration: audioBuf.duration
    });

    var offlineCtx = new OfflineAudioContext(
        audioBuf.numberOfChannels,
        audioBuf.duration * SAMPLE_RATE,
        SAMPLE_RATE
    );

    var offlineSource = offlineCtx.createBufferSource();
    offlineSource.buffer = audioBuf;
    offlineSource.connect(offlineCtx.destination);
    offlineSource.start();
    offlineCtx.startRendering().then((resampled) => {
        let data;

        // `resampled` contains an AudioBuffer resampled at 16000Hz.
        // use resampled.getChannelData(x) to get an Float32Array for channel x.
        if (resampled.numberOfChannels > 2) {
            console.warn('Audio has more than 2 channels, using first two channels only');
        } else if (resampled.numberOfChannels === 1) {
            data = resampled.getChannelData(0);
        } else if (resampled.numberOfChannels === 2) {
            data = new Float32Array(resampled.length);
            const left = resampled.getChannelData(0);
            const right = resampled.getChannelData(1);
            for (let i = 0; i < resampled.length; i++) {
                data[i] = (left[i] + right[i]) / 2;
            }
        }

        console.log("data[0]", data[0]);
        console.log("SILENCE_THRESH_AMP", SILENCE_THRESH_AMP);
        console.log('Org Data shape:', data.length);

        data = filterSubsequences(data, SILENCE_THRESH_AMP);
        console.log('NoSilence Data shape:', data.length);

        if (data.length < CLIP_LEN) {
            const padded = new Float32Array(CLIP_LEN);
            padded.set(data);
            data = padded;
        } else if (data.length > CLIP_LEN) {
            data = data.slice(0, CLIP_LEN);
        }

        // console.log("maxmin", Math.max.apply(Math,data), Math.min.apply(Math,data));
        classify(data, file.name);
    });
}

/* ---------- UI wiring ---------- */
const fileInput = document.createElement('input');
fileInput.type = 'file';
fileInput.accept = '.wav,.m4a,audio/wav,audio/mp4,audio/x-m4a';
fileInput.style.display = 'none';
document.body.appendChild(fileInput);

document.getElementById('upload-btn').addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) processFile(file);
    e.target.value = '';
});


/* ---- Microphone flow ---- */
async function startListening() {
    if (listening) return
    listening = true
    document.getElementById('mic-btn').classList.add('active')

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    ctx = new AudioContext({
        sampleRate: SAMPLE_RATE, // resample to 16kHz
        // latencyHint: 'interactive'
    });

    // Worklet to tap raw PCM frames → main thread
    await ctx.audioWorklet.addModule(
        URL.createObjectURL(
            new Blob(
                [
                    `
                        class Tap extends AudioWorkletProcessor {
                            process(inputs) {
                            this.port.postMessage(inputs[0][0]); /* mono */
                            return true;
                            }
                        }
                        registerProcessor('tap', Tap);
                    `
                ],
                { type: 'application/javascript' }
            )
        )
    )

    const src = ctx.createMediaStreamSource(stream)
    const tap = new AudioWorkletNode(ctx, 'tap')
    src.connect(tap)

    // [ … 15 900 samples ]  +  128-sample chunk  =  ringPos hits 16 000
    //                        ^
    //              classify() fires here
    //                        |
    //            ringPos = 0 but  28 samples remain in
    //            the *same* chunk → you refill the buffer
    //            → you might hit 16 000 again immediately
    let leftover = 0;
    tap.port.onmessage = ({ data }) => {
        for (let i = 0; i < data.length; ++i) {
            if (ringPos === 0 && leftover) {   // skip the remainder
                leftover--;
                continue;
            }
            ring[ringPos++] = data[i]

            if (ringPos === CLIP_LEN) {
                data = filterSubsequences(ring, SILENCE_THRESH_AMP);
                if (data.length === 0) {
                    console.warn("No click deteced. Skipping...");
                    ringPos = 0;
                    document.getElementById('result').innerHTML = `No click deteced`;
                    return;
                }

                console.log("Ring length:", ring.length);
                console.log('NoSilence Data shape:', data.length);

                if (data.length < CLIP_LEN) {
                    const padded = new Float32Array(CLIP_LEN);
                    padded.set(data);
                    data = padded;
                } else if (data.length > CLIP_LEN) {
                    data = data.slice(0, CLIP_LEN);
                }

                classify(data.slice(), "streaming...")
                leftover = data.length - i - 1;          // samples still in this chunk
                ringPos = 0;
            }
        }
    }
}

/* ---- Entry point ---- */
document.getElementById('mic-btn').addEventListener('click', async () => {
    await loadModel()
    await startListening()
})