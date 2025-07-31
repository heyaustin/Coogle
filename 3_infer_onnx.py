import numpy as np
import onnxruntime as ort
import soundfile as sf
import torchaudio
import torch

MODEL_PATH = "model.onnx"   # path to your .onnx file
WAV_PATH = "6.wav"
SAMPLE_RATE = 16000  # Hz, or None to skip resampling
CLIP_LEN = 16000  # 1 second at 16 kHz


def load_audio(wav_path: str, desired_sr: int | None = None) -> np.ndarray:
    """Read a WAV, return mono float32 waveform (optionally resampled)."""
    waveform, sr = torchaudio.load(wav_path)
    # Resample if necessary
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
    # print("waveform shape:", waveform.shape, "sample rate:", sr)

    # Pad / truncate to exactly 1 second
    if waveform.shape[1] < CLIP_LEN:
        pad = CLIP_LEN - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :CLIP_LEN]

    return waveform


def prepare_input(wf: np.ndarray, input_shape) -> np.ndarray:
    """Add batch dim and pad/trim to the model’s expected sample length."""
    # wf = wf.astype(np.float32)[None, :]            # → shape (1, num_samples)
    if (input_shape is not None and len(input_shape) >= 2
            and isinstance(input_shape[1], int) and input_shape[1] > 0):
        target = input_shape[1]
        cur = wf.shape[1]
        if cur < target:
            wf = np.pad(wf, ((0, 0), (0, target - cur)), "constant")
        elif cur > target:
            wf = wf[:, :target]
    return wf


def run(model_path: str, wav_path: str, required_sr: int | None = None):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    meta = sess.get_inputs()[0]
    input_name = meta.name
    input_shape = meta.shape                       # e.g. [None, 16000]

    waveform = load_audio(wav_path, required_sr)
    x = prepare_input(waveform, input_shape)
    x = x.squeeze(0).cpu().numpy().astype("float32")  # shape (num_samples,)
    x = x[None, :]  # shape (1, num_samples)
    print(f"Input shape: {x.shape}, dtype: {x.dtype}")

    outputs = sess.run(None, {input_name: x})
    return outputs


if __name__ == "__main__":
    outs = run(MODEL_PATH, WAV_PATH, SAMPLE_RATE)
    print("Raw model outputs:")
    for i, out in enumerate(outs):
        print(f"  output[{i}] shape = {out.shape}")
        print(out)
        print(out.argmax())
