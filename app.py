from pathlib import Path
import os
import json
import time
import re

from flask import Flask, jsonify, send_from_directory, request
from werkzeug.utils import secure_filename

import numpy as np
import essentia.standard as es

from openai import OpenAI

# ==========================
# 0. 基本配置
# ==========================

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
UPLOAD_DIR = AUDIO_DIR / "uploads"  # 用户上传的音频
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

VERSION_FILES = [
    ("original", "original.wav"),
    ("piano",    "piano.wav"),
    ("duet",     "duet.wav"),
]

app = Flask(__name__, static_folder="static")

# ==========================
# OpenAI / GPT AI 配置（不要把 key 写死在代码里）
# ==========================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

client = None
if OpenAI is not None and OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as exc:
        client = None
        print("[WARN] OpenAI client init failed, AI will run in mock mode:", exc)
else:
    print("[INFO] OPENAI_API_KEY not set (or openai not installed). AI will run in mock mode.")


# ==========================
# 1. 小工具函数
# ==========================

def safe_mean(values, default=0.0):
    if values is None:
        return float(default)
    if isinstance(values, (list, tuple)) and len(values) == 0:
        return float(default)
    try:
        return float(np.mean(values))
    except Exception:
        return float(default)


def clamp01(x):
    try:
        x = float(x)
    except Exception:
        return 0.5
    return float(max(0.0, min(1.0, x)))


def safe_float(x, default=0.0):
    """
    关键：避免 NaN/Inf 传到前端导致显示空白。
    """
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def clean_version_name(name: str) -> str:
    """
    只用于“显示/AI文本”，把:
      "111 (original)" -> "111"
      "222 (piano)"    -> "222"
    """
    s = str(name or "").strip()
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
    return s


def robust_sigmoid_normalize(values):
    """
    稳健归一化：
      - median 作为中心
      - MAD / std 作为尺度
      - sigmoid 映射到 [0,1]
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []

    median = np.median(arr)
    mad = np.median(np.abs(arr - median))

    if mad < 1e-6:
        std = np.std(arr)
        if std < 1e-6:
            return [0.5 for _ in arr]
        scale = std
    else:
        scale = 1.4826 * mad

    z = (arr - median) / (2.0 * scale)
    norm = 1.0 / (1.0 + np.exp(-z))
    norm = np.clip(norm, 0.02, 0.98)
    return norm.tolist()


# ==========================
# 2bis. 结构分段（粗颗粒）
# ==========================

def segment_audio_by_structure(audio, sr):
    duration = len(audio) / sr

    frame_size = 2048
    hop_size = 1024

    w = es.Windowing(type="hann")
    spectrum = es.Spectrum()
    mfcc = es.MFCC()

    mfcc_frames = []
    for frame in es.FrameGenerator(
        audio,
        frameSize=frame_size,
        hopSize=hop_size,
        startFromZero=True,
        validFrameThresholdRatio=0.5,
    ):
        frame = np.array(frame)
        mag_spectrum = spectrum(w(frame))
        _, mfcc_coeffs = mfcc(mag_spectrum)
        mfcc_frames.append(np.array(mfcc_coeffs))

    if not mfcc_frames:
        print("[WARN] No MFCC frames, fall back to single segment.")
        return [(0.0, duration)]

    mfcc_mat = np.vstack(mfcc_frames)
    diff = np.diff(mfcc_mat, axis=0)
    novelty = np.linalg.norm(diff, axis=1)

    if novelty.size == 0:
        print("[WARN] Novelty empty, fall back to single segment.")
        return [(0.0, duration)]

    novelty = novelty - novelty.min()
    if novelty.max() > 0:
        novelty = novelty / novelty.max()

    win = 16
    if novelty.size > win:
        kernel = np.ones(win) / win
        novelty_smooth = np.convolve(novelty, kernel, mode="same")
    else:
        novelty_smooth = novelty

    target_avg_len = 50.0
    min_segments = 3
    max_segments = 6

    est_segments = int(round(duration / target_avg_len))
    target_segments = max(min_segments, min(max_segments, est_segments))
    max_internal_boundaries = max(0, target_segments - 1)

    min_seg_duration_sec = 18.0
    min_frames_between = int(min_seg_duration_sec * sr / hop_size)

    peak_indices = []
    for i in range(1, len(novelty_smooth) - 1):
        if novelty_smooth[i] > novelty_smooth[i - 1] and novelty_smooth[i] >= novelty_smooth[i + 1]:
            peak_indices.append(i)

    if not peak_indices:
        print("[INFO] No peaks, single segment.")
        return [(0.0, duration)]

    peak_indices_sorted = sorted(
        peak_indices,
        key=lambda idx: novelty_smooth[idx],
        reverse=True,
    )

    selected = []
    for idx in peak_indices_sorted:
        if len(selected) >= max_internal_boundaries:
            break
        if not selected:
            selected.append(idx)
        else:
            if all(abs(idx - s) >= min_frames_between for s in selected):
                selected.append(idx)

    boundaries = [0.0]
    for idx in sorted(selected):
        t = (idx * hop_size) / sr
        if 0.0 < t < duration:
            boundaries.append(float(t))
    boundaries.append(duration)
    boundaries = sorted(set(boundaries))

    segments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end <= start:
            continue
        seg_len = end - start
        if segments and seg_len < min_seg_duration_sec * 0.75:
            prev_start, _prev_end = segments[-1]
            segments[-1] = (prev_start, end)
        else:
            segments.append((start, end))

    if not segments:
        print("[WARN] No valid segments, single segment.")
        return [(0.0, duration)]

    print(f"[INFO] Structure-based segmentation: got {len(segments)} segments.")
    return segments


# ==========================
# 3. 计算单段特征（局部统计）
# ==========================

def analyze_segment_features(seg_audio, sr, tempo_value):
    frame_size = 2048
    hop_size = 1024

    energy_alg     = es.Energy()
    loudness_alg   = es.Loudness()
    centroid_alg   = es.SpectralCentroidTime()
    complexity_alg = es.SpectralComplexity()

    window_alg   = es.Windowing(type="hann")
    spectrum_alg = es.Spectrum()
    mfcc_alg     = es.MFCC()

    key_extractor = es.KeyExtractor()

    energy_vals     = []
    loudness_vals   = []
    centroid_vals   = []
    complexity_vals = []
    mfcc_frames     = []

    for frame in es.FrameGenerator(
        seg_audio,
        frameSize=frame_size,
        hopSize=hop_size,
        startFromZero=True,
        validFrameThresholdRatio=0.5,
    ):
        frame = np.array(frame)

        e = float(energy_alg(frame))
        energy_vals.append(e)

        loudness_vals.append(float(loudness_alg(frame)))

        c = float(centroid_alg(frame))
        centroid_vals.append(c)

        comp = float(complexity_alg(frame))
        complexity_vals.append(comp)

        mag_spectrum = spectrum_alg(window_alg(frame))
        _, mfcc_coeffs = mfcc_alg(mag_spectrum)
        mfcc_frames.append(np.array(mfcc_coeffs))

    energy_avg     = safe_mean(energy_vals, default=1e-9)
    loudness_avg   = safe_mean(loudness_vals, default=-60.0)
    brightness_avg = safe_mean(centroid_vals, default=2500.0)
    roughness_avg  = safe_mean(complexity_vals, default=0.0)

    if mfcc_frames:
        mfcc_mat = np.vstack(mfcc_frames)
        mfcc_mean = np.mean(mfcc_mat, axis=0)
    else:
        mfcc_mean = np.zeros(13, dtype=float)

    dyn_alg = es.DynamicComplexity(frameSize=frame_size, sampleRate=sr)
    try:
        dyn_complexity, loudness_spl = dyn_alg(seg_audio)
    except Exception:
        dyn_complexity, loudness_spl = 0.0, 0.0

    energy_db = 10.0 * np.log10(energy_avg + 1e-12)

    try:
        key, scale, key_strength = key_extractor(seg_audio)
        key_str = f"{key} {scale}"
    except Exception:
        key_str = "Unknown"

    return {
        "tempo":              safe_float(tempo_value, 80.0),
        "energy_raw":         safe_float(energy_db, -60.0),
        "roughness_raw":      safe_float(roughness_avg, 0.0),
        "brightness":         safe_float(brightness_avg, 2500.0),
        "loudness":           safe_float(loudness_avg, -60.0),
        "dynamic_complexity": safe_float(dyn_complexity, 0.0),
        "loudness_spl":       safe_float(loudness_spl, 0.0),
        "mfcc":               [safe_float(x, 0.0) for x in mfcc_mean[:5]],
        "key":                key_str,
        "energy":             0.0,
        "roughness":          0.0,
        "valence":            0.5,
    }


# ==========================
# 4. 全局归一化 energy / roughness / valence
# ==========================

def normalize_energies_and_roughness(data):
    if not data or "versions" not in data:
        return data

    loudness_vals = []
    roughness_vals = []
    brightness_vals = []

    for v in data["versions"]:
        for seg in v.get("segments", []):
            loudness_vals.append(safe_float(seg.get("loudness", -60.0), -60.0))
            roughness_vals.append(safe_float(seg.get("roughness_raw", 0.0), 0.0))
            brightness_vals.append(safe_float(seg.get("brightness", 2500.0), 2500.0))

    if not loudness_vals:
        print("[WARN] No loudness values for normalization.")
        return data

    energy_norm_all = robust_sigmoid_normalize(loudness_vals)
    roughness_norm_all = robust_sigmoid_normalize(roughness_vals) if roughness_vals else [0.5 for _ in loudness_vals]

    b_arr = np.asarray(brightness_vals, dtype=float)
    b_min = float(np.min(b_arr)) if b_arr.size else 0.0
    b_max = float(np.max(b_arr)) if b_arr.size else 1.0
    if (b_max - b_min) < 1e-6:
        brightness_norm_all = [0.5 for _ in loudness_vals]
    else:
        brightness_norm_all = ((b_arr - b_min) / (b_max - b_min)).tolist()
        brightness_norm_all = np.clip(brightness_norm_all, 0.02, 0.98).tolist()

    idx = 0
    for v in data["versions"]:
        for seg in v.get("segments", []):
            e_norm = energy_norm_all[idx]
            r_norm = roughness_norm_all[idx]
            b_norm = brightness_norm_all[idx]

            seg["energy"] = clamp01(e_norm)
            seg["roughness"] = clamp01(r_norm)
            seg["valence"] = clamp01(0.5 * e_norm + 0.5 * b_norm)

            seg["brightness"] = safe_float(seg.get("brightness", 2500.0), 2500.0)
            seg["loudness"] = safe_float(seg.get("loudness", -60.0), -60.0)

            idx += 1

    print("[INFO] Global normalization done.")
    return data


# ==========================
# 5. 分析一首歌（自动结构分段）
# ==========================

def analyse_one_file(path: Path, slot: str, display_name: str):
    if not path.exists():
        print(f"[WARN] Audio file not found: {path}")
        return None

    print(f"[INFO] Analysing file (auto) slot='{slot}', label='{display_name}'")

    sr = 44100
    loader = es.MonoLoader(filename=str(path), sampleRate=sr)
    audio = loader()

    duration = len(audio) / sr
    print(f"  - duration: {duration:.1f} s, sr={sr}")

    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, _ = rhythm_extractor(audio)
    print(f"  - BPM ~ {bpm:.1f}, beats: {len(beats)}")

    segments_bounds = segment_audio_by_structure(audio, sr)

    segments_json = []
    for idx, (start, end) in enumerate(segments_bounds, start=1):
        start_sample = int(start * sr)
        end_sample   = int(end * sr)
        seg_audio    = audio[start_sample:end_sample]

        features = analyze_segment_features(seg_audio, sr, tempo_value=bpm)
        segment_name = f"section{idx}"

        segment_obj = {
            "name":               segment_name,
            "start":              safe_float(start, 0.0),
            "end":                safe_float(end, safe_float(duration, 0.0)),
            "tempo":              safe_float(features.get("tempo", 80.0), 80.0),
            "energy":             safe_float(features.get("energy", 0.0), 0.0),
            "valence":            safe_float(features.get("valence", 0.5), 0.5),
            "brightness":         safe_float(features.get("brightness", 2500.0), 2500.0),
            "roughness":          safe_float(features.get("roughness", 0.0), 0.0),
            "dynamic_complexity": safe_float(features.get("dynamic_complexity", 0.0), 0.0),
            "loudness":           safe_float(features.get("loudness", -60.0), -60.0),
            "loudness_spl":       safe_float(features.get("loudness_spl", 0.0), 0.0),
            "energy_raw":         safe_float(features.get("energy_raw", -60.0), -60.0),
            "roughness_raw":      safe_float(features.get("roughness_raw", 0.0), 0.0),
            "mfcc":               features.get("mfcc", [0, 0, 0, 0, 0]),
            "key":                features.get("key", "Unknown"),
        }
        segments_json.append(segment_obj)

    return {
        "slot":     slot,
        "name":     display_name,
        "segments": segments_json,
    }


# ==========================
# 6. 自定义分段版本的分析
# ==========================

def analyse_one_file_with_custom_segments(path: Path, slot: str, display_name: str, seg_defs):
    if not path.exists():
        print(f"[WARN] Audio file not found: {path}")
        return None

    print(f"[INFO] Analysing file (custom) slot='{slot}', label='{display_name}'")

    sr = 44100
    loader = es.MonoLoader(filename=str(path), sampleRate=sr)
    audio = loader()

    duration = len(audio) / sr
    print(f"  - duration: {duration:.1f} s, sr={sr}")

    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, _ = rhythm_extractor(audio)
    print(f"  - BPM ~ {bpm:.1f}, beats: {len(beats)}")

    segments_json = []
    for seg in seg_defs:
        seg_name = str(seg.get("name", "segment"))
        start = safe_float(seg.get("start", 0.0), 0.0)
        end   = safe_float(seg.get("end", duration), duration)

        start = max(0.0, min(start, duration))
        end   = max(start, min(end, duration))
        if end <= start:
            continue

        start_sample = int(start * sr)
        end_sample   = int(end * sr)
        seg_audio    = audio[start_sample:end_sample]

        features = analyze_segment_features(seg_audio, sr, tempo_value=bpm)

        segment_obj = {
            "name":               seg_name,
            "start":              safe_float(start, 0.0),
            "end":                safe_float(end, duration),
            "tempo":              safe_float(features.get("tempo", 80.0), 80.0),
            "energy":             safe_float(features.get("energy", 0.0), 0.0),
            "valence":            safe_float(features.get("valence", 0.5), 0.5),
            "brightness":         safe_float(features.get("brightness", 2500.0), 2500.0),
            "roughness":          safe_float(features.get("roughness", 0.0), 0.0),
            "dynamic_complexity": safe_float(features.get("dynamic_complexity", 0.0), 0.0),
            "loudness":           safe_float(features.get("loudness", -60.0), -60.0),
            "loudness_spl":       safe_float(features.get("loudness_spl", 0.0), 0.0),
            "energy_raw":         safe_float(features.get("energy_raw", -60.0), -60.0),
            "roughness_raw":      safe_float(features.get("roughness_raw", 0.0), 0.0),
            "mfcc":               features.get("mfcc", [0, 0, 0, 0, 0]),
            "key":                features.get("key", "Unknown"),
        }
        segments_json.append(segment_obj)

    return {
        "slot":     slot,
        "name":     display_name,
        "segments": segments_json,
    }


# ==========================
# 7. 启动时分析默认三版本（demo）
# ==========================

def analyse_all_versions(audio_dir: Path):
    versions_data = []
    for slot, filename in VERSION_FILES:
        path = audio_dir / filename
        vdata = analyse_one_file(path, slot, display_name=slot)
        if vdata is not None:
            versions_data.append(vdata)

    result = {"versions": versions_data}
    result = normalize_energies_and_roughness(result)
    print("[INFO] Finished analysing demo versions.")
    return result


print("[INFO] Starting Essentia analysis (demo audio)...")
ANALYSED_DATA = analyse_all_versions(AUDIO_DIR)
print("[INFO] ANALYSED_DATA ready.")


# ==========================
# 8. Flask 路由
# ==========================

@app.route("/")
def home():
    return "Backend running. Go to /view."


@app.route("/view")
def view():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/demo")
def api_demo():
    return jsonify(ANALYSED_DATA)


@app.route("/api/custom_segments", methods=["POST"])
def api_custom_segments():
    """
    ✅ 重要修复：自定义分段只分析“真实存在的 uploaded 音频”
    - 不再回退到 demo 音频
    - 用户只上传 2 个，就只会分析 2 个
    """
    payload = request.get_json(force=True, silent=True)
    if not payload or "versions" not in payload:
        return jsonify({"error": "Invalid payload, expected 'versions' field."}), 400

    versions_result = []

    for v in payload["versions"]:
        slot = v.get("slot") or v.get("name")
        seg_defs = v.get("segments", [])
        label = v.get("label", slot)

        if not slot or not seg_defs:
            continue

        uploaded_path = UPLOAD_DIR / f"uploaded_{slot}.wav"

        # ✅ 不回退 demo：没上传就跳过，避免“凭空多出第三个版本”
        if not uploaded_path.exists():
            print(f"[WARN] custom_segments: uploaded audio missing for slot={slot}, skip.")
            continue

        vdata = analyse_one_file_with_custom_segments(uploaded_path, slot, label, seg_defs)
        if vdata is not None:
            versions_result.append(vdata)

    if not versions_result:
        return jsonify({"error": "No valid uploaded versions in custom segmentation."}), 400

    data = {"versions": versions_result}
    data = normalize_energies_and_roughness(data)
    return jsonify(data)


@app.route("/api/upload_analyse", methods=["POST"])
def api_upload_analyse():
    has_any = False
    for slot in ["original", "piano", "duet"]:
        f = request.files.get(slot)
        if f and f.filename:
            has_any = True
            break

    if not has_any:
        return jsonify({"error": "Please upload at least one audio file."}), 400

    versions_data = []

    for slot in ["original", "piano", "duet"]:
        f = request.files.get(slot)
        if not f or not f.filename:
            continue

        label = request.form.get(f"label_{slot}", slot)

        filename = secure_filename(f"uploaded_{slot}.wav")
        save_path = UPLOAD_DIR / filename
        f.save(save_path)
        print(f"[INFO] Saved uploaded '{slot}' as {save_path}")

        vdata = analyse_one_file(save_path, slot, display_name=label)
        if vdata is not None:
            versions_data.append(vdata)

    if not versions_data:
        return jsonify({"error": "Failed to analyse uploaded audio files."}), 500

    data = {"versions": versions_data}
    data = normalize_energies_and_roughness(data)
    return jsonify(data)


@app.route("/audio/<slot>")
def serve_audio(slot):
    """
    播放用：
      /audio/original
      /audio/piano
      /audio/duet

    ✅ 支持 query 参数控制来源
      - ?source=demo     强制播 demo
      - ?source=uploaded 强制播 uploaded（存在才会播）
      - 不带参数：默认优先 uploaded
    """
    source = (request.args.get("source") or "").strip().lower()

    uploaded_path = UPLOAD_DIR / f"uploaded_{slot}.wav"

    if source == "demo":
        for s, fname in VERSION_FILES:
            if s == slot:
                default_path = AUDIO_DIR / fname
                if default_path.exists():
                    return send_from_directory(AUDIO_DIR, fname)
        return jsonify({"error": f"Demo audio for slot '{slot}' not found."}), 404

    if source == "uploaded":
        if uploaded_path.exists():
            return send_from_directory(UPLOAD_DIR, uploaded_path.name)
        return jsonify({"error": f"Uploaded audio for slot '{slot}' not found."}), 404

    if uploaded_path.exists():
        return send_from_directory(UPLOAD_DIR, uploaded_path.name)

    for s, fname in VERSION_FILES:
        if s == slot:
            default_path = AUDIO_DIR / fname
            if default_path.exists():
                return send_from_directory(AUDIO_DIR, fname)

    return jsonify({"error": f"Audio for slot '{slot}' not found."}), 404


# ==========================
# 9. AI Enhancement 路由
# ==========================

@app.route("/api/ai_enhance", methods=["POST"])
def api_ai_enhance():
    payload = request.get_json(force=True, silent=True) or {}
    v_name = clean_version_name(payload.get("version_name", "Unknown version"))
    feats = payload.get("features") or {}

    energy     = safe_float(feats.get("energy", 0.5), 0.5)
    valence    = safe_float(feats.get("valence", 0.5), 0.5)
    roughness  = safe_float(feats.get("roughness", 0.0), 0.0)
    brightness = safe_float(feats.get("brightness", 2500.0), 2500.0)
    loudness   = safe_float(feats.get("loudness", -20.0), -20.0)
    key_sig    = feats.get("key") or "Unknown"

    sys_prompt = (
        "You are an expert music critic and generative artist. "
        "You translate audio features into concise emotional English descriptions and representative colors."
    )

    user_prompt = f"""
Analyze this music version segment:

- Version: {v_name}
- Key / Tonality: {key_sig}
- Energy (0–1): {energy:.2f}
- Valence (0–1): {valence:.2f}
- Roughness (0–1): {roughness:.2f}
- Brightness (Hz, roughly related to spectral centroid): {brightness:.1f}
- Loudness (dB): {loudness:.1f}

Tasks:
1. [commentary] Write a short, vivid description of the emotion in at most 40 English words.
2. [rgb] Choose one RGB color [R,G,B] (0–255) that fits this emotion.

Output JSON only in the format:
{{"commentary": "...", "rgb": [R, G, B]}}
"""

    if client is None:
        time.sleep(0.2)
        mood = "energetic" if energy > 0.6 else "calm"
        tone = "bright" if valence >= 0.5 else "melancholic"
        txt = f"{v_name} feels {mood} and {tone}, with {key_sig} color and texture."
        rgb = [255, 180, 120] if valence >= 0.5 else [90, 110, 210]
        return jsonify({"commentary": txt, "rgb": rgb})

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content)

        commentary = (parsed.get("commentary") or "").strip()
        rgb = parsed.get("rgb")

        if isinstance(rgb, list) and len(rgb) == 3:
            rgb = [int(max(0, min(255, int(v)))) for v in rgb]
        else:
            rgb = None

        return jsonify({"commentary": commentary, "rgb": rgb})

    except Exception as exc:
        print("[ERROR] /api/ai_enhance:", exc)
        return jsonify({"commentary": "AI analysis unavailable.", "rgb": None})


@app.route("/api/ai_compare_all", methods=["POST"])
def api_ai_compare_all():
    """
    ✅ 修复：
    - per_version 用 slot 做 key，永不丢失
    - 只分析 payload 里真实传入的版本（你只传2个就只分析2个）
    - AI 文本用 display_name（用户命名）
    """
    payload = request.get_json(force=True, silent=True) or {}
    versions = payload.get("versions") or []

    if not versions or not isinstance(versions, list):
        return jsonify({"error": "Invalid payload. Expect {versions:[...] }"}), 400

    cleaned_versions = []
    for v in versions:
        if not isinstance(v, dict):
            continue
        slot = (v.get("slot") or "").strip()
        name = (v.get("name") or slot or "version").strip()
        segs = v.get("segments") or []
        if not slot:
            continue
        if not isinstance(segs, list) or len(segs) == 0:
            continue
        cleaned_versions.append({"slot": slot, "name": name, "segments": segs})

    if not cleaned_versions:
        return jsonify({"error": "No valid versions (need slot + segments)."}), 400

    # mock mode
    if (not OPENAI_API_KEY) or client is None:
        per_version = {}
        for v in cleaned_versions:
            slot = v["slot"]
            display_name = clean_version_name(v["name"])
            per_version[slot] = {
                "display_name": display_name,
                "commentary": f"[Mock] {display_name}: richer AI output requires OPENAI_API_KEY.",
                "rgb": [120, 180, 255]
            }
        return jsonify({
            "per_version": per_version,
            "overall_compare": "[Mock] Upload versions and set OPENAI_API_KEY for detailed comparison."
        })

    def pick_representative_segments(segs):
        if not segs:
            return []
        s2 = [s for s in segs if isinstance(s, dict)]
        if not s2:
            return []

        def gv(s, k, d=0.0):
            try:
                vv = s.get(k, d)
                return float(vv) if vv is not None else float(d)
            except Exception:
                return float(d)

        by_energy = sorted(s2, key=lambda s: gv(s, "energy", 0.5))
        by_val    = sorted(s2, key=lambda s: gv(s, "valence", 0.5))
        by_rough  = sorted(s2, key=lambda s: gv(s, "roughness", 0.3))

        picks = [by_energy[0], by_energy[-1], by_val[0], by_val[-1], by_rough[-1]]

        seen = set()
        uniq = []
        for s in picks:
            key = (str(s.get("name")), float(s.get("start", 0.0)))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(s)
        return uniq[:5]

    compact_versions = []
    for v in cleaned_versions:
        slot = v["slot"]
        display_name = clean_version_name(v["name"])
        segs = v["segments"]
        rep = pick_representative_segments(segs)

        def mean_of(key, default):
            vals = []
            for s in segs:
                try:
                    val = s.get(key, None)
                    if val is None:
                        continue
                    vals.append(float(val))
                except Exception:
                    pass
            return float(np.mean(vals)) if vals else float(default)

        summary = {
            "tempo": mean_of("tempo", 90),
            "energy": mean_of("energy", 0.5),
            "valence": mean_of("valence", 0.5),
            "roughness": mean_of("roughness", 0.3),
            "brightness": mean_of("brightness", 2400),
            "loudness": mean_of("loudness", -20),
            "key": (segs[0].get("key") if segs and isinstance(segs[0], dict) else "Unknown")
        }

        compact_versions.append({
            "slot": slot,
            "display_name": display_name,
            "summary": summary,
            "representative_segments": rep
        })

    sys_prompt = (
        "You are an expert music analyst. "
        "Write concrete, specific comparisons between versions and between segments. "
        "Always mention measurable differences using the provided features. "
        "When referring to a segment, cite its segment name."
    )

    user_prompt = f"""
We have multiple versions of the same song.
Each version includes:
- slot (stable id)
- display_name (what to show the user)
- summary
- representative_segments

DATA (JSON):
{json.dumps(compact_versions, ensure_ascii=False)}

TASK:
Return JSON with:
1) per_version: keyed by slot. Each value:
   - display_name (copy from input)
   - commentary: 120-200 words. Must refer to display_name (NOT words like original/piano/duet).
   - rgb: [R,G,B]
2) overall_compare: 120-220 words comparing all display_name versions.

Return JSON only:
{{
  "per_version": {{
    "original": {{"display_name":"...", "commentary":"...", "rgb":[..]}},
    "piano":    {{"display_name":"...", "commentary":"...", "rgb":[..]}}
  }},
  "overall_compare": "..."
}}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content)

        per_version = parsed.get("per_version") or {}
        for slot, obj in per_version.items():
            rgb = obj.get("rgb")
            if isinstance(rgb, list) and len(rgb) == 3:
                obj["rgb"] = [int(max(0, min(255, int(x)))) for x in rgb]
            else:
                obj["rgb"] = None

        return jsonify({
            "per_version": per_version,
            "overall_compare": parsed.get("overall_compare", "")
        })
    except Exception as exc:
        print("[ERROR] /api/ai_compare_all:", exc)
        return jsonify({
            "per_version": {},
            "overall_compare": "AI analysis unavailable."
        }), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
