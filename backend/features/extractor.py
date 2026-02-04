from pathlib import Path
from typing import Iterable, Dict, Any, List

import librosa
import numpy as np
import pandas as pd

from backend.data.loader import Recording

def load_wave(path: Path, target_sr: int = 16_000, max_duration_s: int = 120) -> tuple[np.ndarray, int]:
    y,sr = librosa.load(path, sr=target_sr, mono=True)
    y,_ = librosa.effects.trim(y,top_db=30)
    max_len = int(max_duration_s * sr)
    if len(y) > max_len:
        y = y[:max_len]
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y, int(sr)

def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size > 0 else 0.0

def safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if x.size > 0 else 0.0

def extract_features_from_waveform(y: np.ndarray, sr:int) -> Dict[str, Any]:
    duration_s = len(y) / sr if len(y) > 0 else 0.0

    rms = librosa.feature.rms(y=y)[0]
    energy_mean = safe_mean(rms)
    energy_std = safe_std(rms)

    pitches, mags = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[mags > np.median(mags)]
    pitch_mean = safe_mean(pitch_vals)
    pitch_std = safe_std(pitch_vals)

    zcr = librosa.feature.zero_crossing_rate(y)[0]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds = np.std(mfcc, axis=1)

    thr = np.percentile(rms, 20)
    is_silence = rms < thr
    pauses = []
    cur = 0
    for v in is_silence:
        if v:
            cur += 1
        elif cur > 0:
            pauses.append(cur)
            cur = 0
    if cur > 0:
        pauses.append(cur)
    pause_count = len(pauses)
    pause_mean = float(np.mean(pauses)) if pauses else 0.0

    voiced_frames = (~is_silence).sum()
    speech_rate_proxy = float(voiced_frames / duration_s) if duration_s else 0.0

    feats: Dict[str, Any] = {
        "duration_s": duration_s,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "zcr_mean": safe_mean(zcr),
        "zcr_std": safe_std(zcr),
        "pause_count": float(pause_count),
        "pause_mean_frames": pause_mean,
        "speech_rate_proxy": speech_rate_proxy,
    }

    for i, (m, s) in enumerate(zip(mfcc_means, mfcc_stds), start=1):
        feats[f"mfcc{i}_mean"] = float(m)
        feats[f"mfcc{i}_std"] = float(s)

    return feats

def extract_features_for_recordings(recs: Iterable[Recording]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rec in recs:
        y, sr = load_wave(rec.path)
        feats = extract_features_from_waveform(y, sr)
        row: Dict[str, Any] = {
            "id": rec.id,
            "emotion": rec.emotion,
            "actor": rec.actor,
            "phq9_total": rec.phq9_total,
            "gad7_total": rec.gad7_total,
        }
        row.update(feats)
        rows.append(row)
    return pd.DataFrame(rows)