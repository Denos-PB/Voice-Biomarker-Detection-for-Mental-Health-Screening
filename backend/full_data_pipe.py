from __future__ import annotations

import os
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[0]
_REPO_ROOT = _BACKEND_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import soundfile as sf

from backend.data.create_metadata import main as create_metadata_main
from backend.data.loader import load_metadata, dataframe_to_recordings, Recording
from backend.data.audioaug import AudioAugmentation
from backend.features.extractor import extract_features_for_recordings

def main() -> None:
    dataset = "ravdess"
    root = _BACKEND_DIR
    audio_root = root / "data" / "raw" / dataset / "audio"
    metadata_csv = root / "data" / "raw" / dataset / "metadata.csv"
    audio_augmented_root = root / "data" / "raw" / dataset / "audio_augmented"
    out_path = root / "data" / "processed" / f"{dataset}_features.parquet"

    os.chdir(root)
    create_metadata_main()

    df_meta = load_metadata(metadata_csv, audio_root)
    recs = dataframe_to_recordings(df_meta)
    print(f"Loaded {len(recs)} recordings from metadata.")

    audio_augmented_root.mkdir(parents=True, exist_ok=True)
    extra: list[Recording] = []
    for i, rec in enumerate(recs):
        if not rec.path.exists():
            print(f"Skipping missing: {rec.path}")
            continue
        try:
            aug = AudioAugmentation(str(rec.path), sr=22_050, random_state=42 + i)
            noisy, _ = aug.add_background_noise(noise_type="cafe", snr_db=10.0)
            out_name = f"{rec.id}_aug_cafe.wav"
            out_path_wav = audio_augmented_root / out_name
            sf.write(out_path_wav, noisy, aug.sr)
            extra.append(Recording(
                id=f"{rec.id}_aug_cafe",
                path=out_path_wav,
                emotion=rec.emotion,
                actor=rec.actor,
                phq9_total=rec.phq9_total,
                gad7_total=rec.gad7_total,
            ))
        except Exception as e:
            print(f"Augment failed {rec.path}: {e}")
    recs = recs + extra
    print(f"Added {len(extra)} augmented recordings.")

    df_feats = extract_features_for_recordings(recs)
    print(f"Extracted features: {df_feats.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_feats.to_parquet(out_path, index=False)
    print(f"Saved features to {out_path}")


if __name__ == "__main__":
    main()