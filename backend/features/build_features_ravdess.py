from pathlib import Path

from backend.data.loader import load_metadata, dataframe_to_recordings
from backend.features.extractor import extract_features_for_recordings

def main() -> None:
    root = Path(__file__).resolve().parents[3]
    dataset = "ravdess"

    metadata_csv = root / "data" / "raw" / dataset / "metadata.csv"
    audio_root = root / "data" / "raw" / dataset / "audio"
    out_path = root / "data" / "processed" / f"{dataset}_features.parquet"

    df_meta = load_metadata(metadata_csv, audio_root)
    recs = dataframe_to_recordings(df_meta)
    df_feats = extract_features_for_recordings(recs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_feats.to_parquet(out_path, index=False)
    print(f"Saved features to {out_path}")

if __name__ == "__main__":
    main()