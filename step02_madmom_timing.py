# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import numpy as np
import librosa

def beat_and_tempo_map(wav_path: Path):
    # Carrega áudio
    y, sr = librosa.load(str(wav_path), sr=None)

    # Detecta tempo usando librosa
    tempo = librosa.beat.tempo(y=y, sr=sr, start_bpm=120)[0]  # retorna array, pega primeiro

    # Detecta beats
    _, beats = librosa.beat.beat_track(y=y, sr=sr, start_bpm=tempo, units='time')

    # beats são os tempos dos beats
    beats = beats.astype(float)

    # Cria tempo_map simples
    tempo_map = {
        "ppq": 480,
        "bpm_default": float(tempo),
        "beats_s": beats.tolist()
    }
    return beats, tempo_map

def main():
    ap = argparse.ArgumentParser(description="Step02 — librosa: beat times + tempo map.")
    ap.add_argument("--wav", required=True, help="Caminho do WAV (guitar.wav funciona melhor).")
    ap.add_argument("--outdir", default="work/timing", help="Pasta de saída.")
    args = ap.parse_args()

    wav = Path(args.wav)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    beats, tempo_map = beat_and_tempo_map(wav)

    (outdir / "beat_times.json").write_text(json.dumps({"beats_s": beats.tolist()}, indent=2))
    (outdir / "tempo_map.json").write_text(json.dumps(tempo_map, indent=2))
    print("[OK] timing salvo em:", outdir)

if __name__ == "__main__":
    main()
