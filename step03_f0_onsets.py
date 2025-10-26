import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

# libs opcionais
import librosa

# tentamos torchcrepe primeiro (rápido, GPU/CPU)
_HAS_TORCHCREPE = False
try:
    import torch
    import torchcrepe
    _HAS_TORCHCREPE = True
except Exception:
    _HAS_TORCHCREPE = False

# tentamos CREPE (keras) como alternativa
_HAS_CREPE = False
try:
    import crepe as crepe_keras
    _HAS_CREPE = True
except Exception:
    _HAS_CREPE = False


def load_audio_mono(path: Path, target_sr: int = 16000):
    """Carrega áudio mono, normaliza para [-1,1], reamostra para target_sr."""
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim == 2:
        y = np.mean(y, axis=1)
    # evita NaN/inf
    y = np.nan_to_num(y)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
        sr = target_sr
    # normalização leve
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak
    return y.astype(np.float32), sr


def compute_f0_torchcrepe(y: np.ndarray, sr: int, hop_ms: float = 10.0,
                          fmin: float = 55.0, fmax: float = 1760.0,
                          model: str = "full", device: str = "cpu",
                          batch_size: int = 1024, min_conf: float = 0.3):
    """
    f0 com torchcrepe (rápido; aceita GPU). Requer sr=16000 ou 32000 (recomendado 16k).
    Retorna times (s), f0_hz (0 para frames inseguros) e confidences.
    """
    if sr not in (16000, 32000):
        raise ValueError("torchcrepe requer SR 16000 ou 32000. Reamostre antes (o loader já reamostra).")

    hop_length = int(sr * (hop_ms / 1000.0))

    device_t = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    audio = torch.from_numpy(y).to(device_t)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)  # (1, T)

    # torchcrepe espera amplitude em [-1,1]
    with torch.no_grad():
        pitch, periodicity = torchcrepe.predict(
            audio,
            sr,
            hop_length,
            fmin=fmin,
            fmax=fmax,
            model=model,            # 'full' mais preciso; 'tiny' mais rápido
            batch_size=batch_size,
            device=device_t,
            return_periodicity=True,
        )
    # pós-processamento sugerido
    pitch = torchcrepe.filter.median(pitch, 3)
    periodicity = torchcrepe.filter.mean(periodicity, 3)
    # pitch = torchcrepe.threshold.Silence(-60.0)(pitch, periodicity)  # remove silêncio - comentado por erro de shape
    f0 = pitch.squeeze(0).cpu().numpy()
    conf = periodicity.squeeze(0).cpu().numpy()

    # zera f0 abaixo da confiança mínima
    f0_thresholded = np.where(conf >= min_conf, f0, 0.0)

    times = np.arange(len(f0_thresholded)) * (hop_length / sr)
    return times, f0_thresholded.astype(float), conf.astype(float)


def compute_f0_crepe_keras(y: np.ndarray, sr: int, hop_ms: float = 10.0,
                           fmin: float = 55.0, fmax: float = 1760.0, model_capacity: str = "full",
                           min_conf: float = 0.3):
    """
    f0 com CREPE (keras). sr pode ser 44.1k; crepe internamente faz reamostragem.
    """
    step_size = hop_ms  # crepe usa ms diretamente
    time, frequency, confidence, _ = crepe_keras.predict(
        y, sr, step_size=step_size, verbose=0, viterbi=False, model_capacity=model_capacity
    )
    # aplica faixa f0
    frequency = np.where((frequency >= fmin) & (frequency <= fmax), frequency, 0.0)
    # confiança mínima
    frequency = np.where(confidence >= min_conf, frequency, 0.0)
    return time.astype(float), frequency.astype(float), confidence.astype(float)


def compute_onsets_librosa(y: np.ndarray, sr: int, hop_ms: float = 10.0,
                           backtrack: bool = True, pre_max: int = 3, post_max: int = 3,
                           pre_avg: int = 3, post_avg: int = 3, delta: float = 0.0):
    """
    Onsets via fluxo espectral do librosa.
    Retorna tempos (s) dos onsets.
    """
    hop_length = int(sr * (hop_ms / 1000.0))
    # envelope de onsets com filtro mel
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, aggregate=np.median)
    onsets_frames = librosa.onset.onset_detect(
        onset_envelope=oenv, sr=sr, hop_length=hop_length,
        backtrack=backtrack, pre_max=pre_max, post_max=post_max, pre_avg=pre_avg, post_avg=post_avg, delta=delta
    )
    onsets_times = librosa.frames_to_time(onsets_frames, sr=sr, hop_length=hop_length)
    return onsets_times.astype(float)


def save_csv(path: Path, header: str, rows: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Teste de Pitch (f0) & Onsets — CREPE/torchcrepe + librosa.")
    ap.add_argument("--source_dir", required=True, help="Diretório com os stems do demucs (ex: demucs_jobs/20251025-200323/demucs/htdemucs_6s/source/).")
    ap.add_argument("--stem", default="guitar", help="Nome do stem a processar (sem .wav, ex: guitar). Se 'all', processa todos.")
    ap.add_argument("--outdir", default="f0_onsets_out", help="Pasta de saída (CSV/JSON/PNG).")
    ap.add_argument("--backend", choices=["torchcrepe", "crepe"], default="torchcrepe", help="Backend de f0.")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Dispositivo para torchcrepe.")
    ap.add_argument("--sr", type=int, default=16000, help="SR alvo (torchcrepe: 16000 recomendado).")
    ap.add_argument("--hop_ms", type=float, default=10.0, help="Hop em milissegundos para f0/onsets.")
    ap.add_argument("--min_conf", type=float, default=0.3, help="Confiança mínima do f0 (zerei abaixo disso).")
    ap.add_argument("--fmin", type=float, default=55.0, help="f0 mínimo (Hz).")
    ap.add_argument("--fmax", type=float, default=1760.0, help="f0 máximo (Hz).")
    ap.add_argument("--plot", action="store_true", help="Gera PNG com waveform + f0 + onsets.")
    args = ap.parse_args()

    source_dir = Path(args.source_dir)
    outdir = Path(args.outdir)

    if not source_dir.exists() or not source_dir.is_dir():
        raise ValueError(f"Diretório de source não existe: {source_dir}")

    wav_files = list(source_dir.glob("*.wav"))
    if not wav_files:
        raise ValueError(f"Nenhum arquivo .wav encontrado em: {source_dir}")

    if args.stem != "all":
        wav_files = [p for p in wav_files if p.stem == args.stem]
        if not wav_files:
            raise ValueError(f"Stem '{args.stem}.wav' não encontrado em: {source_dir}")

    outdir.mkdir(parents=True, exist_ok=True)

    for wav_path in wav_files:
        stem_name = wav_path.stem
        outdir_stem = outdir / stem_name
        outdir_stem.mkdir(parents=True, exist_ok=True)

        print(f"[info] processando stem: {stem_name} de {wav_path}")
        y, sr = load_audio_mono(wav_path, target_sr=args.sr)

        # ----- f0 -----
        if args.backend == "torchcrepe":
            if not _HAS_TORCHCREPE:
                raise RuntimeError("torchcrepe não está disponível. Instale ou use --backend crepe.")
            print(f"[info] f0: torchcrepe para {stem_name}")
            times, f0, conf = compute_f0_torchcrepe(
                y, sr, hop_ms=args.hop_ms, fmin=args.fmin, fmax=args.fmax,
                model="full", device=args.device, min_conf=args.min_conf
            )
        else:
            if not _HAS_CREPE:
                raise RuntimeError("crepe (keras) não está disponível. Instale ou use --backend torchcrepe.")
            print(f"[info] f0: CREPE (keras) para {stem_name}")
            times, f0, conf = compute_f0_crepe_keras(
                y, sr, hop_ms=args.hop_ms, fmin=args.fmin, fmax=args.fmax,
                model_capacity="full", min_conf=args.min_conf
            )

        # ----- onsets -----
        print(f"[info] onsets: librosa.onset para {stem_name}")
        onsets_times = compute_onsets_librosa(y, sr, hop_ms=args.hop_ms, backtrack=True)

        # ----- salvar resultados -----
        f0_csv = outdir_stem / "f0.csv"
        save_csv(f0_csv, "time_s,f0_hz,confidence", np.column_stack([times, f0, conf]))
        (outdir_stem / "f0.json").write_text(json.dumps({"time_s": times.tolist(), "f0_hz": f0.tolist(), "confidence": conf.tolist()}, ensure_ascii=False, indent=2))
        on_csv = outdir_stem / "onsets.csv"
        save_csv(on_csv, "onset_time_s", onsets_times.reshape(-1, 1))
        (outdir_stem / "onsets.json").write_text(json.dumps({"onsets_time_s": onsets_times.tolist()}, ensure_ascii=False, indent=2))

        print(f"[ok] salvo para {stem_name}: {f0_csv}")
        print(f"[ok] salvo para {stem_name}: {on_csv}")

        # ----- gráfico opcional -----
        if args.plot:
            try:
                import matplotlib.pyplot as plt
                # waveform (normalizada)
                t_wav = np.arange(len(y)) / sr
                plt.figure(figsize=(14, 6))
                plt.plot(t_wav, y, linewidth=0.5, alpha=0.7, label="waveform")
                # f0 (onde > 0)
                mask = f0 > 0
                plt.plot(times[mask], (f0[mask] / np.max(f0[mask])) * 0.8, linewidth=1.2, label="f0 (norm.)")
                # onsets (linhas verticais)
                for t in onsets_times:
                    plt.axvline(t, linestyle="--", linewidth=0.8, alpha=0.6)
                plt.title(f"Waveform + f0 + Onsets ({stem_name})")
                plt.xlabel("Tempo (s)")
                plt.legend(loc="upper right")
                plt.tight_layout()
                png_path = outdir_stem / "f0_onsets.png"
                plt.savefig(png_path, dpi=150)
                plt.close()
                print(f"[ok] salvo para {stem_name}: {png_path}")
            except Exception as e:
                print(f"[warn] falhou ao gerar gráfico para {stem_name}: {e}")

    print("\n=== PRONTO ===")
    print(f"Saída em: {outdir.resolve()}")
    print("Para cada stem: f0.csv, onsets.csv, onsets.json", "(e f0_onsets.png)" if args.plot else "")
    

if __name__ == "__main__":
    main()
