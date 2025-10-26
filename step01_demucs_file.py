import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Forçar backend soundfile para torchaudio
try:
    import torchaudio
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

# ==========================================================
# 🌐 URL GLOBAL — edite aqui sua música do YouTube
YOUTUBE_URL = "https://www.youtube.com/watch?v=I3MTGhRC82s&list=RDI3MTGhRC82s"
# ==========================================================
# Caminho do FFmpeg (ajuste se preciso)
path_to_ffmpeg = r"C:/ffmpeg-6.1-essentials_build/bin"
os.environ["PATH"] += os.pathsep + path_to_ffmpeg

def assert_tool_in_path(tool: str):
    from shutil import which
    if which(tool) is None:
        raise RuntimeError(f"Ferramenta '{tool}' não encontrada no PATH. Instale e/ou ajuste o PATH.")

def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)

def sanitize_name(s: str) -> str:
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:120] if s else "audio"

def download_audio_ytdlp(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(out_dir / "%(title)s.%(ext)s")
    cmd = [
        "yt-dlp",
        url,
        "--no-playlist",        # baixa apenas este vídeo, mesmo com parâmetro list=
        "-f", "bestaudio/best",
        "-o", out_tmpl,
        "-x", "--audio-format", "wav",
        "--embed-metadata"
    ]
    run(cmd)
    candidates = list(out_dir.glob("*.wav"))
    if not candidates:
        raise FileNotFoundError("⚠️ Não foi possível baixar o áudio em formato WAV.")
    # pega o mais recente
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def run_demucs(wav_in: Path, out_root: Path, model: str = "htdemucs_6s") -> Path:
    """
    Executa Demucs via CLI, força CPU, e retorna o diretório de stems mais recente.
    Para guitarra isolada, use 'htdemucs_6s'.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "demucs",
        "-d", "cpu",          # força CPU
        "-n", model,
        "-o", str(out_root),
        str(wav_in)
    ]
    run(cmd)
    model_dir = out_root / model
    if not model_dir.exists():
        raise FileNotFoundError(f"Saída esperada não encontrada: {model_dir}")

    # escolhe o subdir mais recente (um por arquivo processado)
    subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError("Demucs rodou, mas não encontrei subdiretórios de stems.")
    subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return subdirs[0]

def main():
    ap = argparse.ArgumentParser(description="Step01 — Download YouTube + Demucs separation.")
    ap.add_argument("--url", help="URL do YouTube (se não informado, usa YOUTUBE_URL global).")
    ap.add_argument("--audio_file", help="Arquivo WAV local (pula download se informado).")
    args = ap.parse_args()

    if args.audio_file:
        # usa arquivo local
        wav_src = Path(args.audio_file)
        if not wav_src.exists():
            print(f"❌ Arquivo não encontrado: {wav_src}")
            sys.exit(1)
        job_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_dir = Path("demucs_jobs") / job_id
        dl_dir = job_dir / "download"
        demucs_out = job_dir / "demucs"
        job_dir.mkdir(parents=True, exist_ok=True)
        source_wav = job_dir / "source.wav"
        shutil.copy2(wav_src, source_wav)
        print(f"[OK] Áudio local copiado para: {source_wav}")
    else:
        # download do YouTube
        youtube_url = args.url or YOUTUBE_URL
        if not youtube_url or "youtube" not in youtube_url:
            print("❌ URL inválida. Informe --url ou edite YOUTUBE_URL.")
            sys.exit(1)

        # checagens rápidas
        try:
            assert_tool_in_path("yt-dlp")
        except RuntimeError as e:
            print(f"[ERRO] {e}")
            sys.exit(1)
        try:
            assert_tool_in_path("ffmpeg")
        except RuntimeError:
            print("[AVISO] ffmpeg não detectado no PATH. Tente ajustar 'path_to_ffmpeg' no topo do script.")

        job_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_dir = Path("demucs_jobs") / job_id
        dl_dir = job_dir / "download"
        demucs_out = job_dir / "demucs"
        job_dir.mkdir(parents=True, exist_ok=True)

        print(f"[AUDIO] Job: {job_id}")
        print(f"[DOWNLOAD] Baixando áudio do YouTube...")
        wav_src = download_audio_ytdlp(youtube_url, dl_dir)

        source_wav = job_dir / "source.wav"
        shutil.copy2(wav_src, source_wav)
        print(f"[OK] Áudio salvo em: {source_wav}")

    print(f"[DEMIX] Rodando Demucs (modelo: htdemucs_6s, CPU)...")
    stem_dir = run_demucs(source_wav, demucs_out, model="htdemucs_6s")
    print(f"[OK] Stems salvos em: {stem_dir}")

    stems = list(stem_dir.glob("*.wav"))
    if not stems:
        print("Nenhum stem encontrado!")
    else:
        print("\n=== PRONTO ===")
        print(f"Job dir: {job_dir}")
        print(f"- source: {source_wav}")
        for s in stems:
            print(f"- stem: {s.name} -> {s}")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n[ERRO] Comando falhou com código {e.returncode}:\n  {' '.join(e.cmd)}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\n[ERRO] {e}")
        sys.exit(1)
