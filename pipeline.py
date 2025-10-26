#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline completa: Áudio Bruto → Demucs → Madmom → F0/Onsets → Audio2MIDI → MIDI2TAB

Executa os steps em sequência, passando saídas como entradas.
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_step(script: str, args: list, cwd: Path = None):
    """Executa um step e verifica sucesso."""
    script_path = Path(__file__).parent / script  # caminho absoluto do script
    cmd = [sys.executable, str(script_path)] + args
    print(f"\n>>> Executando: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd, cwd=cwd)
        print(f"✓ {script} concluído com sucesso.")
    except subprocess.CalledProcessError as e:
        print(f"✗ Falha em {script}: {e}")
        sys.exit(1)

def main():
    ap = argparse.ArgumentParser(description="Pipeline completa para geração de TAB a partir de áudio.")
    ap.add_argument("--youtube_url", help="URL do YouTube para baixar o áudio.")
    ap.add_argument("--audio_file", help="Arquivo WAV local (pula download se informado).")
    ap.add_argument("--work_dir", default="pipeline_work", help="Diretório de trabalho para saídas.")
    ap.add_argument("--use_ofr", action="store_true", help="Usar modelo OFR em step04 (necessita implementação).")
    args = ap.parse_args()

    if not args.youtube_url and not args.audio_file:
        print("❌ Informe --youtube_url ou --audio_file.")
        sys.exit(1)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Step01: Demucs - baixa YouTube e separa stems
    print("=== STEP 01: Demucs (Download + Separação) ===")
    step01_args = []
    if args.youtube_url:
        step01_args += ["--url", args.youtube_url]
    if args.audio_file:
        step01_args += ["--audio_file", args.audio_file]
    run_step("step01_demucs_file.py", step01_args)  # não define cwd para step01, pois cria demucs_jobs no dir do script

    # Encontrar o job_dir mais recente
    demucs_jobs = Path("demucs_jobs")
    if not demucs_jobs.exists():
        raise FileNotFoundError("demucs_jobs não encontrado. Execute step01 primeiro.")
    job_dirs = [d for d in demucs_jobs.iterdir() if d.is_dir()]
    job_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    job_dir = job_dirs[0] if job_dirs else None
    if not job_dir:
        raise FileNotFoundError("Nenhum job encontrado em demucs_jobs.")
    print(f"Usando job: {job_dir}")

    # Caminhos dos stems
    stem_dir = job_dir / "demucs" / "htdemucs_6s" / "source"
    source_dir = stem_dir
    guitar_wav = source_dir / "guitar.wav"
    if not guitar_wav.exists():
        raise FileNotFoundError(f"guitar.wav não encontrado em {source_dir}")

    # Step02: Madmom - timing para guitar.wav
    print("=== STEP 02: Madmom (Beats + Tempo) ===")
    timing_dir = work_dir / "timing"
    run_step("step02_madmom_timing.py", [
        "--wav", str(guitar_wav),
        "--outdir", str(timing_dir)
    ])

    # Step03: F0 & Onsets - para guitar
    print("=== STEP 03: F0 & Onsets ===")
    pitch_dir = work_dir / "pitch"
    run_step("step03_f0_onsets.py", [
        "--source_dir", str(source_dir),
        "--stem", "guitar",
        "--outdir", str(pitch_dir)
    ])

    # Caminhos dos arquivos gerados
    tempo_map_json = timing_dir / "tempo_map.json"
    f0_json = pitch_dir / "guitar" / "f0.json"
    onsets_json = pitch_dir / "guitar" / "onsets.json"

    # Step04: Audio2MIDI
    print("=== STEP 04: Audio → MIDI ===")
    midi_dir = work_dir / "midi"
    step04_args = [
        "--audio", str(guitar_wav),
        "--tempo_map", str(tempo_map_json),
        "--out_midi_dir", str(midi_dir)
    ]
    if not args.use_ofr:
        step04_args += ["--f0_json", str(f0_json), "--onsets_json", str(onsets_json)]
    else:
        step04_args += ["--use_ofr"]
    run_step("step04_audio2midi.py", step04_args)

    # Step05: MIDI → TAB
    print("=== STEP 05: MIDI → TAB (Fretting-Transformer) ===")
    tab_dir = work_dir / "tab"
    run_step("step05_fretting_transformer.py", [
        "--events_txt", str(midi_dir / "events.txt"),
        "--out_tab_dir", str(tab_dir)
    ])

    print("\n� Pipeline completa!")
    print(f"Saídas finais em: {work_dir}")
    print("- timing/: beat_times.json, tempo_map.json")
    print("- pitch/guitar/: f0.json, onsets.json")
    print("- midi/: midi.mid, events.txt")
    print("- tab/: tab.txt")

if __name__ == "__main__":
    main()