# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import numpy as np
import pretty_midi as pm
from basic_pitch import predict

# ---------- utils ----------
def load_json(path: Path):
    return json.loads(path.read_text())

def seconds_to_ticks(sec, ppq, bpm):
    # ticks = seconds * (PPQ * BPM / 60)
    return int(round(sec * (ppq * bpm / 60.0)))

def quantize_times(times_s, ppq, bpm):
    ticks = [seconds_to_ticks(t, ppq, bpm) for t in times_s]
    return [max(0, int(round(x))) for x in ticks]

def events_from_notes(notes, ppq, bpm):
    # notes: list of (on_s, off_s, pitch, vel)
    # produce NOTE_ON<x>, TIME_SHIFT<y>, NOTE_OFF<x> …
    events = []
    cur_tick = 0
    for on_s, off_s, pitch, vel in notes:
        on_tick  = seconds_to_ticks(on_s,  ppq, bpm)
        off_tick = seconds_to_ticks(off_s, ppq, bpm)
        # time shift to NOTE_ON
        dt = max(0, on_tick - cur_tick)
        if dt > 0: events.append(f"TIME_SHIFT<{dt}>")
        events.append(f"NOTE_ON<{pitch}>")
        # time shift to NOTE_OFF
        dt2 = max(0, off_tick - on_tick)
        if dt2 > 0: events.append(f"TIME_SHIFT<{dt2}>")
        events.append(f"NOTE_OFF<{pitch}>")
        cur_tick = off_tick
    return " ".join(events)

# ---------- OFR hook ----------
def run_ofr_model(audio_wav: Path, out_mid: Path):
    """
    Implementação usando basic-pitch (CRNN/OFR-like model).
    Basic-pitch é um modelo de transcrição de música que estima notas a partir de áudio.
    Retorna lista de notas [(on_s, off_s, pitch, velocity), ...].
    """
    # Carrega áudio
    from basic_pitch import predict_and_save
    import librosa

    # Basic-pitch espera áudio mono, mas lida internamente
    # Predict retorna model_output com 'notes': list of (start_time_seconds, end_time_seconds, pitch_midi)
    model_output = predict(str(audio_wav))

    notes = []
    for start, end, pitch in model_output['notes']:
        # Velocity padrão, pois basic-pitch não estima velocity
        vel = 80
        notes.append((float(start), float(end), int(pitch), vel))

    # Salva MIDI usando basic-pitch (opcional, mas para consistência)
    predict_and_save([str(audio_wav)], '.', save_midi=True, sonify_midi=False, save_model_outputs=False, save_notes=False)

    # Basic-pitch salva como audio_wav.stem_basic_pitch.mid
    # Mas para usar out_mid, vamos criar com pretty_midi
    save_pretty_midi(notes, out_mid, bpm=120.0)  # BPM placeholder, será ajustado depois

    return notes

# ---------- Fallback heurístico ----------
def fallback_notes_from_f0_onsets(f0_json: Path, onsets_json: Path):
    f0 = load_json(f0_json)
    on = load_json(onsets_json)
    times = np.array(f0["time_s"], dtype=float)
    f0hz  = np.array(f0["f0_hz"], dtype=float)
    onsets = list(on["onsets_time_s"])

    # estima pitch próximo ao onset (média em pequena janela)
    win = 3  # ~30 ms se step 10 ms
    notes = []
    for i, t_on in enumerate(onsets):
        # define off como próximo onset ou fim de série
        t_off = onsets[i+1] if i+1 < len(onsets) else (times[-1] if len(times) else t_on + 0.3)
        # janela de f0 perto do onset
        idx = np.searchsorted(times, t_on)
        lo = max(0, idx - win); hi = min(len(times), idx + win + 1)
        f0_local = f0hz[lo:hi]
        # pitch em MIDI
        if len(f0_local) == 0 or np.max(f0_local) <= 0:
            # sem confiança, pula
            continue
        f0_mean = float(np.median(f0_local[f0_local > 0]))
        pitch = int(round(69 + 12 * np.log2(max(1e-6, f0_mean) / 440.0)))
        pitch = max(21, min(108, pitch))  # range (A0..C8 aprox)
        vel = 80
        # evita notas minúsculas
        if t_off - t_on < 0.04:
            t_off = t_on + 0.04
        notes.append((float(t_on), float(t_off), pitch, vel))
    return notes

def save_pretty_midi(notes, out_mid: Path, bpm: float = 120.0):
    pmid = pm.PrettyMIDI(initial_tempo=bpm)
    inst = pm.Instrument(program=25)  # Acoustic Guitar (nylon)
    for on_s, off_s, pitch, vel in notes:
        inst.notes.append(pm.Note(velocity=int(vel), pitch=int(pitch), start=float(on_s), end=float(off_s)))
    pmid.instruments.append(inst)
    out_mid.parent.mkdir(parents=True, exist_ok=True)
    pmid.write(str(out_mid))

def main():
    ap = argparse.ArgumentParser(description="Step04 — Audio→MIDI (OFR) + quantização + tokens.")
    ap.add_argument("--audio", required=True, help="Áudio base (use guitarra isolada).")
    ap.add_argument("--tempo_map", required=True, help="Arquivo tempo_map.json do Step02.")
    ap.add_argument("--f0_json", required=False, help="f0.json (para fallback).")
    ap.add_argument("--onsets_json", required=False, help="onsets.json (para fallback).")
    ap.add_argument("--out_midi_dir", default="work/midi", help="Pasta para midi.mid e events.txt")
    ap.add_argument("--ppq", type=int, default=480)
    ap.add_argument("--use_ofr", action="store_true", help="Use CRNN/OFR (necessita implementação do gancho).")
    args = ap.parse_args()

    audio = Path(args.audio)
    tempo_map = load_json(Path(args.tempo_map))
    bpm = float(tempo_map.get("bpm_default", 120.0))
    ppq = int(tempo_map.get("ppq", args.ppq))

    midi_dir = Path(args.out_midi_dir); midi_dir.mkdir(parents=True, exist_ok=True)
    midi_mid = midi_dir / "midi.mid"
    events_txt = midi_dir / "events.txt"

    # 1) notas via OFR ou fallback f0+onsets
    if args.use_ofr:
        notes = run_ofr_model(audio, midi_mid)  # você implementa
    else:
        if not (args.f0_json and args.onsets_json):
            raise ValueError("Para fallback, informe --f0_json e --onsets_json (gerados no Step03).")
        notes = fallback_notes_from_f0_onsets(Path(args.f0_json), Path(args.onsets_json))
        # salva midi bruto (em segundos, antes da quantização)
        save_pretty_midi(notes, midi_mid, bpm=bpm)

    # 2) quantização simples para PPQ + tokens
    # (opcional: melhore com grade de beats do tempo_map["beats_s"])
    notes_q = []
    for on_s, off_s, pitch, vel in notes:
        on_ticks  = seconds_to_ticks(on_s,  ppq, bpm)
        off_ticks = seconds_to_ticks(off_s, ppq, bpm)
        # volta para segundos após quantização (para escrever MIDI coerente)
        on_s_q  = on_ticks  * (60.0 / (ppq * bpm))
        off_s_q = off_ticks * (60.0 / (ppq * bpm))
        notes_q.append((on_s_q, off_s_q, pitch, vel))
    save_pretty_midi(notes_q, midi_mid, bpm=bpm)

    # 3) gerar tokens NOTE_ON/OFF + TIME_SHIFT
    notes_q.sort(key=lambda x: x[0])
    events = events_from_notes(notes_q, ppq=ppq, bpm=bpm)
    events_txt.write_text(events)
    print("[OK] midi:", midi_mid)
    print("[OK] events:", events_txt)

if __name__ == "__main__":
    main()
