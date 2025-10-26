# -*- coding: utf-8 -*-
"""
Processamento de dados para Fretting-Transformer.
Gera pares INPUT (MIDI tokens) -> OUTPUT (TAB tokens) de arquivos GP.
"""

import guitarpro as gp
from pathlib import Path

def gp_to_midi_tokens(gp_file):
    """
    Extrai tokens MIDI de GP file: NOTE_ON<pitch> TIME_SHIFT<dt> NOTE_OFF<pitch>
    """
    try:
        song = gp.parse(str(gp_file))
    except:
        print(f"Erro ao parsear {gp_file}")
        return ""
    notes = []
    for track in song.tracks:
        if track.isPercussionTrack:
            continue
        for measure in track.measures:
            for voice in measure.voices:
                for beat in voice.beats:
                    for note in beat.notes:
                        onset = beat.start
                        duration = beat.duration.time
                        offset = onset + duration
                        pitch = note.realValue  # MIDI pitch
                        notes.append((onset, offset, pitch))
    
    # Ordenar por onset
    notes.sort(key=lambda x: x[0])
    tokens = []
    cur_time = 0
    for onset, offset, pitch in notes:
        dt = onset - cur_time
        if dt > 0:
            tokens.append(f"TIME_SHIFT<{dt}>")
        tokens.append(f"NOTE_ON<{pitch}>")
        dt_off = offset - onset
        if dt_off > 0:
            tokens.append(f"TIME_SHIFT<{dt_off}>")
        tokens.append(f"NOTE_OFF<{pitch}>")
        cur_time = offset
    return " ".join(tokens)

def process_gp_file(gp_file, output_txt):
    """
    Gera INPUT: midi_tokens OUTPUT: tab_tokens
    """
    midi_tokens = gp_to_midi_tokens(gp_file)
    if not midi_tokens:
        return
    # Para TAB, usar rule-based
    from step05_fretting_transformer import midi_to_guitar_tab
    try:
        song = gp.parse(str(gp_file))
    except:
        return
    tab_tokens = []
    for track in song.tracks:
        if track.isPercussionTrack:
            continue
        for measure in track.measures:
            for voice in measure.voices:
                for beat in voice.beats:
                    for note in beat.notes:
                        pitch = note.realValue
                        pos = midi_to_guitar_tab(pitch)
                        if pos:
                            string, fret = pos
                            tab_tokens.append(f"TAB<{string},{fret}>")
    tab_str = " ".join(tab_tokens)
    
    with open(output_txt, 'w') as f:
        f.write(f"INPUT: {midi_tokens}\nOUTPUT: {tab_str}\n")
    print(f"Processado: {gp_file} -> {output_txt}")

def process_dataset(data_dir, output_dir):
    """
    Processa dataset: GP -> pares input/output.
    """
    output_dir.mkdir(exist_ok=True)
    for gp_file in data_dir.glob("*.gp*"):
        if gp_file.suffix in ['.gp3', '.gp4', '.gp5', '.gtp'] and not gp_file.name.endswith('.tokens.txt'):
            output_txt = output_dir / f"{gp_file.stem}.txt"
            process_gp_file(gp_file, output_txt)

if __name__ == "__main__":
    process_dataset(Path("datasets"), Path("processed_data"))
    print("Processamento concluído.")