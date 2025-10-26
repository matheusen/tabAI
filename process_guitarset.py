# -*- coding: utf-8 -*-
"""
Processar GuitarSet: Converter anotações JAMS para tokens MIDI e TAB.
Baseado no formato do Fretting-Transformer.
"""

import jams
from pathlib import Path
import json

def jams_to_midi_tokens(jams_file):
    """
    Converte anotações JAMS para tokens MIDI: NOTE_ON<pitch> TIME_SHIFT<dt> NOTE_OFF<pitch>
    """
    jam = jams.load(str(jams_file))
    notes = []
    for ann in jam.annotations:
        if ann.namespace == 'note':
            for note in ann.data:
                pitch = int(note.value)
                onset = note.time
                duration = note.duration
                offset = onset + duration
                notes.append((onset, offset, pitch))
    
    # Ordenar por onset
    notes.sort(key=lambda x: x[0])
    tokens = []
    cur_time = 0
    for onset, offset, pitch in notes:
        dt = int((onset - cur_time) * 1000)  # ms
        if dt > 0:
            tokens.append(f"TIME_SHIFT<{dt}>")
        tokens.append(f"NOTE_ON<{pitch}>")
        dt_off = int((offset - onset) * 1000)
        if dt_off > 0:
            tokens.append(f"TIME_SHIFT<{dt_off}>")
        tokens.append(f"NOTE_OFF<{pitch}>")
        cur_time = offset
    return " ".join(tokens)

def generate_tab_from_notes(notes, tuning="EADGBE"):
    """
    Gera TAB tokens simples a partir de notas (usando rule-based).
    """
    from step05_fretting_transformer import midi_to_guitar_tab  # Importar função
    tab_tokens = []
    for onset, offset, pitch in notes:
        pos = midi_to_guitar_tab(pitch, tuning)
        if pos:
            string, fret = pos
            tab_tokens.append(f"TAB<{string},{fret}>")
    return " ".join(tab_tokens)

def process_guitarset(data_dir, output_dir):
    """
    Processa GuitarSet: JAMS -> tokens MIDI e TAB.
    """
    output_dir.mkdir(exist_ok=True)
    for jams_file in data_dir.glob("**/*.jams"):
        midi_tokens = jams_to_midi_tokens(jams_file)
        # Carregar notes para TAB
        jam = jams.load(str(jams_file))
        notes = []
        for ann in jam.annotations:
            if ann.namespace == 'note':
                for note in ann.data:
                    notes.append((note.time, note.time + note.duration, int(note.value)))
        tab_tokens = generate_tab_from_notes(notes)
        
        # Salvar
        out_file = output_dir / f"{jams_file.stem}.txt"
        with open(out_file, 'w') as f:
            f.write(f"INPUT: {midi_tokens}\nOUTPUT: {tab_tokens}\n")
        print(f"Processado: {jams_file} -> {out_file}")

if __name__ == "__main__":
    # Assumir GuitarSet em datasets/guitarset/
    process_guitarset(Path("datasets/guitarset"), Path("processed_data"))
    print("Processamento GuitarSet concluído.")