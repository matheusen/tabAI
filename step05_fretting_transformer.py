# -*- coding: utf-8 -*-
import argparse, json, re
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ---------- utils ----------
def load_text(path: Path):
    return path.read_text().strip()

# ---------- Simple TAB Generator (rule-based) ----------
def midi_to_guitar_tab(pitch, tuning="EADGBE", capo=0):
    """
    Mapeia pitch MIDI para posição de guitarra (string, fret).
    Tuning: EADGBE (padrão), capo ajusta pitches.
    Retorna (string_idx, fret) ou None se impossível.
    Strings: 0=E (low), 1=A, 2=D, 3=G, 4=B, 5=E (high)
    """
    open_notes = {
        "EADGBE": [40, 45, 50, 55, 59, 64],  # E2, A2, D3, G3, B3, E4
    }
    if tuning not in open_notes:
        raise ValueError(f"Tuning {tuning} não suportado.")
    
    pitches = open_notes[tuning]
    # Ajusta capo (capo em fret 0 aumenta pitches)
    pitches = [p + capo for p in pitches]
    
    # Para cada string, calcula fret = pitch - open_note
    # Escolhe a opção com fret mais baixo (preferência por cordas baixas se empate)
    best = None
    for string_idx, open_pitch in enumerate(pitches):
        fret = pitch - open_pitch
        if 0 <= fret <= 24:  # frets típicos
            if best is None or fret < best[1] or (fret == best[1] and string_idx < best[0]):
                best = (string_idx, fret)
    return best

def generate_tab_from_events(events_txt, tuning="EADGBE", capo=0):
    """
    Gera TAB simples a partir de eventos MIDI.
    Parseia NOTE_ON<pitch>, gera TAB<string,fret> para cada nota.
    Ignora TIME_SHIFT e NOTE_OFF para simplicidade.
    """
    # Parse events: extrai pitches de NOTE_ON
    pitches = []
    for token in events_txt.split():
        match = re.match(r"NOTE_ON<(\d+)>", token)
        if match:
            pitch = int(match.group(1))
            pitches.append(pitch)
    
    # Gera TAB para cada pitch
    tab_tokens = []
    for pitch in pitches:
        pos = midi_to_guitar_tab(pitch, tuning, capo)
        if pos:
            string, fret = pos
            tab_tokens.append(f"TAB<{string},{fret}>")
        else:
            tab_tokens.append(f"TAB<UNK>")  # impossível
    
    return " ".join(tab_tokens)

# ---------- Fretting Transformer hook ----------
def run_fretting_transformer(events_txt: str, tuning: str = "EADGBE", capo: int = 0, model_path: str = "./fretting_transformer"):
    """
    Implementação do Fretting-Transformer (T5 encoder-decoder).
    Toma tokens MIDI + condições (TUNING, CAPO) e gera tokens TAB.
    NOTA: Modelo Fretting-Transformer não está disponível publicamente.
    Usa gerador rule-based simples como alternativa prática.
    Retorna string de tokens TAB.
    """
    if model_path and Path(model_path).exists():
        # Carregar modelo treinado
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        input_text = f"translate MIDI to TAB: TUNING<{tuning}> CAPO<{capo}> {events_txt}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
        tab_tokens = tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        # Usar gerador rule-based simples
        print("AVISO: Usando gerador rule-based simples (não T5). Para modelo real, forneça model_path ou treine com train_pipeline.py.")
        tab_tokens = generate_tab_from_events(events_txt, tuning, capo)
    
    return tab_tokens

def main():
    ap = argparse.ArgumentParser(description="Step05 — MIDI→TAB (Fretting-Transformer T5).")
    ap.add_argument("--events_txt", required=True, help="Arquivo events.txt do Step04.")
    ap.add_argument("--tuning", default="EADGBE", help="Afinação da guitarra (ex: EADGBE).")
    ap.add_argument("--capo", type=int, default=0, help="Posição do capo.")
    ap.add_argument("--model_path", help="Caminho para modelo T5 treinado (opcional, usa t5-base se não informado).")
    ap.add_argument("--out_tab_dir", default="work/tab", help="Pasta para tab.txt")
    args = ap.parse_args()

    events_txt = load_text(Path(args.events_txt))
    tab_dir = Path(args.out_tab_dir); tab_dir.mkdir(parents=True, exist_ok=True)
    tab_txt = tab_dir / "tab.txt"

    # Gera TAB tokens
    tab_tokens = run_fretting_transformer(events_txt, tuning=args.tuning, capo=args.capo, model_path=args.model_path)

    tab_txt.write_text(tab_tokens)
    print("[OK] tab:", tab_txt)
    print("Tokens TAB gerados (placeholder com T5-base):", tab_tokens[:200] + "..." if len(tab_tokens) > 200 else tab_tokens)

if __name__ == "__main__":
    main()