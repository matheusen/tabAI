# -*- coding: utf-8 -*-
"""
Pipeline para criação do modelo Fretting-Transformer.
Baseado no artigo: Datasets DadaGP, GuitarToday, Leduc.
Passos: Baixar datasets, processar para tokens, treinar T5.
"""

import os, requests, zipfile, subprocess
from pathlib import Path

# ---------- Configurações ----------
DATA_DIR = Path("datasets")
DATA_DIR.mkdir(exist_ok=True)

# Datasets (fontes públicas similares, pois originais não são diretamente baixáveis)
DATASETS = {
    "dadaGP": {
        "url": None,  # Dataset privado; baixe manualmente de https://github.com/dada-bots/dadaGP (solicitar acesso)
        "desc": "DadaGP: 26k tablaturas GP. Baixe e coloque .gp files em datasets/dadaGP/. Use dadaGP/dadagp.py para tokenizar."
    },
    "guitarset": {
        "url": "https://zenodo.org/record/3371780/files/GuitarSet.tar.gz?download=1",
        "desc": "GuitarSet: Dataset público de guitarra com anotações (similar a DadaGP)."
    },
    "guitar_today_placeholder": {
        "url": None,  # Placeholder: GuitarToday é Patreon, usar exemplos públicos
        "desc": "GuitarToday: Usar dataset fingerstyle público alternativo."
    },
    "leduc_placeholder": {
        "url": None,  # Placeholder: Buscar transcrições jazz públicas
        "desc": "Leduc Dataset: Usar transcrições jazz de fontes abertas."
    }
}

def download_dataset(name, url, dest_dir):
    if url:
        print(f"Baixando {name}...")
        response = requests.get(url, stream=True)
        file_path = dest_dir / f"{name}.tar.gz"
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        # Tentar extrair como zip, se falhar, tentar tar.gz
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(dest_dir / name)
            print(f"{name} extraído como zip.")
        except zipfile.BadZipFile:
            # Assumir tar.gz
            import tarfile
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                tar_ref.extractall(dest_dir / name)
            print(f"{name} extraído como tar.gz.")
    else:
        print(f"{name}: {DATASETS[name]['desc']} - Baixe manualmente ou use alternativo.")

def main():
    print("Iniciando download dos datasets...")
    for name, info in DATASETS.items():
        download_dataset(name, info["url"], DATA_DIR)
    print("Downloads concluídos. Próximo: processamento de dados.")

if __name__ == "__main__":
    main()