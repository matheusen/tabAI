# -*- coding: utf-8 -*-
"""
Script para scraping de tabs Guitar Pro do Ultimate Guitar e Cifra Club, com IA para seleção inteligente de artistas.
ATENÇÃO: Scraping pode violar termos de serviço. Use com responsabilidade e baixe manualmente se possível.
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import requests
from pathlib import Path
import os
import openai
import numpy as np
from dotenv import load_dotenv
import base64
from confluent_kafka import Producer

# Configurar Selenium (assumir ChromeDriver instalado)
options = Options()
options.add_argument("--headless")  # Sem interface
driver = webdriver.Chrome(options=options)

# Carregar variáveis de ambiente do .env
load_dotenv()

# Configurar OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configurar Kafka
KAFKA_CONFIG = {
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'scraper_producer'
}
producer = Producer(KAFKA_CONFIG)

def get_embedding(text):
    """
    Obtém embedding do texto usando OpenAI.
    """
    if not openai.api_key:
        return None
    try:
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Erro ao obter embedding: {e}")
        return None

def cosine_similarity(a, b):
    """
    Calcula similaridade cosseno entre dois vetores.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_vision_advice(screenshot_path, prompt):
    """
    Usa GPT-4 Vision para analisar uma screenshot e dar conselhos.
    """
    if not openai.api_key:
        return None
    try:
        with open(screenshot_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        response = openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Erro ao usar GPT Vision: {e}")
        return None

def get_artists_from_ai(num_artists=10, genre="rock"):
    """
    Usa GPT para gerar uma lista de artistas com tabs GP disponíveis.
    """
    if not openai.api_key:
        print("OPENAI_API_KEY não definida. Defina a variável de ambiente.")
        return []
    
    prompt = f"Liste {num_artists} bandas ou artistas de {genre} famosos que têm tablaturas Guitar Pro disponíveis online. Foque em artistas clássicos e populares. Liste apenas os nomes, um por linha, sem numeração."
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        artists = response.choices[0].message.content.strip().split('\n')
        return [a.strip() for a in artists if a.strip()]
    except Exception as e:
        print(f"Erro ao chamar OpenAI: {e}")
        return []

def search_tabs(artist, num_pages=1):
    """
    Busca tabs de um artista no UG.
    """
    tabs = []
    for page in range(1, num_pages + 1):
        url = f"https://www.ultimate-guitar.com/search.php?search_type=title&value={artist}&page={page}"
        print(f"Buscando em: {url}")
        driver.get(url)
        time.sleep(3)  # Esperar load
        # Pegar todos links /tab/
        links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/tab/']")
        print(f"Encontrados {len(links)} links na página {page}")
        for link in links[:50]:  # Aumentar limite
            href = link.get_attribute("href")
            text = link.text.lower()
            title = link.get_attribute("title") or ""
            print(f"Link: {href} | Texto: {text} | Title: {title}")
            # Filtrar por presença de "guitar pro" no texto, title ou href
            if "guitar pro" in text or "gp" in text or "guitar pro" in title.lower() or "gp" in title.lower() or ".gp" in href or "/guitar-pro/" in href:
                tabs.append(href)
    print(f"Total tabs GP encontradas: {len(tabs)}")
    return tabs

def search_tabs_cifraclub(artist, num_pages=1):
    """
    Busca tabs no Cifra Club.
    """
    tabs = []
    artist_slug = artist.replace(' ', '-').lower()
    url = f"https://www.cifraclub.com.br/{artist_slug}/"
    print(f"Buscando em: {url}")
    driver.get(url)
    time.sleep(3)
    # Pegar links de tablaturas GP
    links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/guitarpro/']")
    print(f"Encontrados {len(links)} links")
    for link in links[:50]:
        href = link.get_attribute("href")
        if href and '/guitarpro/' in href and 'cifraclub.com.br' in href:
            tabs.append(href)
    print(f"Total tabs GP encontradas: {len(tabs)}")
    return tabs

def search_recent_cifraclub(num_pages=5):
    """
    Busca tabs GP mais recentes no Cifra Club.
    """
    tabs = []
    for page in range(1, num_pages + 1):
        if page == 1:
            url = "https://www.cifraclub.com.br/guitar-pro/"
        else:
            url = f"https://www.cifraclub.com.br/guitar-pro/{page}/"
        print(f"Buscando em: {url}")
        driver.get(url)
        time.sleep(5)  # Aumentar tempo para load
        # Esperar por elementos
        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "a")))
        except:
            print("Timeout waiting for page load")
        # Scroll para carregar mais conteúdo se necessário
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        # Tentar clicar em "Carregar mais" se existir
        try:
            load_more = driver.find_element(By.XPATH, "//button[contains(text(), 'Carregar mais') or contains(text(), 'Load more') or contains(@class, 'load-more')]")
            load_more.click()
            time.sleep(3)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        except:
            pass
        # Pegar links de tablaturas GP
        links = driver.find_elements(By.CSS_SELECTOR, "a[href*='guitarpro']")
        print(f"Encontrados {len(links)} links na página {page}")
        for link in links[:5]:  # Debug: print first 5
            href = link.get_attribute("href")
            print(f"Link encontrado: {href}")
        for link in links:
            href = link.get_attribute("href")
            if href and 'guitarpro' in href and 'cifraclub.com.br' in href:
                tabs.append(href)
    print(f"Total tabs GP encontradas: {len(tabs)}")
    return tabs

def download_tab_cifraclub(tab_url, output_dir):
    """
    Baixa GP file do Cifra Club, usando visão se necessário.
    """
    print(f"Tentando baixar de {tab_url}")
    driver.get(tab_url)
    time.sleep(3)
    try:
        # Procurar link de download GP
        download_link = driver.find_element(By.XPATH, "//a[contains(@href, '.gp') or contains(text(), 'Download') or contains(@href, 'download')]")
        gp_url = download_link.get_attribute("href")
        if gp_url:
            print(f"Encontrado link GP: {gp_url}")
            response = requests.get(gp_url)
            # Limpar filename
            filename = gp_url.split("arq=")[-1] if "arq=" in gp_url else gp_url.split("/")[-1]
            filename = filename.replace("?", "").replace("&", "").replace("=", "")
            if not filename.endswith(('.gp3', '.gp4', '.gp5', '.gtp')):
                filename += ".gp3"  # default
            with open(output_dir / filename, 'wb') as f:
                f.write(response.content)
            print(f"Baixado: {filename}")
        else:
            print(f"Link de download não encontrado em {tab_url}")
    except Exception as e:
        print(f"Erro ao baixar {tab_url}: {e}")
        # Usar GPT Vision como fallback
        screenshot_path = 'temp_screenshot.png'
        driver.save_screenshot(screenshot_path)
        prompt = "Esta é uma página da web do Cifra Club com uma tablatura de guitarra. Encontre o botão ou link de download para o arquivo Guitar Pro (.gp). Descreva o elemento a clicar ou forneça a URL de download se possível."
        advice = get_vision_advice(screenshot_path, prompt)
        print(f"Conselho do GPT Vision: {advice}")
        if advice and 'http' in advice:
            # Tentar extrair URL
            import re
            urls = re.findall(r'http[s]?://[^\s]+', advice)
            if urls:
                gp_url = urls[0]
                print(f"Tentando baixar via Vision: {gp_url}")
                try:
                    response = requests.get(gp_url)
                    filename = gp_url.split("/")[-1] or "tab_vision.gp5"
                    with open(output_dir / filename, 'wb') as f:
                        f.write(response.content)
                    print(f"Baixado via Vision: {filename}")
                except Exception as e2:
                    print(f"Erro ao baixar via Vision: {e2}")
            else:
                print("Nenhuma URL encontrada no conselho.")
        else:
            print("Conselho não útil.")
        # Limpar screenshot
        if os.path.exists(screenshot_path):
            os.remove(screenshot_path)

def download_gp(tab_url, output_dir):
    """
    Vai na página da tab e baixa o GP file.
    """
    driver.get(tab_url)
    time.sleep(3)  # Esperar mais
    try:
        # Procurar por links ou botões de download GP
        elements = driver.find_elements(By.XPATH, "//a[contains(@href, '.gp') or contains(text(), 'Download') or contains(@class, 'download')]")
        gp_url = None
        for elem in elements:
            href = elem.get_attribute("href")
            text = elem.text.lower()
            if href and ('.gp' in href or 'guitarpro' in href or 'download' in text):
                gp_url = href
                break
        
        # Se não encontrou, tentar clicar em botão de download e ver se abre modal
        if not gp_url:
            try:
                download_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Download') or contains(@class, 'download')]")
                download_btn.click()
                time.sleep(2)
                # Agora procurar novamente
                elements = driver.find_elements(By.XPATH, "//a[contains(@href, '.gp')]")
                for elem in elements:
                    href = elem.get_attribute("href")
                    if href and '.gp' in href:
                        gp_url = href
                        break
            except:
                pass
        
        if gp_url:
            response = requests.get(gp_url)
            filename = gp_url.split("/")[-1] or f"tab_{tab_url.split('/')[-1]}.gp5"
            with open(output_dir / filename, 'wb') as f:
                f.write(response.content)
            print(f"Baixado: {filename}")
        else:
            print(f"Link GP não encontrado em {tab_url}. Verificando se há GP na página...")
            # Verificar se a página menciona GP
            page_text = driver.page_source.lower()
            if "guitar pro" in page_text or ".gp" in page_text:
                print("Página parece ter GP, mas link não encontrado.")
            else:
                print("Página não parece ter GP.")
    except Exception as e:
        print(f"Erro ao baixar {tab_url}: {e}")

# Dicionário de artistas para testar scraping
ARTISTS_TEST = {
    "pink_floyd": "pink floyd",
    "metallica": "metallica",
    "queen": "queen",
    "led_zeppelin": "led zeppelin",
    "the_beatles": "the beatles",
}

def main(site="ug", artist_key="pink_floyd", num_tabs=5, output_dir="datasets/gpfiles"):
    """
    Principal: busca e baixa tabs de um artista.
    site: 'ug' para Ultimate Guitar, 'cifraclub' para Cifra Club, 'cifraclub_recent' para tabs recentes GP, 'ai_cifraclub' para scraping inteligente com IA.
    """
    if site == "ug":
        if artist_key not in ARTISTS_TEST:
            print(f"Artista {artist_key} não encontrado. Use: {list(ARTISTS_TEST.keys())}")
            return
        
        artist = ARTISTS_TEST[artist_key]
        print(f"Baixando tabs de {artist} no Ultimate Guitar...")
        Path(output_dir).mkdir(exist_ok=True)
        tabs = search_tabs(artist, num_pages=2)
        print(f"Encontrados {len(tabs)} links de tabs GP.")
        for tab in tabs[:num_tabs]:
            download_gp(tab, Path(output_dir))
    elif site == "cifraclub":
        artist = artist_key.replace('_', ' ')
        print(f"Baixando tabs de {artist} no Cifra Club...")
        Path(output_dir).mkdir(exist_ok=True)
        tabs = search_tabs_cifraclub(artist, num_pages=2)
        print(f"Encontrados {len(tabs)} links de tabs.")
        for tab in tabs[:num_tabs]:
            download_tab_cifraclub(tab, Path(output_dir))
    elif site == "cifraclub_recent":
        print("Baixando tabs GP mais recentes no Cifra Club...")
        Path(output_dir).mkdir(exist_ok=True)
        tabs = search_recent_cifraclub(num_pages=10)  # Mais páginas para mais tabs
        print(f"Encontrados {len(tabs)} links de tabs recentes.")
        for tab in tabs:  # Baixar todas
            download_tab_cifraclub(tab, Path(output_dir))
    elif site == "ai_cifraclub":
        print("Usando IA para selecionar artistas e filtrar tabs GP relevantes do Cifra Club...")
        Path(output_dir).mkdir(exist_ok=True)
        artists = get_artists_from_ai(num_artists=10)
        print(f"Artistas selecionados pela IA: {artists}")
        ideal_emb = get_embedding("guitar pro tablature for rock band song electric guitar")
        if not ideal_emb:
            print("Não foi possível obter embedding ideal. Abortando.")
            return
        for artist in artists:
            print(f"Processando {artist}...")
            tabs = search_tabs_cifraclub(artist, num_pages=2)
            print(f"Encontrados {len(tabs)} links para {artist}.")
            filtered_tabs = []
            for tab in tabs:
                driver.get(tab)
                time.sleep(2)
                title = driver.title
                if title:
                    emb = get_embedding(title)
                    if emb:
                        sim = cosine_similarity(ideal_emb, emb)
                        print(f"Título: {title} | Similaridade: {sim:.2f}")
                        if sim > 0.8:
                            filtered_tabs.append(tab)
            print(f"Tabs filtradas para {artist}: {len(filtered_tabs)}")
            for tab in filtered_tabs[:num_tabs]:
                producer.produce('gp_download_links', tab.encode('utf-8'))
                producer.flush()
                print(f"Enviado para download: {tab}")
    else:
        print("Site não suportado. Use 'ug', 'cifraclub', 'cifraclub_recent' ou 'ai_cifraclub'.")
    
    driver.quit()
    print("Scraping concluído.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        site = sys.argv[1]
        artist_key = sys.argv[2]
    elif len(sys.argv) == 2:
        site = sys.argv[1]
        artist_key = "pink_floyd" if site != "cifraclub_recent" and site != "ai_cifraclub" else None
    else:
        site = "ai_cifraclub"
        artist_key = None
    main(site, artist_key)