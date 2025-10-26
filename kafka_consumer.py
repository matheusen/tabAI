from confluent_kafka import Consumer, KafkaError
import requests
from pathlib import Path
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import base64
from dotenv import load_dotenv
import openai

# Carregar .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configurar Selenium
options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

# Configurar Kafka Consumer
consumer_config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'download_group',
    'auto.offset.reset': 'earliest'
}
consumer = Consumer(consumer_config)
consumer.subscribe(['gp_download_links'])

# Função para baixar
def download_tab(tab_url, output_dir):
    print(f"Tentando baixar de {tab_url}")
    driver.get(tab_url)
    time.sleep(3)
    try:
        download_link = driver.find_element(By.XPATH, "//a[contains(@href, '.gp') or contains(text(), 'Download') or contains(@href, 'download')]")
        gp_url = download_link.get_attribute("href")
        if gp_url:
            print(f"Encontrado link GP: {gp_url}")
            response = requests.get(gp_url)
            filename = gp_url.split("arq=")[-1] if "arq=" in gp_url else gp_url.split("/")[-1]
            filename = filename.replace("?", "").replace("&", "").replace("=", "")
            if not filename.endswith(('.gp3', '.gp4', '.gp5', '.gtp')):
                filename += ".gp3"
            with open(output_dir / filename, 'wb') as f:
                f.write(response.content)
            print(f"Baixado: {filename}")
        else:
            print(f"Link não encontrado em {tab_url}")
    except Exception as e:
        print(f"Erro ao baixar {tab_url}: {e}")
        # Fallback com visão
        screenshot_path = 'temp_screenshot.png'
        driver.save_screenshot(screenshot_path)
        prompt = "Esta é uma página da web do Cifra Club com uma tablatura de guitarra. Encontre o botão ou link de download para o arquivo Guitar Pro (.gp). Descreva o elemento a clicar ou forneça a URL de download se possível."
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
            advice = response.choices[0].message.content
            print(f"Conselho Vision: {advice}")
            if advice and 'http' in advice:
                import re
                urls = re.findall(r'http[s]?://[^\s]+', advice)
                if urls:
                    gp_url = urls[0]
                    response = requests.get(gp_url)
                    filename = gp_url.split("/")[-1] or "tab_vision.gp5"
                    with open(output_dir / filename, 'wb') as f:
                        f.write(response.content)
                    print(f"Baixado via Vision: {filename}")
        except Exception as e2:
            print(f"Erro Vision: {e2}")
        if os.path.exists(screenshot_path):
            os.remove(screenshot_path)

# Loop de consumo
output_dir = Path("datasets/gpfiles")
output_dir.mkdir(exist_ok=True)

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            else:
                print(f"Erro Kafka: {msg.error()}")
                break
        tab_url = msg.value().decode('utf-8')
        download_tab(tab_url, output_dir)
finally:
    consumer.close()
    driver.quit()