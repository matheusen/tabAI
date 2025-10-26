from confluent_kafka import Consumer, Producer, KafkaError
import json
import subprocess
import os

# Configurar Kafka Consumer
consumer_config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'pipeline_group',
    'auto.offset.reset': 'earliest'
}
consumer = Consumer(consumer_config)
consumer.subscribe(['tab_generation_input'])

# Producer para output
producer_config = {
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'pipeline_output_producer'
}
producer = Producer(producer_config)

def generate_tab(events):
    # Escrever events em arquivo temporário
    with open('temp_events.txt', 'w') as f:
        f.write(events)
    
    # Executar step05_fretting_transformer.py
    result = subprocess.run([
        'python', 'step05_fretting_transformer.py',
        '--events_txt', 'temp_events.txt',
        '--out_tab_dir', 'temp_tab'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        # Ler tab gerada (assumir arquivo em temp_tab)
        tab_files = os.listdir('temp_tab')
        if tab_files:
            with open(f'temp_tab/{tab_files[0]}', 'r') as f:
                tab_content = f.read()
            return tab_content
    return None

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
        data = json.loads(msg.value().decode('utf-8'))
        events = data['events']
        tab = generate_tab(events)
        if tab:
            producer.produce('tab_generation_output', json.dumps({'tab': tab}).encode('utf-8'))
            producer.flush()
            print("Tab gerada e enviada.")
        else:
            print("Falha ao gerar tab.")
finally:
    consumer.close()
    producer.flush()