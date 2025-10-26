from confluent_kafka import Producer
import json

# Configurar Kafka Producer
producer_config = {
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'pipeline_producer'
}
producer = Producer(producer_config)

# Exemplo de eventos (substitua pelo real)
events = "exemplo de eventos MIDI ou texto"

# Enviar para tópico
producer.produce('tab_generation_input', json.dumps({'events': events}).encode('utf-8'))
producer.flush()
print("Eventos enviados para geração de tab.")