import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import random
import datetime
import time

def generate_documents(num_docs, model='Model 1'):
    documents = []
    for i in range(num_docs):
        doc = {
            "model": model,
            "owner": f"Owner {random.randint(1, 100)}",
            "battery_health": random.randint(70, 100),
            "voltage": round(random.uniform(11.0, 14.8), 2),
            "current": round(random.uniform(10.0, 50.0), 2),
            "temperature": round(random.uniform(20.0, 40.0), 2),
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
        documents.append(doc)
    return documents

def generate_models(num_docs, model):
    documents = []
    for i in range(num_docs):
        doc = {
            "model": model,
        }
        documents.append(doc)
    return documents

def save_documents(uri, db_name, collection_name, documents):
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client[db_name]
    collection = db[collection_name]

    # Insert documents into the collection
    result = collection.insert_many(documents)
    print(f"{len(result.inserted_ids)} documents saved successfully.")

    client.close()

if __name__ == "__main__":
    uri = "mongodb+srv://admin:root@mtec-cluster.subjmhs.mongodb.net/?retryWrites=true&w=majority&appName=Mtec-Cluster"
    model = int(input("Enter the model number: "))
    num_docs = 1

    # Save model document
    model_documents = generate_models(num_docs, model=f'Model {model}')
    save_documents(uri, 'vehicles', 'models', model_documents)

    # Continuously generate and save documents
    while True:
        documents = generate_documents(num_docs, model=f'Model {model}')
        save_documents(uri, 'vehicles', f'model_{model}', documents)
        time.sleep(1)