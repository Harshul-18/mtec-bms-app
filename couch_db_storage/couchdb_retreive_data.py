import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd

def retrieve_data(uri, db_name, collection_names, save=False):
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client[db_name]
    all_docs = []

    for collection_name in collection_names:
        collection = db[collection_name]
        docs = collection.find({})
        model_docs = []
        for doc in docs:
            doc.pop('_id', None)
            model_docs.append(doc)
            all_docs.append(doc)
        model_data = pd.DataFrame(model_docs)
        if save:
            model_data.to_csv(f'all_{"_".join(collection_name.lower().split())}_data.csv', index=False)

    all_data = pd.DataFrame(all_docs)
    if save:
        all_data.to_csv(f'all_{db_name}_data.csv', index=False)

    client.close()
    return all_data

if __name__ == "__main__":
    uri = "mongodb+srv://admin:root@mtec-cluster.subjmhs.mongodb.net/?retryWrites=true&w=majority&appName=Mtec-Cluster"
    data = retrieve_data(uri,
                         'vehicles',
                         ['model_1', 'model_2'],
                         save=True)
    print(data.head())
