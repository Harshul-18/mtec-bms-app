import couchdb
import uuid
import datetime
import random
import time

def delete_all_documents(username, password, db_name):
    couchdb_url = f"http://{username}:{password}@localhost:5984/"
    couch = couchdb.Server(couchdb_url)

    if db_name in couch:
        db = couch[db_name]
    else:
        print(f"Database '{db_name}' does not exist.")
        return

    docs = [doc for doc in db]
    for doc_id in docs:
        doc = db[doc_id]
        db.delete(doc)

    print(f"All documents in database '{db_name}' have been deleted.")

if __name__ == "__main__":
    name = input("Enter the database name: ")
    delete_all_documents(username="admin",
                password="root",
                db_name=name)