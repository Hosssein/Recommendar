
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['test']
Recom_table = db.recom
def insert_table(recomList):
    if recomList is not None:
        Recom_table.insert_many(recomList)
