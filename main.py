import pandas as pd
import numpy as np
import json
import csv
import pickle

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
import os.path
from scipy import spatial
from DB import insert_table
from scipy.sparse.linalg import svds


class Recom():
    ratings = None
    items = None
    num_users = 0
    num_items = 0
    num_colors = 0
    user_item =[]
    list_of_color=[]


    def __init__(self):
        self.list_of_items = []
        self.list_of_users = []
        self.__loadItems()
        self.__loadRating()
        self.__create_item_mat()


    def __loadItems(self, fileName="items.json"):
        with open(fileName) as f:
            Recom.items = pd.DataFrame(json.loads(line) for line in f)

        Recom.num_items = Recom.items.id.unique().shape[0]
        Recom.num_colors = Recom.items.color.unique().shape[0]


    def __loadRating(self, fileName="ratings.csv"):
        Recom.ratings = pd.read_csv(fileName,
                                    names=['user_id', 'item_id', 'rating'],
                                    skiprows=1,
                                    dtype={'user_id' : np.int32, 'item_id' : np.int32, 'rating' : np.int32})

        Recom.num_users = Recom.ratings.user_id.unique().shape[0]

    def create_user_item(self, file_path="matrix.pkl"):

        lst = Recom.ratings['user_id']
        for x in lst:
            if x not in self.list_of_users:
                self.list_of_users.append(x)

        if os.path.exists(file_path):
            print("user item file exist!")
            pickle_in = open("matrix.pkl", "rb")
            Recom.user_item = pickle.load(pickle_in)

        else:
            print(self.list_of_items,self.list_of_users)

            with open('ratings.csv') as csvDataFile:
                csvReader = csv.reader(csvDataFile)
                Recom.user_item = np.zeros((Recom.num_users, 1000))
                next(csvReader)
                for x in csvReader:
                    Recom.user_item[self.list_of_users.index(int(x[0])), self.list_of_items.index(int(x[1]))] = int(x[2])
            pickle_out = open("matrix.pkl", "wb")
            pickle.dump(Recom.user_item, pickle_out)
            pickle_out.close()

    def __create_item_mat(self):
        itm = Recom.ratings['item_id']
        #list_of_items = []
        for x in itm:
            if x not in self.list_of_items:
                self.list_of_items.append(x)

        Recom.list_of_color = list(Recom.items.color.unique())

        self.item_mat = np.zeros((Recom.num_items, 3))
        for itm in range(len(Recom.items.index)):
            self.item_mat[self.list_of_items.index(Recom.items.iloc[itm]['id']), 0] = Recom.items.iloc[itm]['price']
            self.item_mat[self.list_of_items.index(Recom.items.iloc[itm]['id']), 1] = Recom.items.iloc[itm]['category']
            self.item_mat[self.list_of_items.index(Recom.items.iloc[itm]['id']), 2] = Recom.list_of_color.index(Recom.items.iloc[itm]['color'])

    def prediction(self, data_mat, sim_matrix, type='user_based'):
        if type == 'user_based':
            mean = data_mat.mean(axis=1)
            diff_2_mean = (data_mat - mean[:, np.newaxis])
            return (sim_matrix.dot(diff_2_mean)+mean[:, np.newaxis]) / np.array([np.abs(sim_matrix).sum(axis=1)]).T
        elif type == 'item_based':
            return data_mat.dot(sim_matrix) / np.array([np.abs(sim_matrix).sum(axis=1)])




if __name__ == '__main__':
    r = Recom()
    r.create_user_item()


    train_data, test_data = train_test_split(Recom.user_item, test_size=0.2)
    item_similarity = pairwise_distances(train_data.T, metric='cosine')
    user_similarity = pairwise_distances(train_data, metric='cosine')
    item_prediction = r.prediction(train_data, item_similarity, type='item')
    user_prediction = r.prediction(train_data, user_similarity, type='user')


    U, S, Vtrans = svds(train_data, k=20)
    diag = np.diag(S)
    prediction = np.dot(np.dot(U, diag), Vtrans)
    list_of_items = []
    with open("iteminput.json") as f:
        for line in f:
            jj = json.loads(line)
            list_of_items.append(int(jj["id"]))


    list_users = []
    with open("userinput.txt") as f:
        list_users = f.read().split()
    userlist = list(map(lambda x: r.list_of_users.index(int(x)), list_users) )
    itemlist = list(map(lambda x: r.list_of_items.index(int(x)), list_of_items))

    #filter prediction matrix
    filtered_mat = prediction[:,itemlist][userlist,:]

    #select proper item for each user
    predicted_items=[]
    for i,itm in enumerate(filtered_mat):
        itm_index = np.argmax(itm)

        recoms = {
            'user_id': list_users[i],
            'item_id': list_of_items[itm_index]
        }
        predicted_items.append(recoms)

    insert_table(predicted_items)

    # with open("iteminput.json") as f:
    #
    #     for line in f:
    #         jj = json.loads(line)
    #         input_vector = []
    #         input_vector.append(jj["price"])
    #         input_vector.append(jj["category"])
    #         input_vector.append(Recom.list_of_color.index(jj["color"]))
    #         sim_list = []
    #         max_val = 0.
    #         max_ind = -1
    #         for i,item in enumerate(r.item_mat):
    #
    #             tmp = 1 - spatial.distance.cosine(input_vector, item)
    #             #print(max_val,tmp)
    #             if max_val < tmp:
    #                 max_val = tmp
    #                 max_ind = r.list_of_items[i]
    #                 #print(max_ind,max_val)
    #             #sim_list.append(1 - spatial.distance.cosine(input_vector, x))
    #
    #         recoms = {
    #             'user_id': 3,
    #             'item_id': jj['id']
    #         }
    #         list_of_recoms.append(recoms)
    #
    # #print(list_of_recoms)

