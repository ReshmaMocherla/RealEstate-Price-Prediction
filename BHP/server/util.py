import json
import pickle
import numpy as np

__locations  = None
__data_column = None
__model = None

def get_estimated_price(location,total_sqft,bhk,bath):
    try:
        loc_index = __data_column.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_column))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)


def get_location_names():
    return __locations

def load_saved_artifacts():
    print("Loading saved artifacts...start")
    global __data_column
    global __locations

    with open("./artifacts/Columns.json",'r') as f:
        __data_column = json.load(f)['data_column']
        __locations = __data_column[3:]

    global __model
    with open("./artifacts/banglore_home_prices_model.pickle",'rb') as f:
        __model = pickle.load(f)
    print("Loading saved artifacts...done")




if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000,3,3))
    print(get_estimated_price('1st Phase JP Nagar',1000,2,2))
    print(get_estimated_price('Kalhali',1000,2,2))#other location
    print(get_estimated_price('Ejipura',1000,2,2))#other location