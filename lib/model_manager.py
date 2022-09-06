"""
Functions for saving and loading trained model
"""
import pickle


def save_model(model):
    """
    Saves model to local directory
    """
    filename = 'trained_model/finalized_model.pkl'
    pickle.dump(model, open(filename, 'wb'))


def load_model():
    """
    Loads model into python from local directory
    """
    filename = 'trained_model/finalized_model.pkl'
    return pickle.load(open(filename, 'rb'))
