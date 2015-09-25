import pickle


def save_to_file(data, file_path):
    file = open(file_path, 'wb')
    pickle.dump(data, file)
    file.close()


def get_trained_data(file_path):
    file = open(file_path, 'rb')
    data = pickle.load(file)
    file.close()
    return data
