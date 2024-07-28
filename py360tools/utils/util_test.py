import pickle


def load_test_data(file_name, default_data):
    """
    if file_name exists, loads it and returns it.
    else, run func(**kwargs), save result in file_name as pickle and return


    :param file_name: The pickle file name.
    :type file_name: Path
    :param default_data:
    :return:
    """
    if file_name.exists():
        return pickle.loads(file_name.read_bytes())

    file_name.parent.mkdir(parents=True, exist_ok=True)
    file_name.write_bytes(pickle.dumps(default_data))
    return default_data


def create_test_default(path, func, *args, **kwargs):
    try:
        data = pickle.loads(path.read_bytes())
    except FileNotFoundError:
        data = func(*args, **kwargs)
        path.write_bytes(pickle.dumps(data))
    return data
