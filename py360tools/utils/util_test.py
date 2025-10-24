import pickle
from pathlib import Path
from typing import Callable


def load_test_data(file_name, default_data):
    """
    If file_name exists, loads it and returns it.
    Else, save default_data in file_name as pickle and
    return data


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


def create_test_default(path: Path, func: Callable, *args, **kwargs):
    """
    If 'path' exists, loads it and returns it.
    Else, run func with (*args, **kwargs) and save results in 'path' as pickle and
    return results


    :param path: The pickle file name.
    :param func: The function to run if 'path' doesn't exist.
    :return:
    """
    try:
        data = pickle.loads(path.read_bytes())
    except FileNotFoundError:
        data = func(*args, **kwargs)
        path.write_bytes(pickle.dumps(data))
    return data
