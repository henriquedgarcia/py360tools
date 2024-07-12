import numpy as np
import pandas as pd
from numpy.linalg import norm

from lib.transform.transform import ea2xyz


def position2displacement(df_positions):
    """
    Converts a position to a trajectory dataframe. The positions should have the 2 columns:
        - [0] yaw (float) in rads
        - [1] pitch (float) in rads

    The two columns of the dataframe will be converted into a numpy array.

    :param df_positions: Contains the position of the center of the viewport.
    :type df_positions: pd.DataFrame
    :return : the displacement of the center of viewport by frame in radians
    :rtype: pd.DataFrame
    """

    # convert to numpy
    positions = np.array(df_positions[['yaw', 'pitch']])
    pos = ea2xyz(ea=positions.T).T

    # Calculate angle displacement = arc_cos(dot(v1, v2))
    dot_product = [np.sum(pos[i] * pos[i + 1] / (norm(pos[i]) * norm(pos[i + 1]))) for i in range(len(pos) - 1)]
    inst_angle = np.arccos(dot_product)

    return pd.DataFrame(inst_angle, columns=['displacement'])
