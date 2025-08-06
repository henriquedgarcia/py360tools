from pathlib import Path
from typing import Union

import cv2


class ReadVideo:
    def __init__(self, video_path: Path, gray=True, dtype='float64'):
        self.cap = cv2.VideoCapture(f'{video_path}')
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        self.gray = gray
        self.dtype = dtype

    def __next__(self):
        """

        :return:
        """
        ok, frame = self.cap.read()
        if not ok:
            raise StopIteration

        if self.gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame.astype(self.dtype)

    def __iter__(self):
        self.reset() # Ensure we start from the beginning when a new iteration begins
        return self

    def reset(self):
        """
        Resets the video capture's frame position to the beginning.

        This function sets the frame position of the video capture object back
        to the first frame, allowing the video to be replayed from the start.

        :return: None
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def set_position(self, position: Union[int, float]):
        """
        Sets the position of the video frame for playback.

        This method adjusts the current frame position in the video stream to
        the specified value using OpenCV.

        :param position: The target frame position to set. Can be an integer or
            a float value, representing the exact position in the video.
        :return: None
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
