#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pytta


class AudioPlayer(object):
    """
    Example class for simple audio player app based on PyTTa
    """
    def __init__(self, fileName):
        """
        fileName is a str with the absolute or relative path to a WAVE file
        """
        self.audio = pytta.read_wav(fileName)
        self.streaming = pytta.generate.stream('O', excitation=self.audio)
        return

    def play(self):
        """
        Start playback of the wave file
        """
        self.streaming.start()
        return

    def pause(self):
        """
        Stop playback of the wave file
        """
        self.streaming.stop()
        return

    def stop(self):
        """
        Start playback of the wave file and move the audio to the beggining
        """
        self.streaming.stop()
        self.kn = 0
        return

    def exec_(self):
        """
        Application context for continuous read of command line arguments to
        control the reproduction of the audio file
        """
        # It goes on, and on, and on, and on, and on, and on, ..., and on, ...
        while True:
            # sys.stdin is a file-like object with the command line inputs
            command = sys.stdin.readline()

            # check if the read command can be used by the application
            if command in ['stop', 'play', 'pause']:
                eval('self.' + command)

            # or if it ends it
            elif command == 'exit':
                break

            # or if it is ignored
            else:
                pass

        # untill program closure
        return sys.exit()


if __name__ == "__main__":
    """
    This IF statement guarantees that the execution of the file will only ocurr
    when explicitly told so, e.g.:

        ~ $ python audio_player.py 'mywavefile.wav'

    """

    player = AudioPlayer(sys.argv[1])
    sys.exit(player.exec_())
