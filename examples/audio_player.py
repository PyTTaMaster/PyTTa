#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pytta


def parseArgs(args):
    file = None
    for arg in args:
        if arg.split('.')[-1] in ['wav', 'WAV', 'Wav']:
            file = arg
    return file


class AudioPlayer(object):
    """
    Example class for simple audio player app based on PyTTa
    """

    def __init__(self, fileName=None):
        """
        fileName is a str with the absolute or relative path to a WAVE file
        """
        print("Welcome to MyAudioPlayer!")
        print("To quit the program use the command:\n -exit")

        self.load(fileName)
        self.commandList = ['-load', '-play', '-pause', '-stop', '-exit']
        return

    def load(self, fileName=None):
        if fileName is None:
            print("Please, insert a file name: ")
            fileName = input()
        if fileName == '-exit':
            self.exit()
        self.file = fileName
        self.audio = pytta.read_wav(self.file)
        self.streaming = pytta.generate.stream('O', excitation=self.audio)
        print("Opened file", self.file)
        print("Available commands are:\n", "-play;\n", "-pause;\n", "-stop.")
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

    def exit(self):
        sys.exit()
        return

    def exec_(self):
        """
        Application context for continuous read of command line arguments to
        control the reproduction of the audio file
        """

        # It goes on, and on, and on, and on, and on, and on, ..., and on, ...
        while True:
            if self.file is None:
                self.load()

            # sys.stdin is a file-like object with the command line inputs
            command = input()

            # check if the read command can be used by the application
            if command in self.commandList:
                eval('self.' + command[1:] + '()')

            # or if it is ignored
            else:
                print("Unknown command", command, "\nSkipping.")
                pass

        # untill program closure
        return


if __name__ == "__main__":
    """
    This IF statement guarantees that the execution of the file will only ocurr
    when explicitly told so, e.g.:

        ~ $ python audio_player.py mywavefile.wav

    """
    file = parseArgs(sys.argv[:])
    player = AudioPlayer(file)
    sys.exit(player.exec_())
