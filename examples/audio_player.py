#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytta
import soundfile as sf


def parseArgs(args):
    file = None
    for arg in args:
        if arg.split('.')[-1] in sf.available_formats().keys():
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

        self.executing = True
        self.load_(fileName)
        self.commandList = ['-load', '-play', '-pause', '-stop', '-exit']
        return

    def load_(self, fileName=None):
        if fileName is None:
            print("Please, insert a file name: ")
            fileName = input()
        if fileName == '-exit':
            self.exit_()
            return
        self.file = sf.SoundFile(fileName)
        self.audio = pytta.SignalObj(self.file.read(), 'time',
                                     self.file.samplerate)
        self.streaming = pytta.generate.stream('O', excitation=self.audio)
        print("Opened file", self.file.name)
        print("Available commands are:\n", "-play;\n", "-pause;\n", "-stop.")
        return

    def play_(self):
        """
        Start playback of the wave file
        """
        self.streaming.start()
        return

    def pause_(self):
        """
        Stop playback of the wave file
        """
        self.streaming.stop()
        return

    def stop_(self):
        """
        Start playback of the wave file and move the audio to the beggining
        """
        self.streaming.stop()
        self.kn = 0
        return

    def exit_(self):
        try:
            self.stream.close()
        except AttributeError:
            pass
        self.executing = False
        return

    def bye_(self):
        print('Bye, bye!')
        return

    def exec_(self):
        """
        Application context for continuous read of command line arguments to
        control the reproduction of the audio file
        """
        if not self.executing:
            self.bye_()
            return
        # It goes on, and on, and on, and on, and on, and on, ..., and on, ...
        while self.executing:

            try:
                # check if the command can be used by the application
                if self.command in self.commandList:

                    # True: evaluates it as a function
                    eval('self.' + self.command[1:] + '_()')

                # False: it is ignored
                else:
                    print("Unknown command", self.command, "\nSkipping.")

                # read command from command line
                self.command = input()

            except AttributeError:
                # read command from command line
                self.command = input()

            finally:
                self.bye_()
        return


if __name__ == "__main__":
    """
    This IF statement guarantees that the execution of the file will only ocurr
    when explicitly told so, e.g.:

        ~ $ python audio_player.py mywavefile.wav

    """
    file = None
    player = AudioPlayer(file)
    player.exec_()
