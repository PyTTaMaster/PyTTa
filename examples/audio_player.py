#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pytta
import soundfile as sf


def parseArgs(arg):
    file = None
    if arg.split('.')[-1].upper() in sf.available_formats().keys():
        file = arg
    else:
        file = None
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
        self.commandList = ['-load', '-play', '-pause', '-stop', '-exit']
        self.load_(fileName)
        return

    def renew_audio(self):
        self.audio = pytta.SignalObj(self.file.read(), 'time', self.file.samplerate)
        self.file.close()
        return

    def reset_stream(self):
        try:
            self.streaming.close()  # tries to close stream obj to avoid PortAudioError
        except AttributeError:      # if it fails by AttributeError, means that the stream is not
            pass                    # instantiated yet, so it can pass by this step
        self.streaming = pytta.generate.stream('O', excitation=self.audio)
        self.newFileRead = False
        return

    def load_(self, fileName=None):
        if fileName is None or fileName == '':
            print("Please, insert a valid audio file name: ")
            fileName = input()
        if fileName == '-exit':
            self.exit_()
            return
        try:
            self.file = sf.SoundFile(fileName)
            self.newFileRead = True
            print("Opened file", self.file.name.split(os.sep)[-1])
            print("Available commands are:\n", *self.commandList[:-1])
        except RuntimeError:
            print("The file could not be opened!")
            self.load_()
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
        self.streaming.kn = 0
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

            if self.newFileRead:
                self.renew_audio()
                self.reset_stream()

            # TRY-except: TRY to run the following code:
            try:
                arg = ''
                comm = self.command.split(' ')
                if len(comm) > 1:
                    arg, comm = comm[1], comm[0]
                else:
                    comm = comm[0]

                # check if the command can be used by the application
                    # True: evaluates it as a function
                if comm[:] == '-load':
                    eval('self.' + comm[1:] + '_(' + 'arg' + ')')
                elif comm[:] in self.commandList[1:]:
                    eval('self.' + comm[1:] + '_()')

                # False: it is ignored
                else:
                    print("Unknown command", self.command, "\nSkipping.")

                if self.executing:
                    # read input from command line
                    self.command = input()

            # try-EXCEPT: EXCEPT if there's no attribute, then do this:
            except AttributeError:
                # read command from command line
                self.command = input()

        # end of while block, end of running. Bye, bye!
        self.bye_()
        return


if __name__ == "__main__":
    """
    This IF statement guarantees that the execution of the file will only ocurr
    when explicitly told so, e.g.:

        ~ $ python audio_player.py mywavefile.wav

    It is simmilar to the "int main() {}" statement on C/C++
    """
    try:
        file = parseArgs(sys.argv[1])
    except IndexError:
        file = None
    finally:
        player = AudioPlayer(file)
        player.exec_()
