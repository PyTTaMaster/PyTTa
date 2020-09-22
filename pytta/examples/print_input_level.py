# -*- coding: utf-8 -*-

# import numpy as np
# import sounddevice as sd
# # import matplotlib
# # matplotlib.use('TkAgg')
# from matplotlib import pyplot as plt
# # from threading import Thread
# from pytta.classes.measurement import Measurement
# from pytta.classes.streaming import Recorder, Player, PlaybackRecorder, Monitor


# class DoomyGraphicMonitor(Monitor):
#     """
#     Dummy (duh!) class for simple default graphical level meter to visualize the audio streaming.
#     """

#     integ = {           # samples integration frequency in Hz, it is the inverse of the samples
#         'fast': 8,      # integration time, so integ=('fast', 8) means that it will update the graphic at
#         'slow': 1,      # every 1/8 of a second, as does the Fast integration time, and so on.
#         'peak': 12.5    # This is done in samples domain, the amount of samples is =samplingRate//integ
#     }                   # fast -> samplingRate//8; slow -> samplingRate//1; peak -> samplingRate//12.5

#     def __init__(self, samplingrate, numchannels, integtime):
#         """
#         Instantiate and copy any information about the audio reproduction. The
#         """
#         numsamples = samplingrate // self.integ[integtime]
#         super().__init__(numsamples, samplingrate, numchannels)
#         return

#     def setup1(self):
#         f = plt.figure('Dummy a favor', figsize=[1.8, 3.6])
#         f.add_axes([0.2, 0.1, 0.7, 0.8])
#         ylim0 = 20*np.log10(1e-5)
#         ylim1 = np.log10(1) + 5
#         f.axes[0].set_ylim([ylim0, ylim1])
#         b = f.axes[0].bar(['in', 'out'], [-50-ylim0, -50-ylim0], [0.3, 0.3], bottom=ylim0)
#         b.patches[0].set_color('green')
#         b.patches[1].set_color('green')
#         f.show()
#         f.canvas.draw()
#         self.fig = f
#         self.bars = b
#         self.ylim0 = f.axes[0].get_ylim()[0]
#         return

#     def set_heigth(self, height, idx):
#         if height >= -10:
#             self.bars.patches[idx].set_color('yellow')
#         elif height >= 0:
#             self.bars.patches[idx].set_color('red')
#         else:
#             self.bars.patches[idx].set_color('green')
#         self.bars.patches[idx].set_height(height - self.ylim0)
#         return

#     def update_plot(self, inHeight, outHeight):
#         self.set_heigth(inHeight, 0)
#         self.set_heigth(outHeight, 1)
#         self.fig.canvas.draw()
#         self.fig.canvas.flush_events()
#         return

#     def callback1(self, indata: np.ndarray, outdata: np.ndarray, frames: int, status: sd.CallbackFlags):
#         """
#         Standard output print sound full scale level:
#         ----------------------------------------------

#             Prints the Full Scale level in decibel of the incoming sound at integration time

#         :param data: the audio data from opened stream
#         :type data: numpy.ndarray
#         :param frames: the number of samples (rows) for each column
#         :type frames: int
#         :param status: A condition about incoming and/or outgoing data
#         :type status: sounddevice.CallbackFlags
#         :return: Nothing for a "Continue" and a flag for stopping or aborting
#         :rtype: sounddevice.CallbackStop or sounddevice.CallbackAbort

#         Buffers enough samples to make approximately 125 ms of signal and calculate
#         the root-mean-square of the buffer, then resets it
#         """
#         self.inData[self.counter:frames + self.counter] \
#             = indata if not (indata is None) else 1e-13
#         self.outData[self.counter:frames + self.counter, :] \
#             = outdata if not (outdata is None) else 1e-13
#         self.counter += frames
#         if (self.numSamples - self.counter) < frames:
#             dBin = 20 * np.log10(float(np.mean(self.inData ** 2, axis=0)) ** 0.5)
#             dBout = 20 * np.log10(float(np.mean(self.outData ** 2, axis=0)) ** 0.5)
#             self.update_plot(dBin, dBout)
#             self.counter = 0
#         return

#     def tear_down1(self):
#         plt.close(self.fig)
#         return

#     def setup2(self):
#         return

#     def tear_down2(self):
#         return

#     def callback2(self, indata, outdata, frames, status):
#         self.inData[self.counter:frames + self.counter] \
#             = indata.copy() if not (indata is None) else 1e-13
#         self.outData[self.counter:frames + self.counter, :] \
#             = outdata.copy() if not (outdata is None) else 1e-13
#         self.counter += frames
#         if (self.numSamples - self.counter) < frames:
#             dBin = 20 * np.log10(float(np.mean(self.inData ** 2, axis=0)) ** 0.5)
#             dBout = 20 * np.log10(float(np.mean(self.outData ** 2, axis=0)) ** 0.5)
#             print(f'dB in: {dBin}   dB out: {dBout}')
#             self.counter = 0
#         return

#     def callback3(self, indata, outdata, frames, status):
#         print(indata)
#         return

#     def setup(self):
#         self.setup1()
#         return

#     def callback(self, indata, outdata, frames, status):
#         self.callback1(indata, outdata, frames, status)
#         return

#     def tear_down(self):
#         self.tear_down1()
#         return

# def rec(msmnt: Measurement=None, monitor: object=None, bs: int=0):
#     if msmnt is None:
#         msmnt = generate.measurement('rec')
#     with Recorder(msmnt, 'float32', bs) as rec:  # Creates context with Recorder object
#         if monitor is not None:
#             rec.set_monitoring(monitor)  # Sets the monitor function
#         rec.run()    # start to record
#         signal = rec.retrieve()
#     return signal

# def play(msmnt: Measurement=None, monitor: object=None, bs: int=0):
#     if msmnt is None:
#         msmnt = generate.measurement('playrec')
#     with Player(msmnt, 'float32', bs) as player:  # Creates context with Player object
#         if monitor is not None:
#             player.set_monitoring(monitor)  # Sets the monitor function
#         player.run()    # start to reproduce
#     return

# def playrec(msmnt: Measurement=None, monitor: object=None, bs: int=0):
#     if msmnt is None:
#         msmnt = generate.measurement('playrec')
#     with PlaybackRecorder(msmnt, 'float32', bs) as pbr:  # Creates context with Player object
#         pbr.set_monitoring(monitor)  # Sets the monitor function

#         signal = pbr.retrieve()
#     return signal


if __name__ == "__main__":
    import pytta as pta

    dev = [0, 0]
    # print(pta.list_devices())
    # devices = input("Device ID -> in, out: ")
    # dev = [int(device.strip(" ")) for device in devices.split(",")]

    print(dev)

    noise = pta.generate.noise(fftDegree=17)

    measure = pta.generate.measurement(excitation=noise, device=dev)  # generates a default PlayRecMeasure object

    doomsy = pta.Monitor(int(44100//8))
    # doomsy = DoomyGraphicMonitor(measure.samplingRate,  # Instantiates Monitor subclass
    #                              measure.numInChannels,
    #                              'fast')  # blocksize

    # signal = playrec(measure, monitor=doomsy, bs=bs)

    stream = pta.Streaming('io', measure, monitor=None)
    # data = stream.record()
    # stream.play()
    data = stream.playrec()

    # print(data.shape)
