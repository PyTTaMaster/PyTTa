# -*- coding: utf-8 -*-


import pytta

samplerate = 44100  # number of samples that represents one second of data

timeweight = samplerate // 8  # fast integration time on sound level meters


dev = pytta.get_device_from_user()

mon = pytta.Monitor(timeweight)

ms = pytta.Measurement(samplingRate=44100,
                       numSamples=2**18,
                       device=dev,
                       inChannels=2,
                       outChannels=2)

print("\nRecording:")
recsigobj = ms.record(monitor = mon)


# Define the recently recorded audio data as the excitation signal
ms.excitation = recsigobj

print("\nPlaying:")
ms.play(monitor = mon)
