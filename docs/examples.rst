Examples
========

Some examples


Sweep and play
--------------

Create an exponential sine sweep from 50 Hz to 16 kHz

    >>> swp = pytta.generate.sweep(freqMin=50, freqMax=16e3)
    >>> swp.play()
    

Recording
---------

Create a measurement object that records sound and is capable of calibration of levels.

    >>> recms = pytta.generate.measurement('rec')
    >>> rec = recms.run()
    >>> rec.play()
    

Playback and Record
-------------------

Create a measurement object that plays a signal and records microphone input simultaneously.

    >>> prms = pytta.generate.measurement('playrec', excitation=swp)
    >>> rec = prms.run()
    >>> rec.play()
    

More examples
-------------

Still to be done.

