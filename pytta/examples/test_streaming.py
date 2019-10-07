# -*- coding: utf-8 -*-


if __name__ == "__main__":
    from pytta import generate, Recorder, SignalObj
    recmeasure = generate.measurement('rec')

    with Recorder(recmeasure) as rec:
        rec.set_monitoring(True)
        rec.run()
        signal = SignalObj(rec.recData, 'time', rec.samplingRate,
                           freqMin=20, freqMax=20e3)

    signal.plot_time()
