# -*- coding: utf-8 -*-
"""
This submodule provides the tools for instantiating the measurement and
signal objects to be used. We strongly recommend the use of this submodule
instead of directly instantiating classes, except when necessary.

The signal generating functions already have set up a few good practices
on signal generation and reproduction through audio IO interfaces, like
silences at beginning and ending of the signal, as well as fade ins and
fade out to avoid abrupt audio currents from flowing and causing undesired
peaks at start/ending of reproduction.

On the measurement side, it tries to set up the environment by already
giving excitation signals, or by generating a SWEEP from default values

User intended functions:

    >>> pytta.generate.sin()
    >>> pytta.generate.sweep()
    >>> pytta.generate.random_noise()
    >>> pytta.generate.impulse()
    >>> pytta.generate.measurement()

For further information see the specific function documentation

@authors:
- João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br
- Matheus Lazarin Alberto, mtslazarin@gmail.com

"""


# Import modules
from pytta import default
from pytta.classes import SignalObj, RecMeasure, FRFMeasure, \
                          PlayRecMeasure, Streaming
from pytta.classes import OctFilter as _OctFilter
from scipy import signal as ss
import numpy as np
import traceback
from warnings import warn


def sin(Arms=0.5,
        freq=1000,
        timeLength=1,
        phase=2*np.pi,
        samplingRate=default.samplingRate,
        fftDegree=None):
    """
    Generates a sine signal with the traditional parameters plus some PyTTa
    options.

    Creation parameters:
    --------------------

        * Arms (float) (optional):
            The signal's RMS amplitude.

            >>> Apeak = Arms*sqrt(2);

        * freq (float) (optional):
            Nothing to say;

        * timeLength (float) (optional):
            Sine timeLength in seconds;

        * fftDegree (int) (optional);
            2**fftDegree signal's number of samples;

        * phase (float) (optional):
            Sine phase in radians;

        * samplingRate (int) (optional):
            Nothing to say;

    """
    # Code snippet to guarantee that generated object name is
    # the declared at global scope
    # for frame, line in traceback.walk_stack(None):
    for framenline in traceback.walk_stack(None):
        # varnames = frame.f_code.co_varnames
        varnames = framenline[0].f_code.co_varnames
        if varnames == ():
            break
    # creation_file, creation_line, creation_function, \
    #     creation_text = \
    extracted_text = \
        traceback.extract_stack(framenline[0], 1)[0]
        # traceback.extract_stack(frame, 1)[0]
    # creation_name = creation_text.split("=")[0].strip()
    creation_name = extracted_text[3].split("=")[0].strip()

    if fftDegree is not None:
        timeLength = 2**(fftDegree)/samplingRate

    t = np.linspace(0,
                    timeLength - (1/samplingRate),
                    int(samplingRate*timeLength))
    sin = Arms*(2**(1/2)) * np.sin(2*np.pi*freq*t+phase)
    sinSigObj = SignalObj(sin, domain='time', samplingRate=samplingRate,
                          freqMin=default.freqMin, freqMax=default.freqMax)
    sinSigObj.creation_name = creation_name
    return sinSigObj

def sweep(freqMin=None,
          freqMax=None,
          samplingRate=None,
          fftDegree=None,
          startMargin=None,
          stopMargin=None,
          method='logarithmic',
          windowing='hann'):
    """
    Generates a chirp signal defined by the "method" input, windowed, with
    silence interval at the beggining and end of the signal, plus a hanning
    fade in and fade out.

    >>> x = pytta.generate.sweep()
    >>> x.plot_time()

    Return a signalObj containing a logarithmic chirp signal from 17.8 Hz
    to 22050 Hz, with a fade in beginning at 17.8 Hz time instant and ending at
    the 20 Hz time instant; plus a fade out beginning at 20000 Hz time instant
    and ending at 22050 Hz time instant.

    The fade in and the fade out are made with half hanning window. First half
    for the fade in and last half for the fade out. Different number of points
    are used for each fade, so the number of time samples during each frequency
    is respected.

    Input arguments (default), (type):
    ------------------------

        * freqMin (20), (float)

        * freqMax (20), (float)

        * samplingRate (44100), (int)

        * fftDegree (18), (float)

        * startMargin (0.3), (float)

        * stopMargin (0.7), (float)

        * method (logarithmic'), (string)

        * windowing ('hann'), (string)


    """
    # Code snippet to guarantee that generated object name is
    # the declared at global scope
    # for frame, line in traceback.walk_stack(None):
    for framenline in traceback.walk_stack(None):
        # varnames = frame.f_code.co_varnames
        varnames = framenline[0].f_code.co_varnames
        if varnames == ():
            break
    # creation_file, creation_line, creation_function, \
    #     creation_text = \
    extracted_text = \
        traceback.extract_stack(framenline[0], 1)[0]
        # traceback.extract_stack(frame, 1)[0]
    # creation_name = creation_text.split("=")[0].strip()
    creation_name = extracted_text[3].split("=")[0].strip()


    # It was done like this because a function default argument is a value
    # assigned at import time, and PyTTa have a default object that handles
    # default values for all functions and all classes across all submodules.
    # In order to it work as expected, the values should be reassigned at
    # every function call to get updated default values. Otherwise, despite
    # how the default has it's properties values changed, it won't change
    # for the function calls.
    if freqMin is None:
        freqMin = default.freqMin
    if freqMax is None:
        freqMax = default.freqMax
    if samplingRate is None:
        samplingRate = default.samplingRate
    if fftDegree is None:
        fftDegree = default.fftDegree
    if startMargin is None:
        startMargin = default.startMargin
    if stopMargin is None:
        stopMargin = default.stopMargin

    # frequency limits [Hz]
    freqLimits = {'freqMin': freqMin / (2**(1/6)),
                  'freqMax': min(freqMax*(2**(1/6)), samplingRate/2)}
    samplingTime = 1/samplingRate  # [s] sampling period

    stopSamples = stopMargin*samplingRate
    # [samples] initial silence number of samples

    startSamples = startMargin*samplingRate
    # [samples] ending silence number of samples

    marginSamples = startSamples + stopSamples
    # [samples] total silence number of samples

    numSamples = 2**fftDegree  # [samples] full signal number of samples

    sweepSamples = numSamples - marginSamples + 1
    # [samples] actual sweep number of samples

    if sweepSamples < samplingRate/10:
        raise Exception('Too small resultant sweep. For such big margins you' +
                        ' must increase your fftDegree.')

    sweepTime = sweepSamples/samplingRate  # [s] sweep's time length
    timeVecSweep = np.arange(0, sweepTime, samplingTime)  # [s] time vector
    if timeVecSweep.size > sweepSamples:
        timeVecSweep = timeVecSweep[0:int(sweepSamples)]  # adjust length
    sweep = 0.95*ss.chirp(timeVecSweep,
                          freqLimits['freqMin'],
                          sweepTime,
                          freqLimits['freqMax'],
                          'logarithmic',
                          phi=-90)  # sweep, time domain
    sweep = __do_sweep_windowing(sweep,
                                 timeVecSweep,
                                 freqLimits,
                                 freqMin,
                                 freqMax,
                                 windowing)  # fade in and fade out
    # add initial and ending slices
    timeSignal = np.concatenate((np.zeros(int(startSamples)),
                                 sweep,
                                 np.zeros(int(stopSamples))))
    if timeSignal.size != numSamples:
        timeSignal = timeSignal[0:int(numSamples)]  # adjust length

    # transforms into a pytta signalObj and sets the correct name
    sweepSignal = SignalObj(signalArray=timeSignal, domain='time',
                            samplingRate=samplingRate,
                            **freqLimits)

    sweepSignal.creation_name = creation_name

    return sweepSignal


def __do_sweep_windowing(inputSweep,
                         timeVecSweep,
                         freqLimits,
                         freqMin,
                         freqMax,
                         window):
    """
    Applies a fade in and fade out that are minimum at the chirp start and end,
    and maximum between the time intervals corresponding to Finf and Fsup.
    """

    # frequencies at time instants: freq(t)
    freqSweep = freqLimits['freqMin']*(
            (freqLimits['freqMax'] / freqLimits['freqMin'])**(
                    1/max(timeVecSweep))) ** timeVecSweep

    # exact sample where the chirp reaches freqMin [Hz]
    freqMinSample = np.where(freqSweep <= freqMin)
    freqMinSample = freqMinSample[-1][-1]

    # exact sample where the chirp reaches freqMax [Hz]
    freqMaxSample = np.where(freqSweep <= freqMax)
    freqMaxSample = len(freqSweep) - freqMaxSample[-1][-1]
    windowStart = ss.hanning(2*freqMinSample)
    windowEnd = ss.hanning(2*freqMaxSample)

    # Uses first half of windowStart, last half of windowEnd, and a vector of
    # ones with the remaining length, in between the half windows
    fullWindow = np.concatenate((windowStart[0:freqMinSample],
                                 np.ones(int(len(freqSweep) -
                                             freqMinSample -
                                             freqMaxSample + 1)),
                                 windowEnd[freqMaxSample:-1]))
    newSweep = fullWindow * inputSweep
    return newSweep

# FIXME: This looks incorrect because the signal has normal
#        distribution, so no limits but an average and standard deviation.
def random_noise(kind='white',
                 samplingRate=None,
                 fftDegree=None,
                 startMargin=None,
                 stopMargin=None,
                 windowing='hann'):
    """
    Generates a noise of kind White, Pink (TO DO) or Blue (TO DO), with a
    silence at the beginning and ending of the signal, plus a fade in to avoid
    abrupt speaker excursion. All noises have normalized amplitude.

        White noise is generated using numpy.randn between [[1];[-1]];

        Pink noise is still in progress;

        Blue noise is still in progress;
    """
    # Code snippet to guarantee that generated object name is
    # the declared at global scope
    # for frame, line in traceback.walk_stack(None):
    for framenline in traceback.walk_stack(None):
        # varnames = frame.f_code.co_varnames
        varnames = framenline[0].f_code.co_varnames
        if varnames == ():
            break
    # creation_file, creation_line, creation_function, \
    #     creation_text = \
    extracted_text = \
        traceback.extract_stack(framenline[0], 1)[0]
        # traceback.extract_stack(frame, 1)[0]
    # creation_name = creation_text.split("=")[0].strip()
    creation_name = extracted_text[3].split("=")[0].strip()

    # It was done like this because a function default argument is a value
    # assigned at import time, and PyTTa have a default object that handles
    # default values for all functions and all classes across all submodules.
    # In order to it work as expected, the values should be reassigned at
    # every function call to get updated default values. Otherwise, despite
    # how the default has it's properties values changed, it won't change
    # for the function calls.
    if samplingRate is None:
        samplingRate = default.samplingRate
    if fftDegree is None:
        fftDegree = default.fftDegree
    if startMargin is None:
        startMargin = default.startMargin
    if stopMargin is None:
        stopMargin = default.stopMargin

    # [samples] initial silence number of samples
    stopSamples = round(stopMargin*samplingRate)

    # [samples] ending silence number of samples
    startSamples = round(startMargin*samplingRate)

    # [samples] total silence number of samples
    marginSamples = startSamples + stopSamples

    # [samples] full signal number of samples
    numSamples = 2**fftDegree

    # [samples] Actual noise number of samples
    noiseSamples = int(numSamples - marginSamples)
    if kind.upper() in ['WHITE', 'FLAT']:
        noiseSignal = np.random.randn(noiseSamples)
#    elif kind.upper() == 'PINK:  # TODO
#        noiseSignal = np.randn(Nnoise)
#        noiseSignal = noiseSignal/max(abs(noiseSignal))
#        noiseSignal = __do_pink_filtering(noiseSignal)
#    elif kind.upper() == 'BLUE:  # TODO
#        noiseSignal = np.randn(Nnoise)
#        noiseSignal = noiseSignal/max(abs(noiseSignal))
#        noiseSignal = __do_blue_filtering(noiseSignal)

    noiseSignal = __do_noise_windowing(noiseSignal, noiseSamples, windowing)
    noiseSignal = noiseSignal / max(abs(noiseSignal))
    noiseSignal = np.concatenate((np.zeros(int(startSamples)),
                                 noiseSignal,
                                 np.zeros(int(stopSamples))))
    noiseSignal = SignalObj(signalArray=noiseSignal, domain='time',
                            freqMin=default.freqMin, freqMax=default.freqMax,
                            samplingRate=samplingRate)
    noiseSignal.creation_name = creation_name
    return noiseSignal


def __do_noise_windowing(inputNoise,
                         noiseSamples,
                         window):
    # sample equivalent to the first five percent of noise duration
    fivePercentSample = int((5/100) * (noiseSamples))
    windowStart = ss.hanning(2*fivePercentSample)
    fullWindow = np.concatenate((windowStart[0:fivePercentSample],
                                 np.ones(int(noiseSamples-fivePercentSample))))
    newNoise = fullWindow * inputNoise
    return newNoise


def impulse(samplingRate=None,
            fftDegree=None):
    """
    Generates a normalized impulse signal at time zero,
    with zeros to fill the time length
    """
    # Code snippet to guarantee that generated object name is
    # the declared at global scope
    # for frame, line in traceback.walk_stack(None):
    for framenline in traceback.walk_stack(None):
        # varnames = frame.f_code.co_varnames
        varnames = framenline[0].f_code.co_varnames
        if varnames == ():
            break
    # creation_file, creation_line, creation_function, \
    #     creation_text = \
    extracted_text = \
        traceback.extract_stack(framenline[0], 1)[0]
        # traceback.extract_stack(frame, 1)[0]
    # creation_name = creation_text.split("=")[0].strip()
    creation_name = extracted_text[3].split("=")[0].strip()

    if samplingRate is None:
        samplingRate = default.samplingRate
    if fftDegree is None:
        fftDegree = default.fftDegree

    numSamples = 2**fftDegree
    impulseSignal = np.zeros(numSamples)
    impulseSignal[0] = 1.0
    impulseSignal = SignalObj(signalArray=impulseSignal,
                              domain='time',
                              samplingRate=samplingRate,
                              signalType='energy')
    impulseSignal.creation_name = creation_name
    return impulseSignal


def measurement(kind='playrec',
                samplingRate=None,
                freqMin=None,
                freqMax=None,
                device=None,
                inChannels=None,
                outChannels=None,
                *args,
                **kwargs):
    """
    Generates a measurement object of type Recording, Playback and Recording,
    Transferfunction, with the proper initiation arguments, a sampling rate,
    frequency limits, audio input and output devices and channels

        >>> pytta.generate.measurement(kind,
                                       [lengthDomain,
                                       fftDegree,
                                       timeLength,
                                       excitation,
                                       outputAmplification],
                                       samplingRate,
                                       freqMin,
                                       freqMax,
                                       device,
                                       inChannels,
                                       outChannels,
                                       comment)

    The parameters between brackets are different for each value of the (kind)
    parameter.

    >>> msRec = pytta.generate.measurement(kind='rec')
    >>> msPlayRec = pytta.generate.measurement(kind='playrec')
    >>> msFRF = pytta.generate.measurement(kind='frf')

    The input arguments may be different for each measurement kind.

    Options for (kind='rec'):

        - lengthDomain: 'time' or 'samples', defines if the recording length
            will be set by time length, or number of samples;

        - timeLength: [s] used only if (domain='time'), set the duration of the
            recording, in seconds;

        - fftDegree: represents a power of two value that defines the number of
            samples to be recorded:

                >>> numSamples = 2**fftDegree

        - samplingRate: [Hz] sampling frequency of the recording;
        - freqMin: [Hz] smallest frequency of interest;
        - freqMax: [Hz] highest frequency of interest;
        - device: audio I/O device to use for recording;
        - inChannels: list of active channels to record;
        - comment: any commentary about the recording.


    Options for (kind='playrec'):

        - excitation: object of SignalObj class, used for the playback;
        - outputAmplification: output gain in dB;
        - samplingRate: [Hz] sampling frequency of the recording;
        - freqMin: [Hz] smallest frequency of interest;
        - freqMax: [Hz] highest frequency of interest;
        - device: audio I/O device to use for recording;
        - inChannels: list of active channels to record;
        - outChannels: list of active channels to send the playback signal, for
            M channels it is mandatory for the excitation signal to have M
            columns in the timeSignal parameter.
        - comment: any commentary about the recording.

    Options for (kind='frf'):

        - same as for (kind='playrec');
        - regularization: [boolean] option for Kirkeby regularization
    """
    # Code snippet to guarantee that generated object name is
    # the declared at global scope
    # for frame, line in traceback.walk_stack(None):
    for framenline in traceback.walk_stack(None):
        # varnames = frame.f_code.co_varnames
        varnames = framenline[0].f_code.co_varnames
        if varnames == ():
            break
    # creation_file, creation_line, creation_function, \
    #     creation_text = \
    extracted_text = \
        traceback.extract_stack(framenline[0], 1)[0]
        # traceback.extract_stack(frame, 1)[0]
    # creation_name = creation_text.split("=")[0].strip()
    creation_name = extracted_text[3].split("=")[0].strip()

# Default Parameters
    if freqMin is None:
        freqMin = default.freqMin
    if freqMax is None:
        freqMax = default.freqMax
    if samplingRate is None:
        samplingRate = default.samplingRate
    if device is None:
        device = default.device
    if inChannels is None:
        inChannels = default.inChannel[:]
    if outChannels is None:
        outChannels = default.outChannel[:]

# Kind REC
    if kind in ['rec', 'record', 'recording', 'r']:
        recordObj = RecMeasure(samplingRate=samplingRate, freqMin=freqMin,
                               freqMax=freqMax, device=device,
                               inChannels=inChannels, **kwargs)
        if ('lengthDomain' in kwargs) or args:
            if kwargs.get('lengthDomain') == 'time':
                recordObj.lengthDomain = 'time'
                try:
                    recordObj.timeLength = kwargs.get('timeLength')
                except KeyError:
                    recordObj.timeLength = default.timeLength
            elif kwargs.get('lengthDomain') == 'samples':
                recordObj.lengthDomain = 'samples'
                try:
                    recordObj.fftDegree = kwargs.get('fftDegree')
                except KeyError:
                    recordObj.fftDegree = default.fftDegree
        else:
            recordObj.lengthDomain = 'samples'
            recordObj.fftDegree = default.fftDegree
        recordObj.creation_name = creation_name
        return recordObj

# Kind PLAYREC
    elif kind in ['playrec', 'playbackrecord', 'pr']:
        if 'outputAmplification' in kwargs:
            outputAmplification = kwargs.get('outputAmplification')
            kwargs.pop('outputAmplification', None)
        else:
            outputAmplification = 0
        if ('excitation' in kwargs.keys()) or args:
            signalIn = kwargs.get('excitation') or args[0]
            kwargs.pop('excitation', None)
        else:
            signalIn = sweep(samplingRate=samplingRate,
                             freqMin=freqMin,
                             freqMax=freqMax,
                             **kwargs)

        playRecObj = PlayRecMeasure(excitation=signalIn,
                                    outputAmplification=outputAmplification,
                                    device=device,
                                    inChannels=inChannels,
                                    outChannels=outChannels,
                                    freqMin=freqMin,
                                    freqMax=freqMax, **kwargs)
        playRecObj.creation_name = creation_name
        return playRecObj

# Kind FRF
    elif kind in ['tf', 'frf', 'transferfunction', 'freqresponse']:
        if 'regularization' in kwargs:
            regularization = kwargs.get('regularization')
            kwargs.pop('regularization', None)
        else:
            regularization = True
        if 'outputAmplification' in kwargs:
            outputAmplification = kwargs.get('outputAmplification')
            kwargs.pop('outputAmplification', None)
        else:
            outputAmplification = 0
        if ('excitation' in kwargs) or args:
            signalIn = kwargs.get('excitation') or args[0]
            kwargs.pop('excitation', None)
        else:
            signalIn = sweep(samplingRate=samplingRate,
                             freqMin=freqMin,
                             freqMax=freqMax,
                             **kwargs)

        frfObj = FRFMeasure(excitation=signalIn,
                            outputAmplification=outputAmplification,
                            device=device,
                            inChannels=inChannels,
                            outChannels=outChannels,
                            freqMin=freqMin,
                            freqMax=freqMax,
                            regularization=regularization,
                            **kwargs)
        frfObj.creation_name = creation_name
        return frfObj


def stream(IO='IO',
           device=None,
           integration=None,
           samplingRate=None,
           inChannels=None,
           outChannels=None,
           duration=None,
           excitation=None,
           callback=None,
           *args, **kwargs):
    """
    """
    # Code snippet to guarantee that generated object name is
    # the declared at global scope
    # for frame, line in traceback.walk_stack(None):
    for framenline in traceback.walk_stack(None):
        # varnames = frame.f_code.co_varnames
        varnames = framenline[0].f_code.co_varnames
        if varnames == ():
            break
    # creation_file, creation_line, creation_function, \
    #     creation_text = \
    extracted_text = \
        traceback.extract_stack(framenline[0], 1)[0]
        # traceback.extract_stack(frame, 1)[0]
    # creation_name = creation_text.split("=")[0].strip()
    creation_name = extracted_text[3].split("=")[0].strip()

    if device is None:
        device = default.device
    if integration is None:
        integration = default.integration
    if inChannels is None:
        inChannels = default.inChannel[:]
    if isinstance(excitation, SignalObj):
        excit = True
        excitData = excitation.timeSignal[:]
        samplingRate = excitation.samplingRate
        duration = excitation.timeLength
        outChannels = excitation.channels[:]
    else:
        excit = False
        if samplingRate is None:
            samplingRate = default.samplingRate

    if IO in ['I', 'in', 'input']:
        stream = Streaming(device=device, integration=integration,
                           inChannels=inChannels, duration=duration,
                           callback=callback, samplingRate=samplingRate,
                           *args, **kwargs)

    elif IO in ['O', 'out', 'output']:
        if excit:
            stream = Streaming(device=device, integration=integration,
                               outChannels=outChannels, duration=duration,
                               excitationData=excitData,
                               samplingRate=samplingRate, callback=callback,
                               *args, **kwargs)
        else:
            excitation = sweep(samplingRate=samplingRate)
            outChannels = excitation.channels
            duration = excitation.timeLength
            excitData = excitation.timeSignal[:]
            stream = Streaming(device=device, integration=integration,
                               outChannels=outChannels, duration=duration,
                               excitationData=excitData,
                               samplingRate=samplingRate, callback=callback,
                               *args, **kwargs)

    elif IO in ['IO', 'in-out', 'input-output']:
        if excit:
            stream = Streaming(device=device, integration=integration,
                               inChannels=inChannels, outChannels=outChannels,
                               duration=duration, excitationData=excitData,
                               samplingRate=samplingRate, callback=callback,
                               *args, **kwargs)
        else:
            excitation = sweep(samplingRate=samplingRate)
            outChannels = excitation.channels[:]
            duration = excitation.timeLength
            excitData = excitation.timeSignal[:]
            stream = Streaming(device=device, integration=integration,
                               inChannels=inChannels, outChannels=outChannels,
                               duration=duration, excitationData=excitData,
                               samplingRate=samplingRate, callback=callback,
                               *args, **kwargs)
    else:
        raise ValueError("The IO parameter could not identify whether the\
                         stream will be Input, Output or Input-Output type.")
    stream.creation_name = creation_name
    return stream

def filter(*args, **kwargs) -> _OctFilter:
    warn(DeprecationWarning("'pytta.generate.filter' DEPRECATED and being " +
                                "replaced by 'pytta.generate.octfilter'."))
    return octfilter(*args, **kwargs)


def octfilter(order: int = 4,
              nthOct: int = 3,
              samplingRate: int = 44100,
              minFreq: float = 20,
              maxFreq: float = 20000,
              refFreq: float = 1000,
              base: int = 10) -> _OctFilter:
    # Code snippet to guarantee that generated object name is
    # the declared at global scope
    # for frame, line in traceback.walk_stack(None):
    for framenline in traceback.walk_stack(None):
        # varnames = frame.f_code.co_varnames
        varnames = framenline[0].f_code.co_varnames
        if varnames == ():
            break
    # creation_file, creation_line, creation_function, \
    #     creation_text = \
    extracted_text = \
        traceback.extract_stack(framenline[0], 1)[0]
        # traceback.extract_stack(frame, 1)[0]
    # creation_name = creation_text.split("=")[0].strip()
    creation_name = extracted_text[3].split("=")[0].strip()

    of = _OctFilter(order, nthOct, samplingRate, minFreq,
                   maxFreq, refFreq, base)
    of.creation_name = creation_name
    return of
