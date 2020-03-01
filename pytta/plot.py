""" 
plot:
------

    This plot module receives organized raw data (numpy arrays, lists and
    strings) and plot it through its functions. It is used by the function,
    signal, and analysis modules.

    Available functions:
    ---------------------

        >>> plot.time(curveData, xLabel, yLabel, yLim, xLim, title, decimalSep)
        >>> plot.freq(curveData, smooth, xLabel, yLabel, yLim, xLim, title,
                      decimalSep)
        >>> plot.bars(curveData, xLabel, yLabel, yLim, title, decimalSep,
                      barWidth, errorStyle)

    For further information check the function especific documentation.
    
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import locale
import numpy as np
import scipy.signal as ss
import copy as cp


def time(sigObjs, xLabel, yLabel, yLim, xLim, title, decimalSep):
    """Plots a signal in time domain.

    Parameters (default), (type):
    ------------------------------

        * sigObjs (), (list):
            a list with SignalObjs
        
        * xLabel ('Time [s]'), (str):
            x axis label.

        * yLabel ('Amplitude'), (str):
            y axis label.

        * yLim (), (list):
            inferior and superior limits.

            >>> yLim = [-100, 100]

        * xLim (), (list):
            left and right limits

            >>> xLim = [0, 15]

        * title (), (str):
            plot title

        * decimalSep (','), (str):
            may be dot or comma.

            >>> decimalSep = ',' # in Brazil

    Return:
    --------

        matplotlib.figure.Figure object.
    """
    if decimalSep == ',':
        locale.setlocale(locale.LC_NUMERIC, 'pt_BR.UTF-8')
        plt.rcParams['axes.formatter.use_locale'] = True
    elif decimalSep =='.':
        locale.setlocale(locale.LC_NUMERIC, 'C')
        plt.rcParams['axes.formatter.use_locale'] = False
    else:
        raise ValueError("'decimalSep' must be the string '.' or ','.")

    if xLabel is None:
        xLabel = 'Time [s]'
    if yLabel is None:
        yLabel = 'Amplitude'
    if title is None:
        title = ''

    xLims = np.array([np.nan, np.nan], dtype=float, ndmin=2)
    yLims = np.array([np.nan, np.nan], dtype=float, ndmin=2)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes([0.10, 0.21, 0.88, 0.72], polar=False,
                        projection='rectilinear', xscale='linear')
    ax.set_snap(True)
    ax.grid(color='gray', linestyle='-.', linewidth=0.4)
    fig.suptitle(title, fontsize=20)

    curveData = _curve_data_extrator_time(sigObjs)

    for data in curveData:
        ax.plot(data['x'], data['y'], label=data['label'])
        yLimData = cp.copy(data['y'])
        yLimData[np.isinf(yLimData)] = 0
        yLims = \
            np.vstack((yLims, [np.nanmin(yLimData), np.nanmax(yLimData)]))
        xLims = \
            np.vstack((xLims, [data['x'][0], data['x'][-1]]))
    
    ax.set_xlabel(xLabel, fontsize=20)
    ax.set_ylabel(yLabel, fontsize=20)
    ax.legend(loc='best', fontsize=12)

    if xLim is None:
        xLim = [np.nanmin(xLims[:,0]), np.nanmax(xLims[:,1])]
    ax.set_xlim(xLim)

    if yLim is None:
        yLim = [np.nanmin(yLims[:,0]), np.nanmax(yLims[:,1])]
        margin = (yLim[1] - yLim[0]) / 20
        yLim = [yLim[0]-margin, yLim[1]+margin]
    ax.set_ylim(yLim)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=8))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=10))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
    
    return fig

def _curve_data_extrator_time(sigObjs):
    """
    Extracts data from all curves from each SignalObj

    Parameter (default), (type):
    -----------------------------

        * sigObj (), (list):
            a list with SignalObjs for curve data extraction

    Return (type):
    --------------

        * curveData (list):
            A list with dictionaries containing the information about each
            curve. Kys for time plot:

                - 'x', (), (ndarray): time axis;
                - 'y', (), (ndarray): amplitude axis;
                - 'label', (), (str): curve label.

                >>> curveData = [{'x':x, 'y':y, 'label':'my beautiful curve'}]
    """
    curveData = []
    for sigObj in sigObjs:
        for chIndex in range(sigObj.numChannels):
            chNum = sigObj.channels.mapping[chIndex]
            label = '{} [{}]'.format(sigObj.channels[chNum].name,
                                     sigObj.channels[chNum].unit)
            x = sigObj.timeVector
            y = sigObj.timeSignal[:, chIndex]            
            curveData.append({
                'label':label,
                'x':x,
                'y':y
            })
    return curveData

def time_dB(sigObjs, xLabel, yLabel, yLim, xLim, title, decimalSep):
    """Plots a signal in decibels in time domain.

    Parameters (default), (type):
    ------------------------------

        * sigObjs (), (list):
            a list with SignalObjs.
        
        * xLabel ('Time [s]'), (str):
            x axis label.

        * yLabel ('Amplitude'), (str):
            y axis label.

        * yLim (), (list):
            inferior and superior limits.

            >>> yLim = [-100, 100]

        * xLim (), (list):
            left and right limits

            >>> xLim = [0, 15]

        * title (), (str):
            plot title

        * decimalSep (','), (str):
            may be dot or comma.

            >>> decimalSep = ',' # in Brazil

    Return:
    --------

        matplotlib.figure.Figure object.
    """
    if decimalSep == ',':
        locale.setlocale(locale.LC_NUMERIC, 'pt_BR.UTF-8')
        plt.rcParams['axes.formatter.use_locale'] = True
    elif decimalSep =='.':
        locale.setlocale(locale.LC_NUMERIC, 'C')
        plt.rcParams['axes.formatter.use_locale'] = False
    else:
        raise ValueError("'decimalSep' must be the string '.' or ','.")

    if xLabel is None:
        xLabel = 'Time [s]'
    if yLabel is None:
        yLabel = 'Magnitude'
    if title is None:
        title = ''

    xLims = np.array([np.nan, np.nan], dtype=float, ndmin=2)
    yLims = np.array([np.nan, np.nan], dtype=float, ndmin=2)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes([0.10, 0.21, 0.88, 0.72], polar=False,
                        projection='rectilinear', xscale='linear')
    ax.set_snap(True)
    ax.grid(color='gray', linestyle='-.', linewidth=0.4)
    fig.suptitle(title, fontsize=20)

    curveData = _curve_data_extractor_time_dB(sigObjs)
    for data in curveData:
        dBSignal = 20*np.log10(np.abs(data['y'])/data['dBRef'])
        ax.plot(data['x'], dBSignal, label=data['label'])
        yLimData = cp.copy(dBSignal)
        yLimData[np.isinf(yLimData)] = 0
        yLims = \
            np.vstack((yLims, [np.nanmin(yLimData), np.nanmax(yLimData)]))
        xLims = \
            np.vstack((xLims, [data['x'][0], data['x'][-1]]))
    
    ax.set_xlabel(xLabel, fontsize=20)
    ax.set_ylabel(yLabel, fontsize=20)
    ax.legend(loc='best', fontsize=12)

    if xLim is None:
        xLim = [np.nanmin(xLims[:,0]), np.nanmax(xLims[:,1])]
    ax.set_xlim(xLim)

    if yLim is None:
        yLim = [np.nanmin(yLims[:,0]), np.nanmax(yLims[:,1])]
        margin = (yLim[1] - yLim[0]) / 20
        yLim = [yLim[0]-margin, yLim[1]+margin]
    ax.set_ylim(yLim)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=8))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=10))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
    
    return fig

def _curve_data_extractor_time_dB(sigObjs):
    """
    Extracts data from all curves from each SignalObj

    Parameter (default), (type):
    -----------------------------

        * sigObj (), (list):
            a list with SignalObjs for curve data extraction

    Return (type):
    --------------

     * curveData (), (list):
            A list with dictionaries containing the information about each
            curve. Needed keys for time plot:

                - 'x', (), (ndarray): time axis;
                - 'y', (), (ndarray): linear amplitude axis;
                - 'label', (), (str): curve label;
                - 'dBRef', (), (float): decibel scale reference.

                >>> curveData = [{'x':x, 'y':y, 'label':'my beautiful curve',
                                'dBRef':2e-5}]
    """
    curveData = []
    for sigObj in sigObjs:
        for chIndex in range(sigObj.numChannels):
            chNum = sigObj.channels.mapping[chIndex]
            dBRef = sigObj.channels[chNum].dBRef
            label = '{} [dB ref.: {} {}]'.format(sigObj.channels[chNum].name,
                                                 dBRef,
                                                 sigObj.channels[chNum].unit)
            x = sigObj.timeVector
            y = sigObj.timeSignal[:, chIndex]
            curveData.append({
                'label':label,
                'x':x,
                'y':y,
                'dBRef':dBRef
            })
    return curveData

def freq(sigObjs, smooth, xLabel, yLabel, yLim, xLim, title, decimalSep):
    """Plots a signal decibel magnitude in frequency domain.

    Parameters (default), (type):
    -----------------------------

       * sigObjs (), (list):
            a list with SignalObjs.

        * smooth (False), (bool):
            option for curve smoothing. Uses scipy.signal.savgol_filter.
            Preliminar implementation. Needs review.

        * xLabel ('Time [s]'), (str):
            x axis label.

        * yLabel ('Amplitude'), (str):
            y axis label.

        * yLim (), (list):
            inferior and superior limits.

            >>> yLim = [-100, 100]

        * xLim (), (list):
            left and right limits

            >>> xLim = [15, 21000]

        * title (), (str):
            plot title

        * decimalSep (','), (str):
            may be dot or comma.

            >>> decimalSep = ',' # in Brazil

    Return:
    --------

        matplotlib.figure.Figure object.
    """
    if decimalSep == ',':
        locale.setlocale(locale.LC_NUMERIC, 'pt_BR.UTF-8')
        plt.rcParams['axes.formatter.use_locale'] = True
    elif decimalSep =='.':
        locale.setlocale(locale.LC_NUMERIC, 'C')
        plt.rcParams['axes.formatter.use_locale'] = False
    else:
        raise ValueError("'decimalSep' must be the string '.' or ','.")

    if xLabel is None:
        xLabel = 'Frequency [Hz]'
    if yLabel is None:
        yLabel = 'Magnitude'
    if title is None:
        title = ''

    yLims = np.array([np.nan, np.nan], dtype=float, ndmin=2)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes([0.10, 0.21, 0.88, 0.72], polar=False,
                        projection='rectilinear', xscale='linear')
    ax.set_snap(True)
    ax.grid(color='gray', linestyle='-.', linewidth=0.4)
    fig.suptitle(title, fontsize=20)

    curveData = _curve_data_extractor_freq(sigObjs)
    for data in curveData:
        dBSignal = 20*np.log10(data['y']/data['dBRef'])

        if smooth:
            dBSignal = ss.savgol_filter(dBSignal, 31, 3)
        
        ax.semilogx(data['x'], dBSignal, label=data['label'])

        yLimData = dBSignal
        yLimData[np.isinf(yLimData)] = np.nan
        yLims = \
            np.vstack((yLims, [np.nanmin(yLimData), np.nanmax(yLimData)]))

    ax.set_xlabel(xLabel, fontsize=20)
    ax.set_ylabel(yLabel, fontsize=20)
    ax.legend(loc='best', fontsize=12)

    if xLim is None:
        xLim = [data['minFreq'], data['maxFreq']]
    ax.set_xlim(xLim)

    if yLim is None:
        yLim = [np.nanmin(yLims[:,0]), np.nanmax(yLims[:,1])]
        margin = (yLim[1] - yLim[0]) / 20
        yLim = [yLim[0]-margin, yLim[1]+margin]
    ax.set_ylim(yLim)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=8))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(2,5),
                                                 numticks=5))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
    
    return fig

def _curve_data_extractor_freq(sigObjs):
    """
    Extracts data from all curves from each SignalObj

    Parameter (default), (type):
    -----------------------------

        * sigObjs (), (list):
            a list with SignalObjs for curve data extraction

    Return (type):
    --------------

     * curveData (), (list):
            a list with dictionaries containing the information about
            each curve. Needed keys for frequency plot:

            - 'x', (), (ndarray): time axis;
            - 'y', (), (ndarray): linear effective value axis;
            - 'label', (), (str): curve label;
            - 'dBRef', (), (float): decibel scale reference;
            - 'minFreq', (), (float): SignalObj's minimum analysed frequency
            - 'maxFreq', (), (float): SignalObj's maximum analysed frequency

            >>> curveData = [{'x':x, 'y':y, 'label':'my beautiful curve',
                            'dBRef':2e-5, 'minFreq':20, 'maxFreq':20000}]
    """
    curveData = []
    for sigObj in sigObjs:
        minFreq = sigObj.freqMin
        maxFreq = sigObj.freqMax
        x = sigObj.freqVector
        for chIndex in range(sigObj.numChannels):
            chNum = sigObj.channels.mapping[chIndex]
            unitData = '[{} ref.: {} {}]'.format(sigObj.channels[chNum].dBName,
                                                 sigObj.channels[chNum].dBRef,
                                                 sigObj.channels[chNum].unit)
            label = '{} {}'.format(sigObj.channels[chNum].name, unitData)
            y = sigObj.freqSignal[:, chIndex]
            dBRef = sigObj.channels[chNum].dBRef
            curveData.append({
                'label':label,
                'x':x,
                'y':y,
                'dBRef':dBRef,
                'minFreq':minFreq,
                'maxFreq':maxFreq
            })
    return curveData

def bars(analyses, xLabel, yLabel, yLim, title, decimalSep, barWidth,
         errorStyle):
    """Plot the analysis data in fractinal octave bands.

    Parameters (default), (type):
    -----------------------------

       * analyses (), (list):
            a list with Analyses.
        
        * xLabel ('Time [s]'), (str):
            x axis label.

        * yLabel ('Amplitude'), (str):
            y axis label.

        * yLim (), (list):
            inferior and superior limits.

            >>> yLim = [-100, 100]

        * title (), (str):
            plot title

        * decimalSep (','), (str):
            may be dot or comma.

            >>> decimalSep = ',' # in Brazil

        * barWidth (0.75), float:
            width of the bars from one fractional octave band. 
            0 < barWidth < 1.

        * errorStyle ('standard'), str:
            error curve style. May be 'laza' or None/'standard'.

    Return:
    --------

        matplotlib.figure.Figure object.
    """
    if errorStyle == 'laza':
        ecolor='limegreen'
        elinewidth=20
        capsize=0
        capthick=0
        alpha=.60
    else:
        ecolor='black'
        elinewidth=5
        capsize=10
        capthick=5
        alpha=.60

    if decimalSep == ',':
        locale.setlocale(locale.LC_NUMERIC, 'pt_BR.UTF-8')
        plt.rcParams['axes.formatter.use_locale'] = True
    elif decimalSep =='.':
        locale.setlocale(locale.LC_NUMERIC, 'C')
        plt.rcParams['axes.formatter.use_locale'] = False
    else:
        raise ValueError("'decimalSep' must be the string '.' or ','.")

    if xLabel is None:
        xLabel = 'Frequency bands [Hz]'

    if yLabel is None:
        yLabel = 'Modulus'
    
    if title is None:
        title = ''
    
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes([0.10, 0.21, 0.88, 0.72], polar=False,
                        projection='rectilinear', xscale='linear')
    ax.set_snap(True)
    ax.set_xlabel(xLabel, fontsize=20)
    ax.set_ylabel(yLabel, fontsize=20)
    plt.title(title, fontsize=20)

    yLims = np.array([np.nan, np.nan], dtype=float, ndmin=2)

    curveData = _curve_data_extractor_bars(analyses)
    
    dataSetLen = len(curveData)
    # checking negative plot necessity
    minVal = np.inf
    negativeCounter = 0
    for data in curveData:
        # minimum value
        newMinVal = np.amin(data['data'])
        marginData = \
            [value for value in data['data'] if not np.isinf(value)]
        margin = np.abs((np.nanmax(marginData) -
            np.nanmin(marginData)) / 20)
        newMinVal = newMinVal - margin
        newMinVal += np.sign(newMinVal)
        if newMinVal < minVal:
                minVal = newMinVal
        # negative counter
        for value in data['data']:
            if value < 0:
                negativeCounter += 1

    for dtIdx, data in enumerate(curveData):
        errorLabel = \
            'Error' if data['errorLabel'] is None else data['errorLabel']
        label = \
            '' if data['dataLabel'] is None else data['dataLabel']

        fbar = np.arange(0,len(data['data']))

        if negativeCounter < (len(data['data'])*dataSetLen)//2:
            minVal = 0
        
        ax.bar(fbar + barWidth*dtIdx/dataSetLen,
            -minVal + data['data'],
            width=barWidth/dataSetLen, label=label, zorder=-1)

        if data['error'] is not None:
            ax.errorbar(fbar + barWidth*dtIdx/dataSetLen,
                    -minVal + data['data'],
                    yerr=data['error'], fmt='none',
                    ecolor=ecolor,
                    elinewidth=elinewidth*barWidth/dataSetLen,
                    capsize=capsize*barWidth/dataSetLen,
                    capthick=capthick*barWidth/dataSetLen,
                    zorder=0,
                    fillstyle='full',
                    alpha=alpha,
                    barsabove=True,
                    label=errorLabel)

        error = data['error'] if data['error'] is not None else 0

        ylimInfData = -minVal + data['data'] - error
        ylimInfData = \
            [value for value in ylimInfData if not np.isinf(value)]
        ylimInfMargin = \
            np.abs((np.nanmax(ylimInfData) - np.nanmin(ylimInfData)) / 20)
        ylimInf = np.nanmin(ylimInfData) - ylimInfMargin
    
        ylimSupData = -minVal + data['data'] + error
        ylimSupData = \
            [value for value in ylimSupData if not np.isinf(value)]
        ylimSupMargin = \
            np.abs((np.nanmax(ylimSupData) - np.nanmin(ylimSupData)) / 20)
        ylimSup = np.nanmax(ylimSupData) + ylimSupMargin

        yLims = \
            np.vstack((yLims,
                        [ylimInf, ylimSup]))

    if yLim is None:
        yLim = [np.nanmin(yLims[:,0]), np.nanmax(yLims[:,1])]
    ax.set_ylim(yLim)
    ax.autoscale(enable=True, axis='x', tight=True)
    
    ax.grid(color='gray', linestyle='-.', linewidth=0.4)

    ax.set_xticks(fbar+barWidth*(dataSetLen-1)/dataSetLen-
        barWidth*(dataSetLen-1)/(2*dataSetLen))
    xticks = data['bands']
    ax.set_xticklabels(['{:n}'.format(tick) for tick in xticks],
                        rotation=45, fontsize=14)

    def neg_tick(x, pos):
        return '%.1f' % (x + minVal if x != minVal else 0)
    formatter = ticker.FuncFormatter(neg_tick)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=8))
    ax.yaxis.set_major_formatter(formatter)

    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)        
    
    ax.legend(loc='best', fontsize=13, framealpha=0.6)
    
    return fig

def _curve_data_extractor_bars(analyses):
    """
    Extracts data from from each Analysis

    Parameter (default), (type):
    -----------------------------

        * analyses (), (list):
            a list with Analysis for curve data extraction

    Return (type):
    --------------

        * curveData (), (list):
            a list with dictionaries containing the information about each
            curve. Needed keys for analysis plot:

            - 'bands', (), (ndarray): fractional octave bands;
            - 'data', (), (ndarray): magnitude axis;
            - 'dataLabel', (), (str): data label;
            - 'error', (), (ndarray): error for each fractional octave band;
            - 'errorLabel', (), (str): error label;
            
            >>> curveData = [{'bands':[100, 200, 400], 'data':[1, 5 ,7],
                            'dataLabel':'my precious data',
                            'error':[2e-5, 1, 3],
                            'errorLabel':'my mistakes'}]
    """
    curveData = []
    for analysis in analyses:
        bands = analysis.bands
        anData = analysis.data
        dataLabel = analysis.dataLabel
        error = analysis.error
        errorLabel = analysis.errorLabel
        curveData.append({
            'bands':bands,
            'data':anData,
            'dataLabel':dataLabel,
            'error':error,
            'errorLabel':errorLabel})
    return curveData

def spectrogram(sigObjs, winType, winSize,
                overlap, xLabel, yLabel, xLim, yLim,
                title, decimalSep):
    """Plots a signal spectrogram in frequency domain.

    Parameters (default), (type):
    -----------------------------

        * sigObjs (), (list):
            a list with SignalObjs.

        * winType (), (str):
            window type for the time slicing, defaults to 'hann'.

        * winSize (), (int):
            window size in samples

        * overlap (), (float):
            window overlap in %

        * xLabel ('Time [s]'), (str):
            x axis label.

        * yLabel ('Frequency [Hz]'), (str):
            y axis label.

        * yLim (), (list):
            inferior and superior frequency limits.

            >>> yLim = [20, 1000]

        * xLim (), (list):
            left and right time limits

            >>> xLim = [1, 3]

        * title (), (str):
            plot title

        * decimalSep (','), (str):
            may be dot or comma.

            >>> decimalSep = ',' # in Brazil

    Return:
    --------

        List of matplotlib.figure.Figure objects for each item in curveData.
    """

    if xLabel is None:
        xLabel = 'Time [s]'
    if yLabel is None:
        yLabel = 'Frequency [Hz]'

    figs = []
    curveData = _curve_data_extractor_spectrogram(sigObjs)
    for data in curveData:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_axes([0.10, 0.15, 0.93, 0.77], polar=False,
                        projection='rectilinear')
        ax.set_snap(False)

        _spectrogram, _specTime, _specFreq = \
            _calc_spectrogram(data['timeSignal'], data['timeVector'],
                              data['samplingRate'], overlap, winType,
                              winSize, data['dBRef'])

        pcmesh = ax.pcolormesh(_specTime, _specFreq, _spectrogram,
                            cmap=plt.jet(), vmin=-120)

        if xLim is None:
            xLim = (data['timeVector'][0], data['timeVector'][-1])
        ax.set_xlim(xLim)

        if yLim is None:
            yLim = (data['minFreq'], data['maxFreq'])
        ax.set_ylim(yLim)

        ax.yaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=8))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=10))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(14)        
        
        ax.set_xlabel(xLabel, fontsize=20)
        ax.set_ylabel(yLabel, fontsize=20)

        cbar = fig.colorbar(pcmesh)
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.set_ylabel(data['label'], fontsize=14)

    return figs

def _curve_data_extractor_spectrogram(sigObjs):
    """
    Extracts data from all curves from each SignalObj

    Parameter (default), (type):
    -----------------------------

        * sigObjs (), (list):
            a list with SignalObjs for curve data extraction

    Return (type):
    --------------

        * curveData (), (list):
                    a list with dictionaries containing the information about
                    each plot. Needed keys for spectrogram plot:

                    - 'timeVector', (), (ndarray): time axis;
                    - 'timeSignal', (), (ndarray): signal in time domain;
                    - 'samplingRate', (), (int): signal sampling rate;
                    - 'label', (), (str): curve label;
                    - 'dBRef', (), (float): decibel scale reference;
                    - 'minFreq', (), (float): SignalObj's minimum analysed frequency
                    - 'maxFreq', (), (float): SignalObj's maximum analysed frequency

                    >>> curveData = [{'timeVector':timeVector, 'timeSignal':timeSignal,
                                    'samplingRate':samplingRate', label':'my plotz',
                                    'dBRef':2e-5, 'minFreq':20, 'maxFreq':20000}]
    """
    curveData = []
    for sigObj in sigObjs:
        minFreq = sigObj.freqMin
        maxFreq = sigObj.freqMax
        timeVector = sigObj.timeVector
        samplingRate = sigObj.samplingRate

        for chIndex in range(sigObj.numChannels):
            chNum = sigObj.channels.mapping[chIndex]
            unitData = '[{} ref.: {} {}]'.format(sigObj.channels[chNum].dBName,
                                                sigObj.channels[chNum].dBRef,
                                                sigObj.channels[chNum].unit)
            label = '{} {}'.format(sigObj.channels[chNum].name, unitData)
            timeSignal = sigObj.timeSignal[:, chIndex]
            dBRef = sigObj.channels[chNum].dBRef
            curveData.append({
                'timeVector':timeVector,
                'timeSignal':timeSignal,
                'samplingRate':samplingRate,
                'label':label,
                'dBRef':dBRef,
                'minFreq':minFreq,
                'maxFreq':maxFreq
            })
    return curveData
    
def _calc_spectrogram(timeSignal, timeVector, samplingRate, overlap, winType,
                      winSize, dBRef):
    window = eval('ss.windows.' + winType)(winSize)
    nextIdx = int(winSize*overlap)
    rng = int(timeSignal.shape[0]/winSize/overlap - 1)
    _spectrogram = np.zeros((winSize//2 + 1, rng))
    _specFreq = np.linspace(0, samplingRate//2, winSize//2 + 1)
    _specTime = np.linspace(0, timeVector[-1], rng)

    for N in range(rng):
        try:
            strIdx = N*nextIdx
            endIdx = winSize + N*nextIdx
            sliceAudio = window*timeSignal[strIdx:endIdx]
            sliceFFT = np.fft.rfft(sliceAudio, axis=0)
            sliceMag = np.absolute(sliceFFT) * (2/sliceFFT.size)
            _spectrogram[:, N] = 20*np.log10(sliceMag/dBRef)

        except IndexError:
            sliceAudio = timeSignal[-winSize:]
            sliceFFT = np.fft.rfft(sliceAudio, axis=0)
            sliceMag = np.absolute(sliceFFT) * (2/sliceFFT.size)
            _spectrogram[:, N] = 20*np.log10(sliceMag)

    return _spectrogram, _specTime, _specFreq
