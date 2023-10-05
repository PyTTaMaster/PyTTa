"""
Data visualization module.

This plot module receives organized raw data (numpy arrays, lists and
strings) and plot it through its functions. It is used by the function,
signal, and analysis modules.

Available functions:

    >>> plot.time(curveData, xLabel, yLabel, yLim, xLim, title, decimalSep)
    >>> plot.freq(curveData, smooth, xLabel, yLabel, yLim, xLim, title,
                  decimalSep)
    >>> plot.bars(curveData, xLabel, yLabel, yLim, title, decimalSep,
                  barWidth, errorStyle)

For further information check the function especific documentation.

"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cmx
import matplotlib
import locale
import numpy as np
import scipy.signal as ss
import copy as cp
from plotly import io
io.renderers.default = 'browser'


def time(sigObjs, xLabel, yLabel, yLim, xLim, title, decimalSep, timeUnit):
    """
    Plots a signal in time domain.

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

         * timeUnit ('s'), (str):
            'ms' or 's'.

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

    if timeUnit in ['s', 'seconds', 'S']:
        timeScale = 1
        timeUnit = 's'
    elif timeUnit in ['ms', 'milliseconds', 'mseconds', 'MS']:
        timeScale = 1000
        timeUnit = 'ms'
    else:
        raise ValueError("'timeUnit' must be the string 's' or 'ms'.")

    if xLabel is None:
        xLabel = 'Time ['+timeUnit+']'
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
        ax.plot(timeScale*data['x'], data['y'], label=data['label'])
        yLimData = timeScale*cp.copy(data['y'])
        yLimData[np.isinf(yLimData)] = 0
        yLims = \
            np.vstack((yLims, [np.nanmin(yLimData), np.nanmax(yLimData)]))
        xLims = \
            np.vstack((xLims, [data['x'][0], data['x'][-1]]))

    ax.set_xlabel(xLabel, fontsize=16)
    ax.set_ylabel(yLabel, fontsize=16)
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
    Extracts data from all curves from each SignalObj.

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


def time_dB(sigObjs, xLabel, yLabel, yLim, xLim, title, decimalSep, timeUnit):
    """
    Plots a signal in decibels in time domain.

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

        * timeUnit ('s'), (str):
            'ms' or 's'.


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

    if timeUnit in ['s', 'seconds', 'S']:
        timeScale = 1
        timeUnit = 's'
    elif timeUnit in ['ms', 'milliseconds', 'mseconds', 'MS']:
        timeScale = 1000
        timeUnit = 'ms'
    else:
        raise ValueError("'timeUnit' must be the string 's' or 'ms'.")

    if xLabel is None:
        xLabel = 'Time ['+timeUnit+']'
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
        with np.errstate(divide='ignore'):
            dBSignal = 20*np.log10(np.abs(data['y']/np.sqrt(2))/data['dBRef'])
        ax.plot(timeScale*data['x'], dBSignal, label=data['label'])
        yLimData = cp.copy(dBSignal)
        yLimData[np.isinf(yLimData)] = 0
        yLims = \
            np.vstack((yLims, [np.nanmin(yLimData), np.nanmax(yLimData)]))
        xLims = \
            np.vstack((xLims, [timeScale*data['x'][0],
                               timeScale*data['x'][-1]]))

    ax.set_xlabel(xLabel, fontsize=16)
    ax.set_ylabel(yLabel, fontsize=16)
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
    Extracts data from all curves from each SignalObj.

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
    """
    Plots a signal decibel magnitude in frequency domain.

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
        yLabel = 'Magnitude [dB]'
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
        with np.errstate(divide='ignore'):
            dBSignal = 20*np.log10(np.abs(data['y'])/data['dBRef'])

        if smooth:
            dBSignal = ss.savgol_filter(dBSignal, 31, 3)

        ax.semilogx(data['x'], dBSignal, label=data['label'])

        yLimData = dBSignal
        yLimData[np.isinf(yLimData)] = np.nan
        yLims = \
            np.vstack((yLims, [np.nanmin(yLimData), np.nanmax(yLimData)]))

    ax.set_xlabel(xLabel, fontsize=16)
    ax.set_ylabel(yLabel, fontsize=16)
    ax.legend(loc='best', fontsize=12)

    if yLim is None:
        yLim = [np.nanmin(yLims[:,0]), np.nanmax(yLims[:,1])]
        margin = (yLim[1] - yLim[0]) / 20
        yLim = [yLim[0]-margin, yLim[1]+margin]
    ax.set_ylim(yLim)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=8))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1,2),
                                                 numticks=5))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
    freq_oct_list = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    freq_oct_list_str = [str(item) for item in freq_oct_list]
    ax.set_xticks(freq_oct_list)
    ax.set_xticklabels(freq_oct_list_str)
    if xLim is None:
        xLim = [data['minFreq'], data['maxFreq']]
    ax.set_xlim(xLim)

    return fig


def _curve_data_extractor_freq(sigObjs):
    """
    Extracts data from all curves from each SignalObj.

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


def bars(analyses, xLabel, yLabel, yLim, xLim, title, decimalSep, barWidth,
         errorStyle, forceZeroCentering, overlapBars, color):
    """
    Plot the analysis data in fractinal octave bands.

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

        * xLim (), (list):
            bands limits.

            >>> xLim = [100, 10000]

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

        * forceZeroCentering ('False'), bool:
            force centered bars at Y zero.

        * overlapBars ('False'), bool:
            overlap bars. No side by side bars of different data.

        * color (None), list:
            list containing the color of each Analysis.


    Return:
    --------

        matplotlib.figure.Figure object.
    """
    if isinstance(color, list):
        if len(color) != len(analyses):
            raise ValueError("'color' must be a list with the same number of " +
                             "elements than analyses.")
    else:
        color = [None for an in analyses]

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
    ax.set_xlabel(xLabel, fontsize=16)
    ax.set_ylabel(yLabel, fontsize=16)
    plt.title(title, fontsize=20)

    yLims = np.array([np.nan, np.nan], dtype=float, ndmin=2)

    curveData = _curve_data_extractor_bars(analyses)

    if xLim is not None:
        xLimIdx0 = np.where(curveData[0]['bands']<=xLim[0])
        xLimIdx0 = 0 if len(xLimIdx0[0]) == 0 else xLimIdx0[0][-1]
        xLimIdx1 = np.where(curveData[0]['bands']>=xLim[1])
        xLimIdx1 = len(curveData[0]['bands'])-1 \
                          if len(xLimIdx1[0]) == 0 else xLimIdx1[0][0]
        xLimIdx = [xLimIdx0, xLimIdx1]
    else:
        xLimIdx = [0, len(curveData[0]['data'])]

    dataSetLen = len(curveData)
    # checking negative plot necessity
    minVal = np.inf
    negativeCounter = 0
    allBands = []
    for data in curveData:
        # merge bands
        list_1 = allBands
        list_2 = data['bands']
        set_1 = set(list_1)
        set_2 = set(list_2)
        list_2_items_not_in_list_1 = list(set_2 - set_1)
        allBands = list_1 + list_2_items_not_in_list_1
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
    # Sort merged bands
    sortedIdxs = sorted(range(len(allBands)), key=lambda k: allBands[k])
    sortedList = []
    for idx in sortedIdxs:
        sortedList.append(allBands[idx])
    allBands = np.array(sortedList)

    fbar = np.arange(0,len(allBands))
    lowerBand = 0
    higherBand = np.inf
    for dtIdx, data in enumerate(curveData):
        errorLabel = \
            'Error' if data['errorLabel'] is None else data['errorLabel']
        label = \
            '' if data['dataLabel'] is None else data['dataLabel']

        # fbar = np.arange(0,len(data['data']))
        fbarRange = [np.where(allBands<=data['bands'][0])[0][-1],
                     np.where(allBands>=data['bands'][-1])[0][0]]
        if allBands[fbarRange[0]] > lowerBand:
            lowerBand = allBands[fbarRange[0]]
        if allBands[fbarRange[1]] > higherBand:
            higherBand = allBands[fbarRange[1]]
        if negativeCounter < (len(data['data'])*dataSetLen)//2 or \
            forceZeroCentering:
            minVal = 0
        if overlapBars:
            ax.bar(fbar[fbarRange[0]:fbarRange[1]+1],
                -minVal + data['data'],
                width=barWidth, label=label, zorder=-1,
                color=color[dtIdx])
        else:
            ax.bar(fbar[fbarRange[0]:fbarRange[1]+1] + dtIdx*barWidth/dataSetLen,
                -minVal + data['data'],
                width=barWidth/dataSetLen, label=label, zorder=-1,
                color=color[dtIdx])

        if data['error'] is not None:
            ax.errorbar(fbar[fbarRange[0]:fbarRange[1]+1] + dtIdx*barWidth/dataSetLen,
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
            [value for value in ylimInfData[xLimIdx[0]:xLimIdx[1]] if not np.isinf(value)]
        ylimInfMargin = \
            np.abs((np.nanmax(ylimInfData) - np.nanmin(ylimInfData)) / 20)
        ylimInf = np.nanmin(ylimInfData) - ylimInfMargin

        ylimSupData = -minVal + data['data'] + error
        ylimSupData = \
            [value for value in ylimSupData[xLimIdx[0]:xLimIdx[1]] if not np.isinf(value)]
        ylimSupMargin = \
            np.abs((np.nanmax(ylimSupData) - np.nanmin(ylimSupData)) / 20)
        ylimSup = np.nanmax(ylimSupData) + ylimSupMargin

        if ylimSup == ylimInf:
            if ylimSup > 0:
                ylimInf = 0
            else:
                ylimSup = 0

        yLims = \
            np.vstack((yLims,
                        [ylimInf, ylimSup]))

    if yLim is None:
        yLim = [np.nanmin(yLims[:,0]), np.nanmax(yLims[:,1])]
    ax.set_ylim(yLim)

    ax.grid(color='gray', linestyle='-.', linewidth=0.4)

    # ax.set_xticks(fbar+barWidth*(dataSetLen-1)/dataSetLen-
    #     barWidth*(dataSetLen-1)/(2*dataSetLen))
    ax.set_xticks(fbar+barWidth/2-barWidth/(dataSetLen*2))
    # ax.set_xticks(fbar)
    xticks = allBands
    ax.set_xticklabels(['{:n}'.format(tick) for tick in xticks],
                        rotation=45, fontsize=14)

    ax.autoscale(enable=True, axis='x', tight=True)

    # def neg_tick(x, pos):
    #     return '%.1f' % (x + minVal if x != minVal else 0)
    # formatter = ticker.FuncFormatter(neg_tick)
    formatter = ticker.ScalarFormatter()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=8))
    ax.yaxis.set_major_formatter(formatter)

    if xLim is not None:
        # ax.set_xlim([xLimIdx[0],
        ax.set_xlim([xLimIdx[0]-barWidth/(dataSetLen*2),
                     xLimIdx[1]+barWidth/(dataSetLen*2)+(dataSetLen-1)*barWidth/dataSetLen])

    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)

    ax.legend(loc='best', fontsize=13, framealpha=0.6)

    return fig


def _curve_data_extractor_bars(analyses):
    """
    Extracts data from from each Analysis.

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
    """
    Plots a signal spectrogram in frequency domain.

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
        figs.append(fig)
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

        ax.set_xlabel(xLabel, fontsize=16)
        ax.set_ylabel(yLabel, fontsize=16)

        cbar = fig.colorbar(pcmesh)
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.set_ylabel(data['label'], fontsize=14)

    return figs


def _curve_data_extractor_spectrogram(sigObjs):
    """
    Extracts data from all curves from each SignalObj.

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


def waterfall(sigObjs, step=2 ** 9, n=2 ** 13, fmin=None, fmax=None, pmin=None,
              pmax=None, tmax=None, xaxis='linear', time_tick=None,
              freq_tick=None, mag_tick=None, tick_fontsize=None, fpad=1,
              delta=60, dBref=2e-5, fill_value='pmin', fill_below=True,
              overhead=3, winAlpha=0, plots=['waterfall'], show=True,
              cmap='jet', alpha=[1, 1], saveFig=False, figRatio=[1, 1, 1],
              figsize=(950, 950), camera=[2, 1, 2]):
    """
       Plots a signal in dB and its decay in the time domain in a 3D waterfall plot.

       Parameters (default), (type):
       ------------------------------

           * sigObjs (), (list):
               a list with SignalObjs.

           * receivers ([0]), (list):
               list containing the index of the receivers.

           * step (2**9), (int):
               time steps in samples.

           * n (2**14), (int):
               FFT size in samples.

           * fmin (), (int):
               left limit.

           * fmax (), (int):
               right limit.

           * pmin (), (int):
               inferior limit.

           * pmax (), (int):
               superior limit.

           * tmax (), (int):
               time limit.

           * xaxis ('linear'), (str):
               x axis scale.

               >>> xaxis = 'linear' # log scale
               >>> xaxis = 'log' # log scale

           * time_tick (), (float):
               time axis tick interval.

           * freq_tick (), (float):
               frequency axis tick interval.

           * mag_tick (), (float):
               magnitude axis tick interval.

           * tick_fontsize (), (float):
               fontsize of the X, Y and Z axis ticks.

           * fpad (1), (int):
               frequency pad for inferior and superior limits.

           * delta (60), (int):
               time decay delta from the superior limit.

           * dBred (2e-5), (float):
               dB scale referece.

           * fill_value ('pmin'), (str):
               fill option for the base of the plot.

               >>> fill_value = 'NaN' # transparent
               >>> fill_value = 'pmin' # solid

           * fill_below (True), (bool):
               option to chose to fill the area below the 3D curve or not.

           * overhead (3), (int):
               overhead above pmax to be displayed in the Z axis.

           * winAlpha (0), (float):
               alpha value of the Tukey window.

               >>> winAlpha = 0 # rectangular
               >>> winAlpha = 1 # tukey

           * plots (['waterfall']), (list):
               list containint the plots that will be displayed.

           * show (True), (bool):
               option to display the graph in the screen or not.

           * cmap ('jet'), (str):
               colormap that will be used to color the curves.

           * alpha ([1, 1]), (list):
               transparency of curve and filling. 1 is solid, 0 is transparent.

           * saveFig (False), (bool or str):
               option to save the plot as a .png file - the value will be used as part of the filename.

                   >>> saveFig = 'my_beatiful_project'

           * figRatio ([1, 1, 1]), (list):
               list containing float values for the ratios of the X, Y and Z axis.

           * figsize (950, 950), (tuple):
               width and height of the plot in pixels.

           * camera ([2, 1, 2]), (list):
               3D camera position - is used to save the plot and for the initial view.

       Return:
       --------

           ()
       """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.fftpack import fft
    import scipy.signal as ss
    import more_itertools

    figs = []
    curveData = _curve_data_extractor_waterfall(sigObjs)
    for data in curveData:

        ht = data['ht']  # Impulse Response
        fs = data['samplingRate']

        if fmin is None and fmax is None:
            xmin, idx_min = _find_nearest(data['freq'], data['minFreq'])
            xmax, idx_max = _find_nearest(data['freq'], data['maxFreq'])
        else:
            xmin, idx_min = _find_nearest(data['freq'], fmin)
            xmax, idx_max = _find_nearest(data['freq'], fmax)

        if xaxis == 'linear':
            x_range = [xmin - fpad, xmax + fpad]
        elif xaxis == 'log':
            x_range = [np.log10(xmin - fpad), np.log10(xmax + fpad)]

        freq = np.linspace(0, (n - 1) * fs / n, n)  # Frequency vector of the FFT
        xmin_FFT, idx_min_FFT = _find_nearest(freq, xmin)  # Freq. range
        xmax_FFT, idx_max_FFT = _find_nearest(freq, xmax)
        freq = freq[idx_min_FFT - fpad: idx_max_FFT + 1 + fpad]  # Crop to freq range

        # Display selected values
        print(f'Time steps (step): {((step - 1) / fs) * 10e2:.2f} [ms] | {step} [samples]')
        print(f'FFT size (n): {freq[1] - freq[0]:.2f} [Hz] | {n:.0f} [samples]')

        # Apply rolling window
        sliced = np.asarray(list(more_itertools.windowed(ht, n=n, step=step, fillvalue=0)))
        window = np.asarray(
            [ss.tukey(n, alpha=winAlpha) for s in sliced])  # Tukey window (alpha=0 is a rectangular window)
        windowed = sliced * window  # Apply window
        windowedFFT = abs(2 / n * fft(windowed))  # Apply FFT
        windowedFFT = windowedFFT[:, idx_min_FFT - fpad: idx_max_FFT + 1 + fpad]  # Crop the FFT
        windowedFFTdB = 20 * np.log10(abs(windowedFFT) / dBref)  # Spply log scaling
        windowedFFTdBNaN = np.copy(windowedFFTdB)  # Copies to be modified
        windowedFFTdBpmin = np.copy(windowedFFTdB)

        N = len(ht)  # Number of samples in the impulse response
        time_steps = np.linspace(0, (N - 1) / fs, len(sliced))  # Time steps in seconds
        if tmax is None:
            tmax = max(time_steps)
            idx_tmax = len(time_steps)
        else:
            tmax, idx_tmax = _find_nearest(time_steps, tmax)
        time_steps_crop = time_steps[0:idx_tmax]  # Crop to tmax

        if pmax is None:
            pmax = np.real(np.max(windowedFFTdB)) + overhead
        if pmin is None:
            pmin = pmax - delta - overhead

        for i in range(len(windowedFFTdBNaN)):
            windowedFFTdBNaN[i][
                np.where(windowedFFTdBNaN[i] < pmin)] = np.nan  # Fill with NaN for transparent "floor level"
            windowedFFTdBpmin[i][
                np.where(windowedFFTdBpmin[i] < pmin)] = pmin  # Fill with pmin values for colored "floor level"
        windowedFFTdBNaN[:, 0], windowedFFTdBNaN[:,
                                -1] = pmin, pmin  # Create drop in cuurve at beggining and end
        windowedFFTdBpmin[:, 0], windowedFFTdBpmin[:, -1] = pmin, pmin

        # Plotting
        specs = []
        subplot_titles = []
        row = 0
        fig = None
        if 'waterfall' in plots:
            row += 1
            specs.append([{"type": "mesh3d"}])
            subplot_titles.append(f'Waterfall')
            fig = make_subplots(
                rows=row, cols=1,
                specs=specs,
                vertical_spacing=0,
                subplot_titles=subplot_titles
            )

            # Setting curves
            X, Y = np.meshgrid(freq, time_steps_crop)
            Z = windowedFFTdBNaN if fill_value == 'NaN' else windowedFFTdBpmin

            # Setting colormap
            cm = plt.get_cmap(cmap)
            cs = [int(np.average(windowedFFTdBpmin[i])) for i in range(len(windowedFFTdBpmin))]
            cNorm = matplotlib.colors.Normalize(vmin=pmin, vmax=None)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
            col1 = 255
            col2 = 1

            for i in range(len(windowedFFTdB)):
                col = scalarMap.to_rgba(cs[i], alpha=alpha[0])
                # Plotting curves
                fig.add_trace(go.Scatter3d(x=[np.real(time_steps[i])] * len(Z[i]),
                                           y=np.real(freq),
                                           z=np.real(Z[i]),
                                           mode='lines', showlegend=False,
                                           hovertext=f'Time: {np.real(time_steps[i]):0.2f} [s]',
                                           marker=dict(size=3,
                                                       color=f"rgb({col[0] * col1 * col2},"
                                                             f"{col[1] * col1 * col2}, "
                                                             f"{col[2] * col1 * col2})",
                                                       opacity=alpha[0])),
                              row=row, col=1
                              )
                # Filling area below the curves (might lead to high display time, disable if needed)
                if fill_below:
                    for j in range(0, 2):
                        try:
                            verts, tri = _triangulate_curve(
                                [np.real(time_steps[i])] * len(np.real(Z[i][j::])),
                                np.real(freq)[j::],
                                np.real(Z[i][j::]),
                                pmin
                            )
                        except:
                            verts, tri = _triangulate_curve(
                                [np.real(time_steps[i])] * len(np.real(np.append(Z[i][j::],
                                                                                 pmin))),
                                np.real(np.append(freq[j::], freq[-1])),
                                np.real(np.append(Z[i][j::], pmin)),
                                pmin
                            )

                        x, y, z = verts.T
                        I, J, K = tri.T
                        fig.add_traces(go.Mesh3d(x=x, y=y, z=z,
                                                 i=I,
                                                 j=J,
                                                 k=K,
                                                 color=f"rgb({col[0] * col1},{col[1] * col1},{col[2] * col1})",
                                                 opacity=alpha[1])
                                       )

            # Setting figure layout
            fig.update_layout(
                scene_aspectmode='manual',
                scene_aspectratio=dict(x=figRatio[0], y=figRatio[1], z=figRatio[2]),
                scene_camera=dict(up=dict(x=0, y=0, z=1),
                                  center=dict(x=0, y=0, z=0),
                                  eye=dict(x=camera[0], y=camera[1], z=camera[2])
                                  ),
                scene=dict(
                    xaxis_title='Time [s]',
                    yaxis_title='Frequency [Hz]',
                    zaxis_title='Magnitude [dB]',
                    xaxis_color='black',
                    yaxis_color='black',
                    zaxis_color='black',
                    xaxis=dict(nticks=10,
                               dtick=time_tick,
                               range=[0, tmax + 0.1],
                               tickfont=dict(color='black', size=tick_fontsize),
                               backgroundcolor="rgba(0, 0, 0, 0)",
                               gridcolor="lightgrey",
                               showbackground=True,
                               zerolinecolor="lightgrey", ),
                    yaxis=dict(range=x_range, type=xaxis,
                               dtick=freq_tick, tickformat=".0f",
                               tickfont=dict(color='black', size=tick_fontsize),
                               backgroundcolor="rgba(0, 0, 0, 0)",
                               gridcolor="lightgrey",
                               showbackground=True,
                               zerolinecolor="lightgrey", ),
                    zaxis=dict(range=[pmin - 0.1, pmax],
                               dtick=mag_tick,
                               tickfont=dict(color='black', size=tick_fontsize),
                               backgroundcolor="rgba(0, 0, 0, 0)",
                               gridcolor="lightgrey",
                               showbackground=True,
                               zerolinecolor="lightgrey", ),
                ),
                font=dict(color='black', size=14),
                plot_bgcolor="rgba(0, 0, 0, 0)",
                paper_bgcolor="rgba(0, 0, 0, 0)",
                width=figsize[0], height=figsize[1],
                margin=dict(r=0, b=0, l=0, t=0),
            )

        # Saving plot as .png
        if saveFig and fig:
            import os

            directory = os.getcwd()
            fig.write_image(directory + os.sep + saveFig + '_waterfall.png', scale=5)
            print('Waterfall saved at ' + directory + os.sep + saveFig + '_waterfall.png')

        # Boolean to display plot
        if show and fig:
            fig.show()

        figs.append(fig)

    return figs


def _curve_data_extractor_waterfall(sigObjs):
    """
    Extracts data from all curves from each SignalObj.

    Parameter (default), (type):
    -----------------------------

        * sigObj (), (list):
            a list with SignalObjs for curve data extraction

    Return (type):
    --------------

        * curveData (list):
            A list with dictionaries containing the information about each
            curve. Keys for waterfall plot:

                - 'time', (), (ndarray): time vector;
                - 'freq', (), (ndarray): freq vector;
                - 'ht', (), (ndarray): impulsive response in time domain;
                - 'minFreq', (), (float): SignalObj's minimum frequency
                - 'maxFreq', (), (float): SignalObj's maximum frequency
                - 'samplingRate', (), (int): signal sampling rate;
                - 'label', (), (str): curve label;

                >>> curveData = [{'x':x, 'y':y, 'label':'my beautiful curve'}]
    """
    curveData = []
    for sigObj in sigObjs:
        samplingRate = sigObj.samplingRate
        for chIndex in range(sigObj.numChannels):
            chNum = sigObj.channels.mapping[chIndex]
            label = '{} [{}]'.format(sigObj.channels[chNum].name,
                                     sigObj.channels[chNum].unit)
            curveData.append({
                'time':sigObj.timeVector,
                'freq':sigObj.freqVector,
                'ht':sigObj.timeSignal[:, chIndex],
                'dBRef':sigObj.channels[chNum].dBRef,
                'minFreq':sigObj.freqMin,
                'maxFreq':sigObj.freqMax,
                'samplingRate':samplingRate,
                'label':label
            })
    return curveData


def _colored_lines(fig, ax, X, Y, Z, label, cmap='jet', bar=True):
    """
    Make a waterfall plot
    Input:
        fig,ax : matplotlib figure and axes to populate
        Z : n,m numpy array. Must be a 2d array even if only one line should be plotted
        X,Y : n,m array
    """
    # Set normalization to the same values for all plots
    norm = plt.Normalize(Z.min().min(), Z.max().max())
    # Check sizes to loop always over the smallest dimension
    n, m = Z.shape
    if n > m:
        X = X.T
        Y = Y.T
        Z = Z.T
        m, n = n, m

    for j in range(n):
        # reshape the X,Z into pairs
        points = np.array([X[j, :], Z[j, :]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        # Set the values used for colormapping
        lc.set_array((Z[j, 1:] + Z[j, :-1]) / 2)
        lc.set_linewidth(
            2
        )  # set linewidth a little larger to see properly the colormap variation
        _ = ax.add_collection3d(lc,
                                zs=(Y[j, 1:] + Y[j, :-1]) / 2,
                                zdir='y')  # add line to axes

    if bar:
        fig.colorbar(lc,orientation='vertical').set_label(label=label,
                                                          size=15)
    cbar_ax = fig.axes[-1]
    cbar_ax.tick_params(labelsize=12)


def _find_nearest(array, value):
    """
    Function to find closest frequency in frequency array.
    Returns closest value and position index.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def _polygon_under_graph(xlist, ylist, Pmin):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (xlist, ylist) line graph.  Assumes the xs are in ascending order.
    """
    ylist[0] = Pmin
    ylist[-1] = Pmin
    return [*zip(xlist, ylist)]


def _triangulate_curve(x, y, z, Pmin):
    """
    Creates triangles along the curve points
    """

    if len(x) != len(y) != len(z):
        raise ValueError("The  lists x, y, z, must have the same length")
    n = len(x)
    if n % 2:
        raise ValueError("The length of lists x, y, z must be an even number")
    pts3d = np.vstack((x, y, z)).T
    pts3dp = np.array([[x[2 * k + 1], y[2 * k + 1], Pmin] for k in range(1, n // 2 - 1)])
    pts3d = np.vstack((pts3d, pts3dp))

    # Triangulate the histogram bars:
    tri = [[0, 1, 2], [0, 2, n]]
    for k, i in zip(list(range(n, n - 3 + n // 2)), list(range(3, n - 4, 2))):
        tri.extend([[k, i, i + 1], [k, i + 1, k + 1]])
    tri.extend([[n - 3 + n // 2, n - 3, n - 2], [n - 3 + n // 2, n - 2, n - 1]])

    return pts3d, np.array(tri)
