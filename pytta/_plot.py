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
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import ListedColormap
import matplotlib.cm as cmx
from scipy.fftpack import fft
from scipy.signal.windows import tukey
from mpl_toolkits.mplot3d import axes3d
import matplotlib
import locale
import numpy as np
import scipy.signal as ss
import copy as cp
import decimal


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
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1,2),
                                                 numticks=5))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True))
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)

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


def waterfall(sigObjs, step=10, xLim:list=None,
              Pmin=20, Pmax=None, tmin=0, tmax=None, azim=-72, elev=14,
              cmap='jet', winPlot=False, waterfallPlot=True, fill=True,
              lines=False, alpha=1, figsize=(20, 8), winAlpha=0,
              removeGridLines=False, saveFig=False, bar=False, width=0.70,
              size=3, lcol=None, filtered=True):

    curveData = _curve_data_extractor_waterfall(sigObjs)
    figs = []
    for data in curveData:
        if xLim is None:
            xmin, idx_min = _find_nearest(data['freq'], data['minFreq'])
            xmax, idx_max = _find_nearest(data['freq'], data['maxFreq'])
        else:
            xmin, idx_min = _find_nearest(data['freq'], xLim[0])
            xmax, idx_max = _find_nearest(data['freq'], xLim[1])

        if filtered:
            ht = data['ht']
        else:
            ht = data['ht']
        t = data['time']
        fs = data['samplingRate']

        win_size = int(step * 10 ** -3 * fs) + 1
        win_list = [np.zeros(len(ht)) for win in range(int(len(ht) / (1 * win_size)))]
        ht_list = [np.zeros(len(ht)) for win in range(int(len(ht) / (1 * win_size)))]
        fft_list = []
        fft_list_dB = []
        time_steps = []
        df = data['freq'][-1] - data['freq'][-2]
        freq_FFT = np.linspace(0, len(ht) / 2, num=int(len(ht) / df / 2))  # Frequency vector for the FFT

        for _, i in zip(win_list, range(len(win_list))):
            win_list[i][i * win_size::] = ss.windows.tukey(int(len(ht) - i * win_size),
                                                alpha=winAlpha)  # Alpha=0 is rectangular window
            time_steps.append(i * win_size / fs)

        for _ht in range(len(ht_list)):
            for i in range(len(ht)):
                ht_list[_ht][i] = ht[i] * win_list[_ht][i]
            _fft = 2 / len(ht_list[_ht]) * np.fft.fft(ht_list[_ht])
            _fft = _fft[0:int(len(ht_list[_ht]) / 2)]
            fr = _fft[idx_min:idx_max + 1]
            if np.mean(20 * np.log10(np.abs(fr) / data['dBRef'])) >= Pmin:
                fft_list.append(fr)
                fft_list_dB.append(20 * np.log10(np.abs(fr) / data['dBRef']))
            else:
                fft_list.append(np.ones(len(fr)) * Pmin)
                fft_list_dB.append(np.ones(len(fr)) * Pmin)

        for fr in fft_list_dB:
            #         fr[0], fr[-1] = Pmin, Pmin
            fr[-1] = Pmin  # Set last value to zero to create vertical line
            fr[fr < Pmin] = Pmin  # Remove values before minimum desired sound pressure level

        if winPlot:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(t, ht)
            ax.set_xlabel('Time [s]', fontsize=20)
            ax.set_ylabel('Amplitude [Pa]', fontsize=20)
            ax.set_xlim([0, (len(ht) - 1) / fs])
            ax.grid()
            ax.tick_params(labelsize=17)
            for i in range(len(fft_list)):
                ax.plot(t, win_list[i] * max(ht))
                ax.plot(t, ht_list[i])
            fig.tight_layout()

        if waterfallPlot:
            # Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
            fig = plt.figure(figsize=plt.figaspect(width) * size)
            ax = fig.gca(projection='3d')

            x = freq_FFT[idx_min:idx_max + 1]

            if tmax is None:
                y = np.asarray(time_steps)
            else:
                tmax, idx_tmax = _find_nearest(time_steps, tmax)
                y = np.asarray(time_steps[0:idx_tmax])
            X, Y = np.meshgrid(x, y)
            Z = np.asarray(fft_list_dB)

            # Set background to be transparent
            if removeGridLines:
                # make the grid lines transparent
                ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
                ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
                ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.xaxis.set_tick_params(labelsize=15)
            ax.yaxis.set_tick_params(labelsize=15)
            ax.zaxis.set_tick_params(labelsize=15)
            # ax.xaxis._set_scale('log')

            # Changing color of lines and labels
            if lcol is not None:
                plt.rcParams['xtick.color'] = lcol
                plt.rcParams['ytick.color'] = lcol
                plt.rcParams['axes.labelcolor'] = lcol
                plt.rcParams['axes.edgecolor'] = lcol
                ax.xaxis._axinfo["grid"]['color'] = lcol
                ax.yaxis._axinfo["grid"]['color'] = lcol
                ax.zaxis._axinfo["grid"]['color'] = lcol

            # Labels and limits
            ax.set_xlabel('Frequency [Hz]', fontsize=18, labelpad=20)
            ax.set_xlim3d(xmin, xmax)
            if tmax is None:
                ax.set_ylabel('Time [s]', fontsize=18, labelpad=15)
                ax.set_ylim3d(max(data['time']), tmin)
            else:
                ax.set_ylabel('Time [s]', fontsize=18, labelpad=15)
                ax.set_ylim3d(tmax, tmin)
            ax.set_zlabel('SPL [dB]', fontsize=18, labelpad=12)
            if Pmax is None:
                ax.set_zlim3d(Pmin, max(fft_list_dB[0]) + 10)
            else:
                ax.set_zlim3d(Pmin, Pmax + 10)

            # Generate waterfall plot
            if lines:
                _colored_lines(fig, ax, X, Y, Z, label=data['label'],
                               cmap=cmap, bar=bar)
            if fill:  # Fills the area bellow the curves
                # Make verts a list, verts[i] will be a list of (x,y) pairs defining polygon i
                verts = []
                # Set up the x sequence
                xs = freq_FFT[idx_min:idx_max + 1]
                if tmax is None:
                    zs = time_steps
                else:
                    tmax, idx_tmax = _find_nearest(time_steps, tmax)
                    zs = time_steps[0:idx_tmax]

                for i in range(len(zs)):
                    ys = fft_list_dB[i]
                    verts.append(_polygon_under_graph(xs, ys, Pmin))

                cm = plt.get_cmap(cmap)
                if tmax is None:
                    cs = [np.average(fft_list_dB[i]) for i in range(len(fft_list_dB))]
                else:
                    cs = [np.average(fft_list_dB[i]) for i in range(len(fft_list_dB[0:idx_tmax]))]
                cNorm = matplotlib.colors.Normalize(vmin=Pmin, vmax=Pmax)
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
                poly = PolyCollection(verts,
                                        facecolors=scalarMap.to_rgba(cs),
                                        edgecolors=0.5 * scalarMap.to_rgba(cs),
                                        alpha=alpha,
                                        linewidth=3)
                scalarMap.set_array(cs)
                if bar is True:
                    fig.colorbar(scalarMap,
                                    orientation='vertical',
                                    fraction=0.025).set_label(label='SPL [dB]',
                                                            size=15)
                    cbar_ax = fig.axes[-1]
                    cbar_ax.tick_params(labelsize=14)
                ax.add_collection3d(poly, zs=zs, zdir='y')

            ax.view_init(azim=azim, elev=elev)
            fig.tight_layout()

        # if saveFig:
        #     fig.savefig(self.folder + '\\' + saveFig + '_waterfall.png',
        #                 dpi=300,
        #                 transparent=True)
        plt.show()
        if lcol is not None:
            plt.rcParams.update(matplotlib.rcParamsDefault)
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
