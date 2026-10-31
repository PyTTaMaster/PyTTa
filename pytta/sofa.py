"""
PyTTa SOFA input and output:
----------------------------

Read and write impulse responses in the Spatially Oriented Format for
Acoustics (SOFA), standardized as AES69-2022 (SOFA version 2.1).

A SOFA file is a netCDF-4 file, which is an HDF5 file whose dimensions are
HDF5 dimension scales. This module uses h5py only and writes the netCDF
dimension metadata itself, so the files open in netCDF tools (MATLAB ncread
and ncinfo, the SOFA Toolbox, netCDF4-python).

Supported conventions for writing (both at convention version 1.0):

    * SimpleFreeFieldHRIR: head-related impulse responses, source positions
      in spherical coordinates (azimuth in degree, elevation in degree,
      distance in metre);
    * GeneralFIR: impulse responses for general purposes, source positions
      in cartesian coordinates (metre).

Any convention that stores Data.IR (GeneralFIR, SimpleFreeFieldHRIR,
SingleRoomSRIR, ...) can be read.

Data mapping:

    SOFA stores Data.IR with dimensions M x R x N (measurements, receivers,
    samples). Here each measurement m is one SignalObj whose R channels are
    the receivers, so a SOFA file maps to a list of M SignalObj, in the
    order of SourcePosition. SignalObj keeps time data in float32; the
    float64 Data.IR is also returned by read as 'ir'.

Available functions:

    >>> pytta.sofa.read(path)
    >>> pytta.sofa.write(path, signal, source_positions)

For further information, check the function specific documentation.

"""

from datetime import datetime

import h5py
import numpy as np

from .classes import SignalObj

_DIMENSION_NAME = 'This is a netCDF dimension but not a netCDF variable.%10d'

_CONVENTIONS = {
    'SimpleFreeFieldHRIR': {
        'globals': {'DatabaseName': '', 'ListenerShortName': ''},
        'source': ('spherical', 'degree, degree, metre'),
        'ears': True,
    },
    'GeneralFIR': {
        'globals': {},
        'source': ('cartesian', 'metre'),
        'ears': False,
    },
}

_POSITIONS = {'source_position': 'SourcePosition',
              'listener_position': 'ListenerPosition',
              'receiver_position': 'ReceiverPosition',
              'emitter_position': 'EmitterPosition'}


def _text(value):
    """Decode an HDF5 attribute value into str when it is text."""
    if isinstance(value, h5py.Empty):
        return ''
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, np.ndarray) and value.dtype.kind in 'SO':
        return ''.join(_text(v) for v in value.ravel())
    return value


def read(path):
    """
    Read a SOFA file with impulse responses (Data.IR).

    Parameters
    ----------
    path : str
        Path to the .sofa file.

    Returns
    -------
    dict
        * 'signal': list of M SignalObj, one per measurement, with the R
          receivers as channels (float32, as SignalObj stores it; SignalObj
          transposes arrays with more channels than samples, so use 'ir'
          when R > N);
        * 'ir': Data.IR as a float64 array of shape (M, R, N);
        * 'fs': sampling rate in Hz (float);
        * 'delay': Data.Delay in samples, shape (I, R) or (M, R). The delay
          is returned and not applied to the signals;
        * 'source_position', 'listener_position', 'receiver_position',
          'emitter_position': dicts with 'values' (the array as stored,
          e.g. (M, 3) for SourcePosition and (R, 3, I) for
          ReceiverPosition), 'Type' and 'Units';
        * 'attributes': dict of the global attributes, without the
          'GLOBAL:' prefix.

    Raises
    ------
    ValueError
        If the file has no Data.IR, or a Data.SamplingRate that changes
        between measurements, or units other than hertz.

    """
    with h5py.File(path, 'r') as f:
        if 'Data.IR' not in f:
            raise ValueError("pytta.sofa.read only reads files with Data.IR.")
        ir = np.asarray(f['Data.IR'][()], dtype='float64')
        rates = np.unique(f['Data.SamplingRate'][()])
        units = _text(f['Data.SamplingRate'].attrs.get('Units', 'hertz'))
        if len(rates) != 1 or units.lower() != 'hertz':
            raise ValueError("Data.SamplingRate must be one value in hertz.")
        fs = float(rates[0])
        delay = np.asarray(f['Data.Delay'][()]) if 'Data.Delay' in f else None
        result = {'ir': ir, 'fs': fs, 'delay': delay,
                  'attributes': {k: _text(v) for k, v in f.attrs.items()
                                 if not k.startswith('_')}}
        for key, name in _POSITIONS.items():
            if name in f:
                result[key] = {'values': f[name][()],
                               'Type': _text(f[name].attrs.get('Type', '')),
                               'Units': _text(f[name].attrs.get('Units', ''))}
    signals = []
    for m in range(ir.shape[0]):
        signals.append(SignalObj(signalArray=ir[m].T, domain='time',
                                 samplingRate=fs))
    result['signal'] = signals
    return result


def write(path, signal, source_positions, convention='SimpleFreeFieldHRIR',
          **attributes):
    """
    Write impulse responses to a SOFA file (AES69-2022, SOFA 2.1).

    All mandatory global attributes and variables of the convention are
    written. Positions other than SourcePosition get the convention defaults:
    listener at the origin looking along +x with up +z, one emitter at the
    origin of the source, receivers at the origin of the listener (for
    SimpleFreeFieldHRIR with two receivers, the ears at +-0.09 m on y), and
    Data.Delay zero.

    Parameters
    ----------
    path : str
        Path of the .sofa file to create (overwritten if it exists).
    signal : SignalObj or list of SignalObj
        One SignalObj per measurement, all with the same sampling rate,
        number of samples N and number of channels R (the receivers).
    source_positions : array_like, shape (M, 3)
        Source position of each measurement. Spherical (azimuth deg,
        elevation deg, distance m) for SimpleFreeFieldHRIR, cartesian (m)
        for GeneralFIR.
    convention : {'SimpleFreeFieldHRIR', 'GeneralFIR'}
        SOFA convention, version 1.0.
    **attributes : str
        Global attributes, without the 'GLOBAL:' prefix (e.g. Title,
        AuthorContact, Organization, License, DatabaseName). They replace
        the defaults; mandatory ones not given are written empty.

    Raises
    ------
    ValueError
        For an unknown convention, measurements that differ in sampling
        rate, length or number of channels, or a number of source positions
        different from the number of measurements.

    """
    if convention not in _CONVENTIONS:
        raise ValueError(f"convention must be one of {list(_CONVENTIONS)}.")
    conv = _CONVENTIONS[convention]
    signals = [signal] if isinstance(signal, SignalObj) else list(signal)
    src = np.atleast_2d(np.asarray(source_positions, dtype='float64'))
    if src.shape != (len(signals), 3):
        raise ValueError("source_positions must have shape (M, 3), with M "
                         "the number of measurements.")
    if len({(s.samplingRate, s.numSamples, s.numChannels)
            for s in signals}) != 1:
        raise ValueError("All measurements must have the same sampling rate,"
                         " number of samples and number of channels.")
    ir = np.stack([np.asarray(s.timeSignal, dtype='float64').T
                   for s in signals])
    M, R, N = ir.shape
    receivers = np.zeros((R, 3, 1))
    if conv['ears'] and R == 2:
        receivers[:, 1, 0] = [0.09, -0.09]

    from . import __version__
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    glob = {'Conventions': 'SOFA', 'Version': '2.1',
            'SOFAConventions': convention, 'SOFAConventionsVersion': '1.0',
            'APIName': 'pytta', 'APIVersion': __version__,
            'AuthorContact': '', 'DataType': 'FIR',
            'License': 'No license provided, ask the author for permission',
            'Organization': '', 'RoomType': 'free field',
            'DateCreated': now, 'DateModified': now, 'Title': ''}
    glob.update(conv['globals'])
    glob.update(attributes)

    cart = ('cartesian', 'metre')
    variables = [  # name, dimensions, values, (Type, Units)
        ('ListenerPosition', 'IC', [[0., 0., 0.]], cart),
        ('ReceiverPosition', 'RCI', receivers, cart),
        ('SourcePosition', 'MC', src, conv['source']),
        ('EmitterPosition', 'ECI', np.zeros((1, 3, 1)), cart),
        ('ListenerUp', 'IC', [[0., 0., 1.]], None),
        ('ListenerView', 'IC', [[1., 0., 0.]], cart),
        ('Data.IR', 'MRN', ir, None),
        ('Data.SamplingRate', 'I', [float(signals[0].samplingRate)], None),
        ('Data.Delay', 'IR', np.zeros((1, R)), None),
    ]
    sizes = {'C': 3, 'I': 1, 'M': M, 'R': R, 'E': 1, 'N': N}

    with h5py.File(path, 'w', track_order=True) as f:
        for key, value in glob.items():
            _set_text(f.attrs, key, value)
        dimid = {}
        for k, (name, size) in enumerate(sizes.items()):
            # netCDF-4 dimension without a coordinate variable.
            dim = f.create_dataset(name, shape=(size,), dtype='>f4')
            dim.make_scale(_DIMENSION_NAME % size)
            dim.attrs['_Netcdf4Dimid'] = np.int32(k)
            dimid[name] = k
        for name, dims, values, typeunits in variables:
            var = f.create_dataset(name, data=np.asarray(values, 'float64'),
                                   track_order=True)
            for axis, d in enumerate(dims):
                var.dims[axis].attach_scale(f[d])
            var.attrs['_Netcdf4Coordinates'] = np.array(
                [dimid[d] for d in dims], dtype='int32')
            if typeunits is not None:
                _set_text(var.attrs, 'Type', typeunits[0])
                _set_text(var.attrs, 'Units', typeunits[1])
        _set_text(f['Data.SamplingRate'].attrs, 'Units', 'hertz')


def _set_text(attrs, key, value):
    """Store text as a netCDF NC_CHAR attribute (fixed-length string)."""
    raw = str(value).encode('utf-8')
    attrs[key] = np.bytes_(raw) if raw else h5py.Empty(np.dtype('S1'))
