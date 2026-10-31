.. currentmodule:: pytta

Installation
============

PyTTa needs Python 3.10 or later. Sounddevice, used for audio input and
output, needs the PortAudio library, which its wheels for Windows and macOS
include. On Linux, install it with the system package manager, for example::

    sudo apt install libportaudio2

To install the last version published on PyPI, which can be behind the
development branch::

    pip install pytta

To install the development branch::

    pip install "pytta @ git+https://github.com/pyttamaster/pytta@development"

``pytta.plot_waterfall`` needs the optional packages more-itertools and
kaleido.
