## PyTTa - Python in Technical Acoustics

The project began as an effort to create audio, acoustics and vibrational data acquiring and analysis toolbox to a free cost level, high-end results, combining the passion for programming with the expertise in acoustics and vibration of the Acoustics Engineers from the Federal University of Santa Maria.

We are students, teachers, engineers, passionates and inquiring people, on the first steps of a journey throughout the Python path to bring Acoustics to the Open Seas, Open Sources and Open World!

### Usage

We strongly recommend using Anaconda Python Distribution, as it integrates an IDE (Spyder) and lots of packages over which PyTTa is based on
as Numpy, Scipy, Matplotlib and PyPI (pip) which is used to install Sounddevice and PyTTa itself.

### Documentation

This package aims to be an easy interface between acoustician and vibrational engineers in the use of Python for study, engineering or any other ends that it may suite.
From import to each function and attribute we wrote a little documentation that may be useful for understanding everything available, and what is just a means to an end.

To begin, one can try:

    >>> import pytta
    >>> pytta.list_devices()
    >>> pytta.default()

This set of commands will print the available audio I/O devices and the default parameters as they are on the default class object.

To read everything available on the package, and assuming the use of Spyder IDE, one can press "ctrl+i" with the cursor in front of the module, submodule, class, methods, or function names,
this will open the help menu with the documentation of the respective item.
    
    >>> pytta|
    >>> pytta.properties|
    >>> pytta.generate|
    >>> pytta.functions|
    >>> pytta.classes|

The | represents the cursor position to press "ctrl+i" to use the Spyder help widget.

Inside each submodule, the user will find instructions on the available tools, and how to access them.

### Dependencies

- Numpy;
- Scipy;
- Matplotlib;
- Sounddevice;
- Soundfile;
- H5py;
- Numba.

### Installation

The installation can be made through pip, as follows:

    >>> pip install pytta

This will install directly the most stable and up to date version on the repository.

If you want to check the most up to date beta version, please get the development branch source code, clone our repository to your local git, or even install it from pip, as follows:

    >>> pip install git+https://github.com/pyttamaster/pytta@development


### Contributing as a user

Using the toolbox you are already contributing. If you experience any bugs, have any suggestion or idea, have anything to say about it, please consider sending us a message.

### Contributing as a developer

Our workflow consists of parallel development of submodules, either complementing an existing one or creating a new.
Thus, one must create a new branch that points to the development branch. Its name should contain dev- as the suffix, named accordingly with the project intent.
After the new branch is ready and coding finished, we would be glad to receive a pull request.
Consider reading our CodingGuidelines.

### Contact

Contact us at pytta@eac.ufsm.br

## More information

[Main Website](https://sites.google.com/eac.ufsm.br/pytta/)

[Acoustical Engineering UFSM Website](http://www.eac.ufsm.br)

[UFSM Website](https://www.ufsm.br)

[Download Spyder (with Anaconda)](https://www.anaconda.com/download/)

[Or Miniconda](https://conda.io/en/latest/miniconda)
