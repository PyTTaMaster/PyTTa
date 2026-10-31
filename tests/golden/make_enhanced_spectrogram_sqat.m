function make_enhanced_spectrogram_sqat(sqatRoot)
% MAKE_ENHANCED_SPECTROGRAM_SQAT  Golden data for tests/test_enhanced_spectrogram.py.
%
%   make_enhanced_spectrogram_sqat(sqatRoot) runs SQAT_GUI_enhanced_stft
%   from SQAT (https://github.com/ggrecow/SQAT, gui/) on a fixed test signal
%   with both smoothings and the default grid, and writes
%   enhanced_spectrogram_sqat.mat next to this file.
%
%   The signal (0.6 s at 16 kHz, in Pa) holds a 1 kHz tone at 80 dB SPL, a
%   250 Hz tone modulated at 4 Hz, a click at 0.3 s and white noise at
%   40 dB SPL. It is rounded to single precision, as PyTTa stores it, so
%   both sides read the same numbers. The maps are stored on every 16th
%   frequency and every 10th column, enough to pin the method without
%   storing the whole grid; t and f are stored whole.
%
%   Usage:  matlab -batch "make_enhanced_spectrogram_sqat('/path/to/SQAT')"

addpath(fullfile(sqatRoot, 'gui'));

fs = 16000;
n = round(0.6 * fs);
tt = (0:n-1)' / fs;
rng(0, 'twister');
x = 0.2 * sqrt(2) * sin(2*pi*1000*tt) ...
    + 0.05 * (1 + 0.8*sin(2*pi*4*tt)) .* sin(2*pi*250*tt) ...
    + 2e-3 * randn(n, 1);
x(round(0.3 * fs) + 1) = x(round(0.3 * fs) + 1) + 1;
x = double(single(x));

modes = {'readable', 'sharp'};
L = cell(1, 2);
for k = 1:2
    [t, f, Lk] = SQAT_GUI_enhanced_stft(x, fs, modes{k});
    rows = 1:16:numel(f);
    cols = 1:10:numel(t);
    L{k} = Lk(rows, cols);
end
L_readable = L{1};
L_sharp = L{2};

out = fullfile(fileparts(mfilename('fullpath')), 'enhanced_spectrogram_sqat.mat');
x = single(x);
save(out, 'x', 'fs', 't', 'f', 'rows', 'cols', 'L_readable', 'L_sharp', '-v7');
fprintf('wrote %s (%d x %d per map)\n', out, numel(rows), numel(cols));
end
