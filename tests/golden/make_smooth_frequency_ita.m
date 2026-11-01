function make_smooth_frequency_ita(itaRoot)
% MAKE_SMOOTH_FREQUENCY_ITA  Golden data for tests/test_smooth_frequency.py.
%
%   make_smooth_frequency_ita(itaRoot) runs ita_smooth_frequency from the
%   ITA-Toolbox (RWTH Aachen, https://git.rwth-aachen.de/ita/toolbox) on a
%   fixed random signal and writes smooth_frequency_ita.mat next to this
%   file: the time signal, the ITA input spectrum and one output spectrum
%   per bandwidth (1, 1/3 and 1/12 octave).
%
%   The DC and Nyquist bins of the signal are set to zero. PyTTa and the
%   ITA-Toolbox scale these two bins differently from the others, so with
%   them at zero the two spectra differ by one constant factor and the
%   comparison pins the smoothing alone.
%
%   Usage:  matlab -batch "make_smooth_frequency_ita('/path/to/ITA-Toolbox')"

addpath(genpath(itaRoot));

fs = 48000;
n = 2048;
rng(0, 'twister');
x = randn(n, 2);
alt = (-1) .^ (0:n-1)';
x = x - mean(x, 1);                      % DC bin to zero
x = x - alt .* mean(x .* alt, 1);        % Nyquist bin to zero
a = itaAudio(x, fs, 'time');

nthOct = [1 3 12];
cases = struct('nthOct', {}, 'freqData', {});
for k = 1:numel(nthOct)
    r = ita_smooth_frequency(a, 'bandwidth', 1 / nthOct(k));
    cases(k).nthOct   = nthOct(k);
    cases(k).freqData = r.freqData;
end
freqDataIn = a.freqData;

out = fullfile(fileparts(mfilename('fullpath')), 'smooth_frequency_ita.mat');
save(out, 'x', 'fs', 'freqDataIn', 'cases', '-v7');
fprintf('wrote %s (%d cases)\n', out, numel(cases));
end
