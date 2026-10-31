function make_time_window_ita(itaRoot)
% MAKE_TIME_WINDOW_ITA  Golden data for tests/test_time_window.py.
%
%   make_time_window_ita(itaRoot) runs ita_time_window from the ITA-Toolbox
%   (RWTH Aachen, https://git.rwth-aachen.de/ita/toolbox) on a fixed random
%   signal and writes time_window_ita.mat next to this file.
%
%   Intervals are given in SAMPLES, 1-based as MATLAB indexes them, so the
%   file pins the window shape itself and not the seconds-to-samples
%   rounding of either implementation. The Python test subtracts 1.
%
%   The signal is stored in the file: the test never regenerates it.
%
%   Usage:  matlab -batch "make_time_window_ita('/path/to/ITA-Toolbox')"

addpath(genpath(itaRoot));

fs = 1000;
rng(0, 'twister');
x = randn(1000, 2);
a = itaAudio(x, fs, 'time');

% name, interval (1-based samples), window, symmetric
spec = {
    'falling_hann',   [201 400],          @hann,    false
    'rising_hann',    [600 450],          @hann,    false
    'both_ends_hann', [100 1 800 1000],   @hann,    false
    'both_ends_rect', [50 1 900 1000],    @rectwin, false
    'symmetric_hann', [301 500],          @hann,    true
};

cases = struct('name', {}, 'interval', {}, 'window', {}, 'symmetric', {}, 'y', {});
for k = 1:size(spec, 1)
    if spec{k, 4}
        r = ita_time_window(a, spec{k, 2}, spec{k, 3}, 'samples', 'symmetric');
    else
        r = ita_time_window(a, spec{k, 2}, spec{k, 3}, 'samples');
    end
    cases(k).name      = spec{k, 1};
    cases(k).interval  = spec{k, 2};
    cases(k).window    = func2str(spec{k, 3});
    cases(k).symmetric = spec{k, 4};
    cases(k).y         = r.timeData;
end

out = fullfile(fileparts(mfilename('fullpath')), 'time_window_ita.mat');
save(out, 'x', 'fs', 'cases', '-v7');
fprintf('wrote %s (%d cases)\n', out, numel(cases));
end
