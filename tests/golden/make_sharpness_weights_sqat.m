function make_sharpness_weights_sqat(sqatRoot)
% MAKE_SHARPNESS_WEIGHTS_SQAT  Golden data for tests/test_psychoacoustics.py
% (sharpness with the weightings of Aures and von Bismarck, DIN 45692 Annex B).
%
%   make_sharpness_weights_sqat(sqatRoot) runs Sharpness_DIN45692 of SQAT
%   (https://github.com/ggrecow/SQAT) with the weightings 'DIN45692',
%   'aures' and 'bismarck', stationary free-field loudness, on band noises
%   and writes sharpness_weights_sqat.mat next to this file. SQAT is only the
%   test oracle (that file is GPL-3.0; pytta follows the text of the
%   standard).
%
%   Signals: multisine noise with components every 1 Hz from f1 to f2,
%   phases 2*pi*mod(g*k^2, 1), g = 0.6180339887, k = 1..K, scaled to the
%   rms level L (dB re 20 uPa), 1 s at 48 kHz, rounded to single precision.
%
%   Usage:  matlab -batch "make_sharpness_weights_sqat('/path/to/SQAT')"

addpath(sqatRoot);
startup_SQAT([strip(sqatRoot, "right", filesep) filesep]);
defs = [920 1080 40; 920 1080 60; 920 1080 80; 3700 4400 60; 250 10000 60; 200 300 70];
fs = 48000; t = (0:fs-1)' / fs;
cases = struct([]);
for c = 1:size(defs, 1)
    p.f1 = defs(c,1); p.f2 = defs(c,2); p.L = defs(c,3);
    K = p.f2 - p.f1 + 1;
    x = zeros(size(t));
    for k = 1:K
        x = x + sin(2*pi*(p.f1 + k - 1)*t + 2*pi*mod(0.6180339887*k^2, 1));
    end
    x = double(single(x * (2e-5*10^(p.L/20) / sqrt(mean(x.^2)))));
    p.sum_x = sum(x);
    for w = {'DIN45692', 'aures', 'bismarck'}
        OUT = Sharpness_DIN45692(x, fs, w{1}, 0, 1, 0, 0, 0);
        p.(['S_' w{1}]) = OUT.Sharpness;
    end
    if isempty(cases), cases = p; else, cases(end+1) = p; end %#ok<AGROW>
    fprintf('%5d-%5d Hz %2d dB: S %.4f, SA %.4f, SB %.4f\n', p.f1, p.f2, p.L, ...
            p.S_DIN45692, p.S_aures, p.S_bismarck);
end
save(fullfile(fileparts(mfilename('fullpath')), 'sharpness_weights_sqat.mat'), 'cases', 'fs');
end
