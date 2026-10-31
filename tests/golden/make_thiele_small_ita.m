function make_thiele_small_ita(itaRoot)
% MAKE_THIELE_SMALL_ITA  Golden data for tests/test_loudspeaker.py.
%
%   make_thiele_small_ita(itaRoot) builds the electrical impedance of a
%   loudspeaker with known lumped parameters with ita_linear_loudspeaker
%   from the ITA-Toolbox (RWTH Aachen, https://git.rwth-aachen.de/ita/toolbox),
%   once in free air and once with an added mass, runs ita_thiele_small on
%   the pair and writes thiele_small_ita.mat next to this file.
%
%   The file holds the input parameters, the frequency grid (sampling rate
%   and FFT degree), the free air impedance at every 20th bin up to 400 Hz
%   (to check the model) and the parameters estimated by ita_thiele_small.
%   The voice coil inductance is zero: ita_thiele_small reads the
%   impedance peak and bandwidth without an inductance correction.
%
%   Usage:  matlab -batch "make_thiele_small_ita('/path/to/ITA-Toolbox')"

addpath(genpath(itaRoot));

sr = 44100;
fftDegree = 18;
Re  = 6.0;      % Ohm
Mms = 0.012;    % kg
Cms = 7.0e-4;   % m/N
Rms = 1.2;      % kg/s
Bl  = 6.0;      % T m
dm  = 0.010;    % kg, added mass
d   = 0.13;     % m, membrane diameter

TS = struct('R_e', Re, 'L_e', 0, 'm', Mms, 'n', Cms, 'w', Rms, 'M', Bl);
Z = ita_linear_loudspeaker(TS, 'samplingRate', sr, 'fftDegree', fftDegree);
TSm = TS;
TSm.m = Mms + dm;
Zm = ita_linear_loudspeaker(TSm, 'samplingRate', sr, 'fftDegree', fftDegree);

est = ita_thiele_small(Z, Zm, dm, d, 'frequency_limits', [1 7000]);

idx = (1:20:Z.freq2index(400))';
freqSample = Z.freqVector(idx);
zSample = Z.freqData(idx);
c = double(est.c);
rho0 = double(est.rho0);

ita = struct('fs', double(est.f_s), 'Re', double(est.R_e), ...
    'Qms', double(est.Q_m), 'Qes', double(est.Q_e), 'Qts', double(est.Q_tot), ...
    'Mms', double(est.m), 'Cms', double(est.n), 'Rms', double(est.w), ...
    'Bl', double(est.M), 'Vas', double(est.V_equi));

out = fullfile(fileparts(mfilename('fullpath')), 'thiele_small_ita.mat');
save(out, 'sr', 'fftDegree', 'Re', 'Mms', 'Cms', 'Rms', 'Bl', 'dm', 'd', ...
    'c', 'rho0', 'freqSample', 'zSample', 'ita', '-v7');
fprintf('wrote %s\n', out);
disp(ita);
end
