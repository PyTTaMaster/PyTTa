% Golden data for tests/test_din45681.py: ita_tonality (ITA-Toolbox,
% applications/Psychoacoustics) on tones in white noise at 44.1 kHz.
% The noise is uniform, from rand with the Mersenne twister, so that
% numpy.random.RandomState(seed).random_sample rebuilds the same samples;
% the sum of each signal is stored to check that.
% Run from tests/golden with the ITA-Toolbox on the path.
fs = 44100; t = (0:3*fs-1)'/fs; p0 = 2e-5;
cases = struct('name', {}, 'seed', {}, 'noise_rms', {}, 'tones', {}, 'levels', {});
cases(1) = struct('name', '1 kHz in noise', 'seed', 1, 'noise_rms', 0.02, 'tones', 1000, 'levels', 60);
cases(2) = struct('name', '400 Hz in noise', 'seed', 2, 'noise_rms', 0.02, 'tones', 400, 'levels', 55);
cases(3) = struct('name', '3 kHz in noise', 'seed', 3, 'noise_rms', 0.02, 'tones', 3000, 'levels', 50);
cases(4) = struct('name', 'two tones in one band', 'seed', 4, 'noise_rms', 0.02, 'tones', [1000 1050], 'levels', [55 55]);
for k = 1:numel(cases)
    c = cases(k);
    rng(c.seed, 'twister');
    n = rand(numel(t), 1) - 0.5;
    x = c.noise_rms * n / sqrt(1/12);
    for m = 1:numel(c.tones)
        x = x + sqrt(2) * p0 * 10^(c.levels(m)/20) * sin(2*pi*c.tones(m)*t);
    end
    x = double(single(x));
    a = itaAudio(x, fs, 'time'); a.channelUnits = {'Pa'};
    [dl, f, info] = ita_tonality(a);
    close all
    cases(k).sum = sum(x);
    cases(k).deltaL = dl;
    cases(k).frequency = f;
    cases(k).info = info;
    fprintf('%s: deltaL %.3f dB at %g Hz, %d tones\n', c.name, dl, f, size(info, 1));
end
save('din45681_ita.mat', 'cases', 'fs');
