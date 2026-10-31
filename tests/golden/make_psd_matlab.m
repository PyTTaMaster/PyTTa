% Golden data for tests/test_spectrum.py: MATLAB cpsd as ita_psd of the
% ITA-Toolbox calls it (symmetric Hann window of the block size, half-block
% overlap, one-sided density). The noise is rand with the Mersenne twister,
% which numpy.random.RandomState(1).random_sample rebuilds.
fs = 8000; n = 4096; block = 512;
rng(1, 'twister');
x = rand(n, 1) - 0.5;
y = filter([1 0.5 0.25], 1, x);
w = hanning(block);
Pxx = cpsd(x, x, w, block/2, block, fs);
Pxy = cpsd(x, y, w, block/2, block, fs);
save('psd_matlab.mat', 'fs', 'block', 'Pxx', 'Pxy');
