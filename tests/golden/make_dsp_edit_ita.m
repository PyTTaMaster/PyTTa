% Golden data for tests/test_dsp_edit.py: ita_start_IR, ita_envelope and
% ita_quantize of the ITA-Toolbox on synthetic signals. Indices are 1-based.
% Run from tests/golden with the ITA-Toolbox on the path.
fs = 44100; n = 2048;
rng(5, 'twister');
ir = zeros(n, 3);
delays = [37 300 900];
for c = 1:3
    t = (0:n-delays(c)-1)';
    ir(delays(c)+1:end, c) = exp(-t/200) .* (rand(numel(t),1) - 0.5);
    ir(delays(c)-4:delays(c), c) = [0.001 -0.002 0.003 -0.004 0.005]';   % weak lead-in
end
ir = double(single(ir + 1e-5 * (rand(n, 3) - 0.5)));
a = itaAudio(ir, fs, 'time');
start = ita_start_IR(a);
t = (0:n-1)'/fs;
am = (1 + 0.5*cos(2*pi*50*t)) .* cos(2*pi*2000*t);
env = ita_envelope(itaAudio(am, fs, 'time')).timeData;
q = ita_quantize(itaAudio(ir(:,1), fs, 'time'), 'bits', 8).timeData;
ir = single(ir); env = single(env); q = single(q);   % am is rebuilt by the test
save('dsp_edit_ita.mat', 'fs', 'ir', 'start', 'env', 'q');
fprintf('start %s\n', mat2str(start));
