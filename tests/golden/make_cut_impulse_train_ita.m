% Golden data for tests/test_dsp_misc.py: ita_cutImpulseTrain of the
% ITA-Toolbox on a synthetic train of four decaying bursts in two channels,
% some with an earlier weak arrival. Indices are 1-based.
% Run from tests/golden with the ITA-Toolbox on the path.
fs = 8000; n = 12000;
rng(7, 'twister');
x = 1e-3 * (rand(n, 2) - 0.5);
onsets = [800 3700 6100 9400];
for k = 1:numel(onsets)
    t = (0:1499)';
    burst = exp(-t/150) .* (rand(1500, 1) - 0.5);
    x(onsets(k):onsets(k)+1499, 1) = x(onsets(k):onsets(k)+1499, 1) + burst;
    x(onsets(k)+20:onsets(k)+1519, 2) = x(onsets(k)+20:onsets(k)+1519, 2) + 0.5*burst;
end
x = double(single(x));
[cuts, ~, edges] = ita_cutImpulseTrain(itaAudio(x, fs, 'time'), 'length', 0.25, 'startBeforeThreshold', 0.002);
start = edges(:, 1);
c1 = single(cuts(1).timeData); c2 = single(cuts(2).timeData); x = single(x);
save('cut_impulse_train_ita.mat', 'fs', 'x', 'start', 'c1', 'c2');
fprintf('start %s, size %s\n', mat2str(start'), mat2str(size(c1)));
