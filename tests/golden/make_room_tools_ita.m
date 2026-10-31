% Golden data for tests/test_room_tools.py: ita_constants and
% ita_din18041_reverberation_times of the ITA-Toolbox. Run from tests/golden
% with the ITA-Toolbox on the path.
conds = [20 0.5 101325; 0 0.0 101325; 35 0.9 95000];
c = zeros(3,1); rho = c; z0 = c;
for k = 1:3
    [c(k), rho(k), z0(k)] = ita_constants({'c','rho_0','z_0'}, 'T', conds(k,1), 'phi', conds(k,2), 'p', conds(k,3));
    c(k) = double(c(k)); rho(k) = double(rho(k)); z0(k) = double(z0(k));
end
r = ita_din18041_reverberation_times('v', 1000, 'purpose', 'speech');
f = r.freqVector; din_speech = r.freqData;
r = ita_din18041_reverberation_times('v', 1000, 'purpose', 'music');
din_music = r.freqData;
idx = 2:16:numel(f);  % a subset of the 1025 bins
f = f(idx); din_speech = din_speech(idx,:); din_music = din_music(idx,:);
save('room_tools_ita.mat', 'conds', 'c', 'rho', 'z0', 'f', 'din_speech', 'din_music');
fprintf('c %s rho %s\n', mat2str(c', 6), mat2str(rho', 6));
