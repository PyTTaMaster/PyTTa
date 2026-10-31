% Golden data for tests/test_vbap.py: polarch Vector-Base-Amplitude-Panning
% (Archontis Politis, BSD 3-Clause), the copy that Iceberg installs in
% Toolboxes/polarch/vbap. Run from tests/golden with that folder on the path.
% Indices are 1-based.
ls2 = [0 30 -30 110 -110];                                   % 5.0, 2D
ls3 = [0 0; 90 0; 180 0; -90 0; 45 45; 135 45; -135 45; -45 45; 0 90];  % hemisphere
src2 = (-180:7.5:180)';
[a, e] = meshgrid(-180:20:160, -10:20:90); src3 = [a(:) e(:)];
pairs = findLsPairs(ls2);
inv2 = invertLsMtx(ls2, pairs);
g2 = vbap(src2, pairs, inv2);
g2s = vbap(src2, pairs, inv2, 20);
tri = findLsTriplets(ls3);
inv3 = invertLsMtx(ls3, tri);
g3 = vbap(src3, tri, inv3);
g3s = vbap(src3, tri, inv3, 20);
save('vbap_polarch.mat', 'ls2', 'ls3', 'src2', 'src3', 'pairs', 'inv2', 'g2', ...
     'g2s', 'tri', 'inv3', 'g3', 'g3s');
fprintf('%d pairs, %d triplets\n', size(pairs, 1), size(tri, 1));
disp(tri)
