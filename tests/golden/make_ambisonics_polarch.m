% Golden data for tests/test_ambisonics.py: polarch Higher-Order-Ambisonics
% and Vector-Base-Amplitude-Panning (Archontis Politis, BSD 3-Clause), the
% copies that Iceberg installs. Three functions they take from polarch
% Spherical-Harmonic-Transform are replaced by minimal versions: getTdesign
% (the t-designs of pytta/_tdesigns.py), sphDelaunay (the faces of the
% convex hull) and euler2rotationMatrix ('zyx'). rotateHOA_N3D is not here;
% the test checks the rotation by a rule. Run from tests/golden with those
% on the path.
rng(1, 'twister');
dirs = [rand(20,1)*360-180, asind(rand(20,1)*2-1)];
Y4 = getRSH(4, dirs);
sig3 = rand(8, 16) - 0.5;                            % third order, N3D ACN
conv.n2sn = convert_N3D_SN3D(sig3, 'n2sn');
conv.sn2n = convert_N3D_SN3D(sig3, 'sn2n');
conv.n2fuma = convert_N3D_FuMa(sig3, 'n2fuma');
conv.fuma2n = convert_N3D_FuMa(sig3, 'fuma2n');
conv.acn2sid = convert_ACN_SID(sig3, 'acn2sid');
conv.sid2acn = convert_ACN_SID(sig3, 'sid2acn');
maxre = {getMaxREweights(1), getMaxREweights(3), getMaxREweights(5)};
ls_hemi = [0 0; 90 0; 180 0; -90 0; 45 45; 135 45; -135 45; -45 45; 0 90];
ls_full = [(0:90:270)' -30*ones(4,1); (0:45:315)' zeros(8,1); (30:60:330)' 45*ones(6,1); 0 90];
order_hemi = getLayoutAmbisonicOrder(ls_hemi);
order_full = getLayoutAmbisonicOrder(ls_full);
D.sad = ambiDecoder(ls_full, 'sad', 1, 3);
D.mmd = ambiDecoder(ls_full, 'mmd', 1, 3);
D.epad = ambiDecoder(ls_full, 'epad', 0, 3);
D.allrad = ambiDecoder(ls_full, 'allrad', 1, 3);
D.csad = ambiDecoder(ls_full, 'csad', 1, 2);
[D.auto, D.auto_order] = ambiDecoder(ls_full, 'sad');
src = [30 10; -100 40];
s2 = rand(16, 2) - 0.5;
enc3 = encodeHOA_N3D(3, s2, src);
rng(2, 'twister');
hoa2 = rand(400, 9) - 0.5;   % numpy: RandomState(2).random_sample((9, 400)).T - 0.5
Db = cat(3, ambiDecoder(ls_full,'sad',1,2), ambiDecoder(ls_full,'mmd',1,2), ambiDecoder(ls_full,'allrad',1,2));
dec_bands = decodeHOA_N3D(hoa2, Db, [500 2000], 48000);
bf = encodeBformat(s2, src);
D1 = ambiDecoder(ls_full, 'sad', 1, 1);
bf_dec = decodeBformat(bf, D1);
rotbf = rotateBformat(bf, 30, -20, 10);
save('ambisonics_polarch.mat', 'dirs', 'Y4', 'sig3', 'conv', 'maxre', 'ls_hemi', 'ls_full', ...
     'order_hemi', 'order_full', 'D', 'src', 's2', 'enc3', 'Db', 'dec_bands', ...
     'bf', 'D1', 'bf_dec', 'rotbf');
fprintf('orders: hemisphere %d, full %d, auto %d\n', order_hemi, order_full, D.auto_order);
