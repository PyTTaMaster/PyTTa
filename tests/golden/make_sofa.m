% Generate the SOFA golden files used by tests/test_sofa.py.
%
% sofa_golden.sofa is a tiny SimpleFreeFieldHRIR 1.0 file (AES69-2022) with
% M = 3 measurements, R = 2 receivers and N = 8 samples, written with the
% netCDF library built into MATLAB (nccreate, ncwrite, ncwriteatt), so it
% does not depend on pytta. sofa_golden.mat holds what MATLAB ncread and
% ncreadatt return for that file, permuted to the C order used by netCDF
% and by pytta.sofa.read (Data.IR is M x R x N).
%
% Run from tests/golden:
%   matlab -batch "make_sofa"

f = 'sofa_golden.sofa';
if exist(f, 'file')
    delete(f);
end

M = 3; R = 2; N = 8;
[n, r, m] = ndgrid(0:N-1, 0:R-1, 0:M-1);
ir = sin(1 + n + 10*r + 100*m) ./ (1 + n);   % N x R x M in MATLAB order
fs = 44100;
src = [0 0 1.2; 90 0 1.2; 180 45 1.2];        % M x C, spherical
rec = [0 0.09 0; 0 -0.09 0];                  % R x C, cartesian

% MATLAB lists dimensions fastest first, the reverse of netCDF C order:
% {'N','R','M'} gives the netCDF variable Data.IR(M, R, N).
fmt = {'Format', 'netcdf4', 'Datatype', 'double'};
nccreate(f, 'ListenerPosition', 'Dimensions', {'C', 3, 'I', 1}, fmt{:});
nccreate(f, 'ReceiverPosition', 'Dimensions', {'I', 1, 'C', 3, 'R', R}, fmt{:});
nccreate(f, 'SourcePosition', 'Dimensions', {'C', 3, 'M', M}, fmt{:});
nccreate(f, 'EmitterPosition', 'Dimensions', {'I', 1, 'C', 3, 'E', 1}, fmt{:});
nccreate(f, 'ListenerUp', 'Dimensions', {'C', 3, 'I', 1}, fmt{:});
nccreate(f, 'ListenerView', 'Dimensions', {'C', 3, 'I', 1}, fmt{:});
nccreate(f, 'Data.IR', 'Dimensions', {'N', N, 'R', R, 'M', M}, fmt{:});
nccreate(f, 'Data.SamplingRate', 'Dimensions', {'I', 1}, fmt{:});
nccreate(f, 'Data.Delay', 'Dimensions', {'R', R, 'I', 1}, fmt{:});

ncwrite(f, 'ListenerPosition', [0; 0; 0]);
ncwrite(f, 'ReceiverPosition', reshape(rec.', [1 3 R]));
ncwrite(f, 'SourcePosition', src.');
ncwrite(f, 'EmitterPosition', [0 0 0]);
ncwrite(f, 'ListenerUp', [0; 0; 1]);
ncwrite(f, 'ListenerView', [1; 0; 0]);
ncwrite(f, 'Data.IR', ir);
ncwrite(f, 'Data.SamplingRate', fs);
ncwrite(f, 'Data.Delay', zeros(R, 1));

glob = {'Conventions', 'SOFA'; 'Version', '2.1'; ...
        'SOFAConventions', 'SimpleFreeFieldHRIR'; ...
        'SOFAConventionsVersion', '1.0'; 'APIName', 'MATLAB netCDF'; ...
        'APIVersion', version('-release'); 'DataType', 'FIR'; ...
        'AuthorContact', 'pytta developers'; 'Organization', 'PyTTa'; ...
        'License', 'MIT'; 'RoomType', 'free field'; ...
        'DateCreated', '2026-10-31 12:00:00'; ...
        'DateModified', '2026-10-31 12:00:00'; ...
        'Title', 'pytta.sofa golden'; 'DatabaseName', 'pytta tests'; ...
        'ListenerShortName', 'none'};
for k = 1:size(glob, 1)
    ncwriteatt(f, '/', glob{k, 1}, glob{k, 2});
end
pos = {'ListenerPosition', 'cartesian', 'metre'; ...
       'ReceiverPosition', 'cartesian', 'metre'; ...
       'SourcePosition', 'spherical', 'degree, degree, metre'; ...
       'EmitterPosition', 'cartesian', 'metre'; ...
       'ListenerView', 'cartesian', 'metre'};
for k = 1:size(pos, 1)
    ncwriteatt(f, pos{k, 1}, 'Type', pos{k, 2});
    ncwriteatt(f, pos{k, 1}, 'Units', pos{k, 3});
end
ncwriteatt(f, 'Data.SamplingRate', 'Units', 'hertz');

% What MATLAB reads back, in netCDF C order.
IR = permute(ncread(f, 'Data.IR'), [3 2 1]);                      % M x R x N
SamplingRate = ncread(f, 'Data.SamplingRate');
Delay = ncread(f, 'Data.Delay').';                                % I x R
SourcePosition = ncread(f, 'SourcePosition').';                   % M x C
ReceiverPosition = permute(ncread(f, 'ReceiverPosition'), [3 2 1]); % R x C x I
SourcePositionType = ncreadatt(f, 'SourcePosition', 'Type');
SourcePositionUnits = ncreadatt(f, 'SourcePosition', 'Units');
Title = ncreadatt(f, '/', 'Title');
save('sofa_golden.mat', 'IR', 'SamplingRate', 'Delay', 'SourcePosition', ...
     'ReceiverPosition', 'SourcePositionType', 'SourcePositionUnits', ...
     'Title', '-v7');
