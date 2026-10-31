function make_tonality_sqat(sqatRoot)
% MAKE_TONALITY_SQAT  Golden data for tests/test_psychoacoustics.py (tonality).
%
%   make_tonality_sqat(sqatRoot) runs Tonality_Aures1985 from SQAT
%   (https://github.com/ggrecow/SQAT) on short synthetic signals and writes
%   tonality_sqat.mat next to this file. SQAT is used here only as a test
%   oracle: that SQAT file is GPL-3.0 and pytta.psychoacoustics is written
%   from the papers. Python rebuilds the same signals from the
%   parameters stored here (see the helpers of the test file).
%
%   Signals (sound pressure in Pa), rounded to single precision before the
%   call, as SignalObj stores float32, t = (0:n-1)'/fs:
%     'tone': sum over the frequencies f of sqrt(2)*2e-5*10^(L/20)*sin(2*pi*f*t);
%     'tone+noise': the tone plus a multisine noise with components every
%             1 Hz from f1 to f2, phases 2*pi*mod(g*k^2, 1) with
%             g = 0.6180339887 (k = 1..K), which scatter the group delay
%             over the 1 s period so that every frame holds the whole band,
%             scaled to the rms level Ln.
%
%   Tonality: SQAT with the free-field loudness, time_skip 0.
%   Stored per case (struct array "cases"): the signal parameters, the sum
%   and sum of squares of the single precision input, and the outputs K(t),
%   time and Kmean. Also stored: matlab_version and sqat_commit.
%
%   Usage:  matlab -singleCompThread -batch "make_tonality_sqat('/path/to/SQAT')"

addpath(sqatRoot);
startup_SQAT([strip(sqatRoot, "right", filesep) filesep]);  % expects a trailing separator

tdefs = {
%   name            kind          f            L   f1   f2    Ln  fs     dur
    'tone1k_60'     'tone'        1000         60  0    0     0   48000  1.0
    'tone500_60'    'tone'        500          60  0    0     0   48000  1.0
    'tone4k_60'     'tone'        4000         60  0    0     0   48000  1.0
    'tone1k_30'     'tone'        1000         30  0    0     0   48000  1.0
    'tones1k_1k2'   'tone'        [1000 1200]  60  0    0     0   48000  1.0
    'tone1k_noise60' 'tone+noise' 1000         60  200  5000  60  48000  1.0
    'tone1k_noise50' 'tone+noise' 1000         60  200  5000  50  48000  1.0
};
cases = struct([]);
for c = 1:size(tdefs, 1)
    p = cell2struct(tdefs(c, :), {'name','kind','f','L','f1','f2','Ln','fs','dur'}, 2);
    t = (0:round(p.dur * p.fs)-1)' / p.fs;
    x = zeros(size(t));
    for f = p.f
        x = x + sqrt(2)*2e-5*10^(p.L/20) * sin(2*pi*f*t);
    end
    if strcmp(p.kind, 'tone+noise')
        K = p.f2 - p.f1 + 1;
        nz = zeros(size(t));
        for k = 1:K
            nz = nz + sin(2*pi*(p.f1 + k - 1)*t + 2*pi*mod(0.6180339887*k^2, 1));
        end
        x = x + nz * (2e-5*10^(p.Ln/20) / sqrt(mean(nz.^2)));
    end
    x = double(single(x));
    OUT = Tonality_Aures1985(x, p.fs, 0, 0, 0);
    p.sum_x  = sum(x);
    p.sum_x2 = sum(x.^2);
    p.K      = OUT.InstantaneousTonality;
    p.time   = OUT.time;
    p.Kmean  = OUT.Kmean;
    cases = il_append(cases, p);
    fprintf('%-14s Kmean = %.4f t.u. (%d frames)\n', p.name, p.Kmean, numel(p.K));
end
il_save('tonality_sqat.mat', cases, sqatRoot);
end

function cases = il_append(cases, p)
if isempty(cases)
    cases = p;
else
    cases(end+1) = p;
end
end

function il_save(name, cases, sqatRoot) %#ok<INUSL>
matlab_version = version; %#ok<NASGU>
sqat_commit = il_sqat_commit(sqatRoot); %#ok<NASGU>
out = fullfile(fileparts(mfilename('fullpath')), name);
save(out, 'cases', 'matlab_version', 'sqat_commit', '-v7');
fprintf('wrote %s (%d cases)\n', out, numel(cases));
end

function c = il_sqat_commit(sqatRoot)
% Commit of the SQAT checkout, read from its .git folder ('' if unknown).
c = '';
g = fullfile(sqatRoot, '.git');
if ~isfolder(g), return; end
head = strtrim(fileread(fullfile(g, 'HEAD')));
if ~startsWith(head, 'ref: '), c = head; return; end
ref = head(6:end);
if isfile(fullfile(g, ref))
    c = strtrim(fileread(fullfile(g, ref)));
elseif isfile(fullfile(g, 'packed-refs'))
    lines = splitlines(fileread(fullfile(g, 'packed-refs')));
    hit = lines(endsWith(lines, [' ' ref]));
    if ~isempty(hit), c = extractBefore(hit{1}, ' '); end
end
end
