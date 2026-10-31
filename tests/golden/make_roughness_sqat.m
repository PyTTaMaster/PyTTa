function make_roughness_sqat(sqatRoot)
% MAKE_ROUGHNESS_SQAT  Golden data for tests/test_psychoacoustics.py (roughness).
%
%   make_roughness_sqat(sqatRoot) runs Roughness_Daniel1997 from SQAT
%   (https://github.com/ggrecow/SQAT) on short synthetic signals and writes
%   roughness_sqat.mat next to this file. Python rebuilds the same signals
%   from the parameters stored here (see the helpers of the test file).
%
%   Signal (sound pressure in Pa), rounded to single precision before the
%   call, as SignalObj stores float32:
%     x = A*(1 + m*sin(2*pi*fm*t)).*sin(2*pi*fc*t),  t = (0:n-1)'/fs,
%     A = sqrt(2)*2e-5*10^(L/20)/sqrt(1 + m^2/2),
%   so that L is the level of the rms of the modulated signal (m = 0 gives
%   a pure tone).
%
%   Stored per case (struct array "cases"): the signal parameters,
%   time_skip, sum and sum of squares of the single precision input (to
%   check that Python rebuilt the same signal), and the outputs R(t)
%   (InstantaneousRoughness), time, the specific roughness
%   (InstantaneousSpecificRoughness, 47 x frames) and Rmean.
%
%   Also stored: matlab_version and sqat_commit (the commit of the SQAT
%   checkout, read from its .git folder).
%
%   Usage:  matlab -singleCompThread -batch "make_roughness_sqat('/path/to/SQAT')"

addpath(sqatRoot);
startup_SQAT([strip(sqatRoot, "right", filesep) filesep]);  % expects a trailing separator

defs = {
%   name              fc    fm   m    L   fs     dur  skip
    'am1k_70_ref'     1000  70   1    60  48000  2.0  0.5
    'am1k_70_441'     1000  70   1    60  44100  1.0  0
    'am2k_50_m05_32k' 2000  50   0.5  70  32000  1.0  0
    'tone1k_60'       1000  0    0    60  48000  1.0  0
};

cases = struct([]);
for c = 1:size(defs, 1)
    p = cell2struct(defs(c, :), {'name','fc','fm','m','L','fs','dur','time_skip'}, 2);
    n = round(p.dur * p.fs);
    t = (0:n-1)' / p.fs;
    A = sqrt(2)*2e-5*10^(p.L/20) / sqrt(1 + p.m^2/2);
    x = double(single(A * (1 + p.m*sin(2*pi*p.fm*t)) .* sin(2*pi*p.fc*t)));
    OUT = Roughness_Daniel1997(x, p.fs, p.time_skip, 0);
    p.sum_x  = sum(x);
    p.sum_x2 = sum(x.^2);
    p.R      = OUT.InstantaneousRoughness;
    p.time   = OUT.time;
    p.Rspec  = OUT.InstantaneousSpecificRoughness;
    p.Rmean  = OUT.Rmean;
    if isempty(cases)
        cases = p;
    else
        cases(end+1) = p; %#ok<AGROW>
    end
    fprintf('%-16s Rmean = %.6f asper (%d frames)\n', p.name, p.Rmean, numel(p.R));
end

matlab_version = version;
sqat_commit = il_sqat_commit(sqatRoot);
out = fullfile(fileparts(mfilename('fullpath')), 'roughness_sqat.mat');
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
