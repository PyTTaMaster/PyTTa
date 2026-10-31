function make_fluctuation_sqat(sqatRoot)
% MAKE_FLUCTUATION_SQAT  Golden data for tests/test_psychoacoustics.py
% (fluctuation strength).
%
%   make_fluctuation_sqat(sqatRoot) runs FluctuationStrength_Osses2016 from
%   SQAT (https://github.com/ggrecow/SQAT) on short synthetic signals and
%   writes fluctuation_sqat.mat next to this file. SQAT is used here only as
%   a test oracle: that SQAT file is GPL-3.0 and pytta.psychoacoustics is
%   written from the paper. Python rebuilds the same signals from the
%   parameters stored here (see the helpers of the test file).
%
%   Signals (sound pressure in Pa), rounded to single precision before the
%   call, as SignalObj stores float32, t = (0:n-1)'/fs:
%     'am':   A*(1 + m*sin(2*pi*fm*t)).*sin(2*pi*fc*t),
%             A = sqrt(2)*2e-5*10^(L/20)/sqrt(1 + m^2/2) (L: rms level);
%     'fm':   sqrt(2)*2e-5*10^(L/20)*sin(2*pi*fc*t - (df/fm)*cos(2*pi*fm*t))
%             (frequency fc + df*sin(2*pi*fm*t));
%
%   Fluctuation strength: SQAT with method 1 (time-varying), time_skip 0,
%   struct_opt.a0_type = 'fastl2007' (the a0 curve that pytta uses).
%   Stored per case (struct array "cases"): the signal parameters, the sum
%   and sum of squares of the single precision input, and the outputs FS(t),
%   time, specific FS (47 x frames) and FSmean. Also stored: matlab_version and sqat_commit.
%
%   Usage:  matlab -singleCompThread -batch "make_fluctuation_sqat('/path/to/SQAT')"

addpath(sqatRoot);
startup_SQAT([strip(sqatRoot, "right", filesep) filesep]);  % expects a trailing separator

fdefs = {
%   name          kind   fc    fm  m   df   L   fs     dur
    'am1k_4hz'    'am'   1000  4   1   0    70  48000  4.0
    'am1k_16hz'   'am'   1000  16  1   0    70  48000  4.0
    'am1k_4hz_441' 'am'  1000  4   1   0    60  44100  3.0
    'fm1k5_4hz'   'fm'   1500  4   0   700  70  48000  4.0
};
opt.a0_type = 'fastl2007';
cases = struct([]);
for c = 1:size(fdefs, 1)
    p = cell2struct(fdefs(c, :), {'name','kind','fc','fm','m','df','L','fs','dur'}, 2);
    t = (0:round(p.dur * p.fs)-1)' / p.fs;
    if strcmp(p.kind, 'am')
        A = sqrt(2)*2e-5*10^(p.L/20) / sqrt(1 + p.m^2/2);
        x = A * (1 + p.m*sin(2*pi*p.fm*t)) .* sin(2*pi*p.fc*t);
    else
        x = sqrt(2)*2e-5*10^(p.L/20) * sin(2*pi*p.fc*t - (p.df/p.fm)*cos(2*pi*p.fm*t));
    end
    x = double(single(x));
    OUT = FluctuationStrength_Osses2016(x, p.fs, 1, 0, 0, opt);
    p.sum_x  = sum(x);
    p.sum_x2 = sum(x.^2);
    p.FS     = OUT.InstantaneousFluctuationStrength;
    p.time   = OUT.time;
    p.FSspec = OUT.InstantaneousSpecificFluctuationStrength;
    p.FSmean = OUT.FSmean;
    cases = il_append(cases, p);
    fprintf('%-14s FSmean = %.4f vacil (%d frames)\n', p.name, p.FSmean, numel(p.FS));
end
il_save('fluctuation_sqat.mat', cases, sqatRoot);

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
