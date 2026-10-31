function make_loudness_sqat(sqatRoot)
% MAKE_LOUDNESS_SQAT  Golden data for tests/test_psychoacoustics.py (loudness).
%
%   make_loudness_sqat(sqatRoot) runs Loudness_ISO532_1 from SQAT
%   (https://github.com/ggrecow/SQAT) on short synthetic signals and writes
%   loudness_sqat.mat next to this file. Python rebuilds the same signals
%   from the parameters stored here (see the helpers of the test file).
%
%   Signals (sound pressure in Pa). Every signal is rounded to single
%   precision before the call, as SignalObj stores float32, so MATLAB and
%   Python see the same samples:
%     tone      a*sin(2*pi*f*t), a = sqrt(2)*2e-5*10^(L/20)
%     multitone the 28 one-third-octave centre frequencies 25 Hz..12.5 kHz,
%               each at L dB, Schroeder phases pi*k*(k-1)/28, summed in
%               order k = 1..28
%     burst     tone that is non-zero for t0 <= t < t1 only
%     ramp      tone whose level goes linearly from L to L2 dB over the
%               signal: a(t) = sqrt(2)*2e-5*10.^((L+(L2-L)*t/T)/20)
%   with t = (0:n-1)'/fs and T = n/fs.
%
%   Stored per case (cell array "cases" of structs): the signal
%   parameters, the call arguments (field 0 free / 1 diffuse, method 1
%   stationary / 2 time-varying, time_skip), sum and sum of squares of the
%   single precision input (to check that Python rebuilt the same signal),
%   and the outputs: stationary N, LN, Nspec [1x240]; time-varying N(t), LN(t),
%   time, N5, Nmax, and the specific loudness at every 25th output frame
%   (Nspec_t [frames x 240], frames in Nspec_idx, 1-based) to keep the file
%   small.
%
%   Also stored: matlab_version and sqat_commit (the commit of the SQAT
%   checkout, read from its .git folder).
%
%   Usage:  matlab -singleCompThread -batch "make_loudness_sqat('/path/to/SQAT')"

addpath(sqatRoot);
startup_SQAT([strip(sqatRoot, "right", filesep) filesep]);  % expects a trailing separator

defs = {
%   name             kind        f     L   L2  t0   t1   fs     dur  field method skip
    'tone1k_40'      'tone'      1000  40  0   0    0    48000  1.0  0     1      0
    'multitone45'    'multitone' 0     45  0   0    0    48000  1.0  0     1      0.2
    'multitone45_D'  'multitone' 0     45  0   0    0    48000  1.0  1     1      0
    'tone250_80_441' 'tone'      250   80  0   0    0    44100  1.0  0     1      0
    'burst1k_70'     'burst'     1000  70  0   0.1  0.2  48000  0.5  0     2      0
    'ramp1k_40_80_D' 'ramp'      1000  40  80  0    0    48000  1.0  1     2      0.2
    'tone1k_60_441'  'tone'      1000  60  0   0    0    44100  0.5  0     2      0
};

cases = cell(1, size(defs, 1));  % cell array: stationary and time-varying cases store different fields
for c = 1:size(defs, 1)
    d = defs(c, :);
    p = cell2struct(d, {'name','kind','f','L','L2','t0','t1','fs','dur', ...
                        'field','method','time_skip'}, 2);
    x = il_signal(p);
    OUT = Loudness_ISO532_1(x, p.fs, p.field, p.method, p.time_skip, 0);
    p.sum_x  = sum(x);
    p.sum_x2 = sum(x.^2);
    if p.method == 1
        p.N     = OUT.Loudness;
        p.LN    = OUT.LoudnessLevel;
        p.Nspec = OUT.SpecificLoudness;
    else
        p.N    = OUT.InstantaneousLoudness;
        p.LN   = OUT.InstantaneousLoudnessLevel;
        p.time = OUT.time;
        p.N5   = OUT.N5;
        p.Nmax = OUT.Nmax;
        idx = 1:25:numel(OUT.time);
        p.Nspec_idx = idx;
        p.Nspec_t   = OUT.InstantaneousSpecificLoudness(idx, :);
    end
    cases{c} = p;
    fprintf('%-16s N = %s\n', p.name, mat2str(p.N(1:min(3,end))', 6));
end

matlab_version = version;
sqat_commit = il_sqat_commit(sqatRoot);
out = fullfile(fileparts(mfilename('fullpath')), 'loudness_sqat.mat');
save(out, 'cases', 'matlab_version', 'sqat_commit', '-v7');
fprintf('wrote %s (%d cases)\n', out, numel(cases));
end

function x = il_signal(p)
n = round(p.dur * p.fs);
t = (0:n-1)' / p.fs;
switch p.kind
    case 'tone'
        x = sqrt(2)*2e-5*10^(p.L/20) * sin(2*pi*p.f*t);
    case 'burst'
        x = sqrt(2)*2e-5*10^(p.L/20) * sin(2*pi*p.f*t);
        x(~(t >= p.t0 & t < p.t1)) = 0;
    case 'ramp'
        T = n / p.fs;
        x = sqrt(2)*2e-5*10.^((p.L + (p.L2-p.L)*t/T)/20) .* sin(2*pi*p.f*t);
    case 'multitone'
        K = 28;
        a = sqrt(2)*2e-5*10^(p.L/20);
        x = zeros(n, 1);
        for k = 1:K
            fk = 1000 * 10^(((k-1)-16)/10);
            x = x + a * sin(2*pi*fk*t + pi*k*(k-1)/K);
        end
end
x = double(single(x));
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
