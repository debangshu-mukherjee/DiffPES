 function [info] = import_kpoints(filename)
  if nargin == 0
      filename='KPOINTS';
  end
 % import_kpoints - Read k-points from a VASP KPOINTS file
%
% Output:
%   kpts         - Nx3 array of k-points
%   kpoint_mode  - 'Automatic', 'Line-mode', or 'Explicit'
%   info         - struct with metadata (grid size, shift, weights, etc.)
fid = fopen(filename, 'r');
if fid == -1
    error('Cannot open KPOINTS file.');
end

% Line 1: comment
info.comment = strtrim(fgetl(fid));

% Line 2: number of k-points or 0 for automatic
num_kpts_line = textscan(fid, '%f %c %f %s');
num_kpts = cell2mat(num_kpts_line(1));

% Line 3: mode (Monkhorst-Pack, Gamma, Line-mode, etc.)
mode_line = strtrim(lower(fgetl(fid)));

% Determine k-point mode
if strcmpi(mode_line, 'line-mode')
    mode = 'Line-mode';
elseif num_kpts == 0
    mode = 'Automatic';
else
    mode = 'Explicit';
end
tline = fgetl(fid); 
% Parse based on mode
switch mode
    case 'Automatic'
        % Monkhorst-Pack grid definition
        grid = fscanf(fid, '%d', [1, 3]);
        shift = fscanf(fid, '%f', [1, 3]);
        prekpts = [];
        info.grid = grid;
        info.shift = shift;

    case 'Line-mode'
        % Path between high symmetry points
        is_select = contains(lower(mode_line), 'select');
        if is_select
            weights = fscanf(fid, '%f', [1, num_kpts]);
        end
        % Read N line segments
        kpts_raw = textscan(fid, '%s');
        kpts_parsed = kpts_raw{1};
        klines = length(kpts_parsed(:,1))/5;
        prek_labels = {1:klines};
        prekpts = zeros(klines,3);
        n=0;
        for i = 1:length(kpts_parsed(:,1))
            x = i-n*5;
            if x == 1 || x == 2 || x == 3
                prekpts((n+1),x) = str2double(cell2mat(kpts_parsed(i,1)));
            elseif x == 5
                prek_labels(n+1) = kpts_parsed(i,1);
                n = n + 1;
            else
            end  
        end 
        segments = size(prekpts, 1) / 2;
        kpts = zeros(segments+1,3);
        k_labels = cell(1,segments+1);
        for i = 1:length(prek_labels)
            if i == 1
                n=1;
                kpts(n,:) = prekpts(i,:);
                k_labels(n) = prek_labels(i);
            elseif i > 1 && rem(i,2) == 0
                n=n+1;
                kpts(n,:) = prekpts(i,:);
                k_labels(n) = prek_labels(i);
            end
        end
        info.segments = segments;
        info.kpts = kpts;
        info.k_labels = k_labels;
    otherwise % Explicit k-points with optional weights
        kpt_data = textscan(fid, '%f %f %f %f', num_kpts);
        prekpts = [kpt_data{1}, kpt_data{2}, kpt_data{3}];
        weights = kpt_data{4};
        info.weights = weights;
end
info.num_kpts = num_kpts;
info.mode = mode;
fclose(fid);
end