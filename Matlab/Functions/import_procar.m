% 
% % PROCAR EXTRACTOR
% function [projections, energies, kpoints] = import_procar(filename)
%     % Parse a VASP PROCAR file
%     % Returns:
%     %   projections(kpt, band, atom, orbital)
%     %   energies(kpt, band)
%     %   kpoints(kpt, 3)
%     
%  if nargin == 0
        filename='PROCAR';
        norbs = 9;
%   end    
%data load
fidr = fopen(filename, 'r');
 % Skip header lines to find number of k-points, bands, and ions
    header = '';
    while ~contains(header, 'k-points')
        header = fgetl(fidr);
    end
params = textscan(header,'%s %s %s %d %s %s %s %d %s %s %s %d');
nkpts = cell2mat(params(1,4));
nbands = cell2mat(params(1,8));
natoms = cell2mat(params(1,12));

% Preallocate
projections = zeros(nkpts, nbands, natoms, norbs);
energies = zeros(nkpts, nbands);
kpoints = zeros(nkpts, 3);

% Regex to parse energies and k-point
    k_re = 'k-point\s+(\d+)\s*:\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)';
    e_re = '\s*\d+\s+[-\d.]+\s+([-.\dE+]+)';
 % Start parsing
    while ~feof(fidr)
        line = fgetl(fidr);
        if contains(line, 'k-point')
            k_match = regexp(line, k_re, 'tokens');
            k_idx = str2double(k_match{1}{1});
            kpoints(k_idx, :) = [str2double(k_match{1}{2}), str2double(k_match{1}{3}), str2double(k_match{1}{4})];

            % Read all bands for this k-point
            for b = 1:nbands
                while true
                    pos = ftell(fidr);
                    line = fgetl(fidr);
                    if contains(line, 'band')
                        break;
                    end
                end
                checkpoint = 'after while loop';
                e_match = regexp(line, e_re, 'tokens');
                energies(k_idx, b) = str2double(e_match{1}{1});

                % Skip header line for orbitals
                fgetl(fidr);

                % Read projection lines for each atom
                for a = 1:natoms
                    data = textscan(fidr, repmat('%f', 1, norbs+2));
                    projections(k_idx, b, a, :) = cell2mat(data(2:norbs+1));
                end

                % Skip total and blank line
                fgetl(fidr);
                fgetl(fidr);
            end
        end
    end
    fclose(fidr);
%end