function [intensity, E_range] = ARPES_simulation_Basicplus(eigenbands, SurfaceOrb, Ef, sigma, fidelity, temperature, photonE)
% Inputs:
%   E_kb         - [nkpt, nband] band energies (e.g. from EIGENVAL)
%   orb_proj     - [nkpt, nband, natom, norb] orbital projections
%   kpts         - [nkpt, 3] k-point coordinates
%   target_orb   - vector of orbital indices to include (e.g. p-orbitals: [2 3 4])
%   fermi_energy - Fermi level in eV
%   sigma        - Gaussian broadening (e.g. 0.1 eV)

[nkpoints, nbands] = size(eigenbands);
E_range = linspace(min(eigenbands(:)) - 1, max(eigenbands(:)) + 1, fidelity); % energy axis
intensity = zeros(nkpoints, length(E_range));

orb_names = {'s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2'};
target_orb = 2:9;  % exclude 's' for now
orb_weight = ones(length(target_orb));

for i = 1:length(target_orb)
    orb_weight(i) = yeh_lindau_cross_section(photonE, orb_names{target_orb(i)});
end

% Fermi–Dirac cutoff (optional, set to 1 for T=0)
T = temperature; k_B = 8.617333e-5;  % eV/K
fermi_func = @(E) 1 ./ (1 + exp((E - Ef) / (k_B * T)));

fprintf('Constructing Orbital Weights \n')
for k = 1:nkpoints
    for b = 1:nbands
        E = eigenbands(k, b);
        preweight = SurfaceOrb(k, b, :, target_orb);        
        % Selective orbital weights
        for i = 1:length(target_orb)
            preweight(1, 1, :, target_orb(i)) = SurfaceOrb(k, b, :, target_orb(i))*orb_weight(i);
        end
        
        % Total orbital weight for selected orbitals (sum over atoms)
        weight = sum(sum(preweight, 4), 3);



        % Fermi-Dirac weight
        f = fermi_func(E);

        % Gaussian broadening
        g = exp(-(E_range - E).^2 / (2 * sigma^2)) / (sqrt(2 * pi) * sigma);
%         V = voigt_profile(E_range, E, sigma, gamma);

        % Add contribution to intensity
        intensity(k, :) = intensity(k, :) + weight .* f .* g;
    end
end
fprintf('Intensity mapped \n')
end

function sigma = yeh_lindau_cross_section(photon_energy, orbital)
% Returns dipole matrix element strength based on Yeh & Lindau data
% orbital: 's', 'px', 'py', 'pz', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2'

% Simplified: Use approximate values at key photon energies (20, 40, 60 eV)
data = struct( ...
    's',     [0.1, 0.08, 0.06], ...
    'p',     [0.6, 0.9, 1.1], ...
    'd',     [2.0, 1.5, 1.2]);

energies = [20, 40, 60];  % eV

if startsWith(orbital, 'p')
    sigma_vals = data.p;
elseif startsWith(orbital, 'd')
    sigma_vals = data.d;
else
    sigma_vals = data.s;
end

sigma = interp1(energies, sigma_vals, photon_energy, 'linear', 'extrap');
end