function [smeared_intensity, E_range] = ARPES_simulation_Advanced(eigenbands, SurfaceOrb, Ef, sigma, fidelity, temperature, photonE, polarization, IncidentTheta, IncidentPhi, PolarizationAngle)
% Inputs:
%   orb_proj: [Nk x Nb x Natom x Norb]
%   energies: [Nk x Nb] band energies (eV)
%   photon_energy: UV source (eV)
%   polarization: polarization vector or 'unpolarized'
%   T: temperature in Kelvin
%   Ef: Fermi level (eV)

% Incident photon direction (must be normalized)
    theta_inc = deg2rad(IncidentTheta);
    phi = deg2rad(IncidentPhi);
    k_photon = [sin(theta_inc)*cos(phi); sin(theta_inc)*sin(phi); cos(theta_inc)];
    k_photon = k_photon / norm(k_photon);  % Ensure unit vector
% Define a reference vector not parallel to k_photon to construct s-polarization
    if abs(dot(k_photon, [0; 0; 1])) < 0.99
        ref = [0; 0; 1];  % Use z-axis
    else
        ref = [0; 1; 0];  % Use y-axis to avoid collinearity
    end
% Construct s- and p-polarization unit vectors
    e_s = cross(k_photon, ref);
    e_s = e_s / norm(e_s);  % s-polarized: ? to k_photon and in-plane
    e_p = cross(e_s, k_photon);  % p-polarized: ? to both s and k
    e_p = e_p / norm(e_p);  % Ensure unit vector
% CHOOSE POLARIZATION VECTOR
switch lower(polarization)
    case 'lvp'
        efield = e_s;
    case 'lhp'
        efield = e_p;
    case 'lap'
        efield = cos(PolarizationAngle) * e_s + sin(PolarizationAngle) * e_p;
    case 'rcp'  % Right-handed circular polarization
        efield = 1/sqrt(2) * (e_s + 1i * e_p);
    case 'lcp'  % Left-handed circular polarization
        efield = 1/sqrt(2) * (e_s - 1i * e_p);
    case 'unpolarized'  % no polarization
    otherwise
        error('Unknown polarization type');
end

    [ nkpoints, nbands, ~, ~ ] = size(SurfaceOrb);
%USe the fidelity to create a linespace for high resolution enrgy smearing
    E_range = linspace(min(eigenbands(:)) - 1, max(eigenbands(:)) + 1, fidelity); % energy axis
    smeared_intensity = zeros(nkpoints, length(E_range));
    
    target_orbs = 2:9;
% Basis directions of orbitals
    orbital_dirs = struct(...
        's', [0; 0; 0], ...
        'px', [1; 0; 0], ...
        'py', [0; 1; 0], ...
        'pz', [0; 0; 1], ...
        'dxy', [1; 1; 0], ...
        'dxz', [1; 0; 1], ...
        'dyz', [0; 1; 1], ...
        'dz2', [0; 0; 1], ...
        'dx2y2', [1; -1; 0] ...
    );
% Normalize orbital vectors
    orb_names = fieldnames(orbital_dirs);
    for i = 1:numel(orb_names)
        orbital_dirs.(orb_names{i}) = orbital_dirs.(orb_names{i}) / norm(orbital_dirs.(orb_names{i}));
    end
    M = zeros(numel(orb_names));

%Calculate Intensities
if ischar(polarization) && strcmp(polarization, 'unpolarized')
    fprintf('Using unpolarized light source \n')
    intensity_sum = zeros(nkpoints, nbands);
    for e = 1:2
        if e == 1
           efield = e_s;
        else
           efield = e_p;
        end
        % For each orbital, compute projection
            for i = 1:numel(orb_names)
                orb = orb_names{i};
                dir = orbital_dirs.(orb);
                % Dipole matrix element (up to constants)
                    M(i) = abs(dot(efield, dir))^2;
            end
        weights = get_weights(polarization, photonE, orb_names);
        intensity_sum = intensity_sum + weighted_projection(SurfaceOrb, weights, target_orbs);
    end
    intensity = intensity_sum / 2;
else
    fprintf('Using polarized light source \n')
    % For each orbital, compute projection
        for i = 1:numel(orb_names)
            orb = orb_names{i};
            dir = orbital_dirs.(orb);
            % Dipole matrix element (up to constants)
                M(i) = abs(dot(efield, dir))^2;
        end
    weights = get_weights(polarization, photonE, orb_names);
    intensity = weighted_projection(SurfaceOrb, weights, target_orbs);
end

% Fermi–Dirac cutoff (optional, set to 1 for T=0)
    T = temperature; k_B = 8.617333e-5;  % eV/K
    fermi_func = @(E) 1 ./ (1 + exp((E - Ef) / (k_B * T)));

    for k = 1:nkpoints
        for b = 1:nbands
            E = eigenbands(k, b);

            % Fermi-Dirac weight
            f = fermi_func(E);

            % Gaussian broadening
            g = exp(-(E_range - E).^2 / (2 * sigma^2)) / (sqrt(2 * pi) * sigma);

            % Add contribution to intensity
            smeared_intensity(k, :) = smeared_intensity(k, :) + intensity(k,b) .* f .* g;
        end
    end
fprintf('Intensity mapped \n')
end

function weights = get_weights(pol, photon_energy, orb_names)
% Orbital weights = dipole matrix element * directionality
px = pol(1); py = pol(2); pz = pol(3);
weights = zeros(9,1);

for i = 1:9
    sigma = yeh_lindau_cross_section(photon_energy, orb_names{i});
    switch orb_names{i}
        case 'px', weights(i) = sigma * abs(px);
        case 'py', weights(i) = sigma * abs(py);
        case 'pz', weights(i) = sigma * abs(pz);
        case 'dxy', weights(i) = sigma * abs(px * py);
        case 'dyz', weights(i) = sigma * abs(py * pz);
        case 'dz2', weights(i) = sigma * abs(pz);
        case 'dxz', weights(i) = sigma * abs(px * pz);
        case 'dx2y2', weights(i) = sigma * abs(px^2 - py^2);
        otherwise, weights(i) = sigma * 0.1;
    end
end
end

function I = weighted_projection(orb_proj, weights, target_orbs)
[Nk, Nb, Natom, ~] = size(orb_proj);
I = zeros(Nk, Nb);
for k = 1:Nk
    for b = 1:Nb
        val = 0;
        for a = 1:Natom
            for o = target_orbs
                val = val + orb_proj(k,b,a,o) * weights(o);
            end
        end
        I(k,b) = abs(val)^2;
    end
end
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