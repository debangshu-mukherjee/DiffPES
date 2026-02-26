function [intensity, E_range] = ARPES_simulation_Novice(eigenbands, SurfaceOrb, Ef, sigma, gamma, fidelity, temperature, photonE)
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

% orb_names = {'s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2'};
target_orb = 2:9;  % exclude 's' for now
% %interpollation
% if interp == 'yes'
%     
%     

% Fermi–Dirac cutoff (optional, set to 1 for T=0)
T = temperature; k_B = 8.617333e-5;  % eV/K
fermi_func = @(E) 1 ./ (1 + exp((E - Ef) / (k_B * T)));

% Example: enhance p-orbitals at low h?, d-orbitals at high h?
if photonE < 50
    orb_weights = [1 2 2 2 1 1 1 1 1];  % s, px, py, pz, d...
else
    orb_weights = [1 1 1 1 2 2 2 2 2];
end
fprintf('Constructing Orbital Weights \n')
for k = 1:nkpoints
    for b = 1:nbands
        E = eigenbands(k, b);
        preweight = SurfaceOrb(k, b, :, target_orb);        
        % Selective orbital weights
%         for i = 1:length(target_orb)
%             preweight = SurfaceOrb(k, b, :, target_orb(i))*orb_weights(target_orb(i));
%         end
        
        % Total orbital weight for selected orbitals (sum over atoms)
        weight = sum(sum(preweight, 4), 3);



        % Fermi-Dirac weight
        f = fermi_func(E);

        % Gaussian broadening
%         g = exp(-(E_range - E).^2 / (2 * sigma^2)) / (sqrt(2 * pi) * sigma);
        V = voigt_profile(E_range, E, sigma, gamma);

        % Add contribution to intensity
        intensity(k, :) = intensity(k, :) + weight .* f .* V;
    end
end
fprintf('Intensity mapped \n')
end

function V = voigt_profile(E, E0, sigma, gamma)
% VOIGT_PROFILE Calculates normalized Voigt profile at energy E
%   E     = array of energy values (e.g., linspace)
%   E0    = center energy (scalar or array)
%   sigma = Gaussian std. deviation (instrumental resolution) [eV]
%   gamma = Lorentzian HWHM (lifetime broadening) [eV]
%
%   V     = Voigt profile evaluated at E

    % Complex error function approach (Faddeeva)
    z = ((E - E0) + 1i*gamma) / (sigma * sqrt(2));
    V = real(faddeeva(z)) / (sigma * sqrt(2*pi));
end

function w = faddeeva(z)
%FADDEEVA Compute the Faddeeva function w(z) = exp(-z^2) * erfc(-i*z)
% using Weideman's rational approximation
% Valid for complex z
%
% Reference:
% Weideman, J.A.C., "Computation of the complex error function",
% SIAM J. Numer. Anal. 31(5), 1497–1518 (1994)

    persistent C D N

    if isempty(C)
        % Weideman's parameters
        N = 32;     % number of terms
        L = 6;      % optimization parameter (good for double precision)
        k = (1:N)';
        theta = (pi*(k - 0.5)/N);      % Chebyshev nodes
        t = L * tan(theta / 2);       % transformed nodes
        f = exp(-t.^2) .* (L^2 + t.^2);% target function at nodes
        C = (2/N) * real(fft([f; flipud(f)]));
        C = C(1:N);                   % only keep first N terms
        D = t;                        % store poles
    end

    % initialize result
    w = zeros(size(z));

    % vectorized evaluation
    for j = 1:numel(z)
        zj = z(j);
        sum_val = 0;
        for k = 1:N
            sum_val = sum_val + C(k) / (zj - 1i * D(k));
        end
        w(j) = (2 * sum_val) / sqrt(pi);
    end

    % final exponential factor
    w = w .* exp(-z.^2);
end