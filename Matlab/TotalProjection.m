clear all;

%Data Aqcuisition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%Ef = 6.3545; %for manual input of fermi energy, be sure to toggle off DOSCAR
ss = 4 ; %ss =1 for w/o SOC and 4 for with SOC
Scolumn = 2 ; %number of columns for orbital projection 5 for all, 4 for d, 3 for p, 2 for s.
norbs = 9;
spins = 3;
NumSpin=4; % 1 for x, 2 for y, 3 for z, 4 for all

Home = pwd;

cd ../

%POSCAR Read
fprintf(' Reading POSCAR \n');
[ STRUCTURE ] = import_poscar( 'POSCAR' );%This is a function from VASPLAB, make sure it is in your path

%DOSCAR Read
fprintf(' Reading DOSCAR \n');
% [Energies, Total DOS, Ef, Projected DOS ] = import_doscar('DOSCAR');
[EDOS, TDOS, Ef, PDOS ] = import_doscar('DOSCAR');

cd (Home)

% KPOINTS Read
fprintf(' Reading KPOINTS \n');
[KDATA] = import_kpoints('KPOINTS');

% EIGENVAL Read
fprintf(' Reading EIGENVAL \n');
%[ eigenvalues, KPOINTSwieghts, nkpoints, nbands, nelectrons ] = import_eigenval('EIGENVAL');
[ eigenvalues, KPOINTSwieghts, ~, ~, nelectrons ] = import_eigenval('EIGENVAL');
eigenbands = eigenvalues - Ef;

%PROCAR Read
fprintf(' Reading PROCAR ');
fprintf('\n');
fidr = fopen('PROCAR', 'r');
tline = fgetl(fidr); 
tline = fgetl(fidr);
a = textscan(tline,'%s %s %s %d %s %s %s %d %s %s %s %d');
nbands = a{1,8};
nkpoints = a{1,4};
nions = a{1,12};

Orb = zeros(nkpoints, nbands, nions,  norbs);
Spin = zeros(nkpoints, nbands, nions,  spins*2);
Full = zeros(nkpoints, nbands, nions, 4);
kband=zeros(nkpoints,nbands);
bands=zeros(nkpoints,nbands);

tline = fgetl(fidr);
nlines = (nions+1)*ss+1;
for ii = 1:nkpoints
  disp(ii);
  tline = fgetl(fidr);
  tline = fgetl(fidr);
  for jj = 1:nbands
    tline = fgetl(fidr);
    a = textscan(tline,'%s %s %s %s %f %s %s %s');
    kband(ii, jj) = ii;
    bands(ii, jj) = a{1,5};
    tline = fgetl(fidr);  
        for kk = 1:nlines % surface projection
              tline = fgetl(fidr); 
              if ( kk > 1 && kk <= 1+nions )%for s_y | kk == 43)% | kk == 42 | kk == 43) kk=#of the line +1
                  a = textscan(tline,'%s %f %f %f %f %f %f %f %f %f %f');
                  ion = kk-1;
                  % Orbital order: [s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]
                  %                 1   2   3   4    5    6    7     8     9
                  Orb(ii,jj,ion,1) = Orb(ii,jj,kk-1,1)+a{1,Scolumn};    %S orb
                  Orb(ii,jj,ion,2) = Orb(ii,jj,kk-1,2)+a{1,Scolumn+1};   %py orb
                  Orb(ii,jj,ion,3) = Orb(ii,jj,kk-1,3)+a{1,Scolumn+2};   %pz orb
                  Orb(ii,jj,ion,4) = Orb(ii,jj,kk-1,4)+a{1,Scolumn+3};   %px orb
                  Orb(ii,jj,ion,5) = Orb(ii,jj,kk-1,5)+a{1,Scolumn+4};  %Dxy orb
                  Orb(ii,jj,ion,6) = Orb(ii,jj,kk-1,6)+a{1,Scolumn+5};  %Dyz orb
                  Orb(ii,jj,ion,7) = Orb(ii,jj,kk-1,7)+a{1,Scolumn+6};  %Dz2 orb
                  Orb(ii,jj,ion,8) = Orb(ii,jj,kk-1,8)+a{1,Scolumn+7};  %Dxz orb
                  Orb(ii,jj,ion,9) = Orb(ii,jj,kk-1,9)+a{1,Scolumn+8};  %Dx2 orb
                  Full(ii,jj,ion,4) = Full(ii,jj,kk-1,4)+a{1,Scolumn+9};    %Total atom
              end
              
%Spin contributions and directions
             %x
                   if ( kk>=1+1*(nions+1)+1 && kk<=1+1*(nions+1)+nions)
                      ion = kk - (1+1*(nions+1)) ;
                      a = textscan(tline,'%s %f %f %f %f %f %f %f %f %f %f');
                      if (a{1,Scolumn+9} >= 0) %  for spin up
                          Spin(ii,jj,ion,1)=Spin(ii,jj,ion,1)+a{1,Scolumn+9};   %xspinup
                      end
                      if (a{1,Scolumn+9} < 0) %  for spin down
                          Spin(ii,jj,ion,2)=Spin(ii,jj,ion,2)-a{1,Scolumn+9};   %xspindn
                      end
                   end
             %y
                   if ( kk>=1+2*(nions+1)+1 && kk<=1+2*(nions+1)+nions)
                      ion = kk - (1+2*(nions+1)) ;
                      a = textscan(tline,'%s %f %f %f %f %f %f %f %f %f %f');
                      if (a{1,Scolumn+9} >= 0) %  for spin up
                          Spin(ii,jj,ion,3)=Spin(ii,jj,ion,3)+a{1,Scolumn+9};   %yspinup
                      end
                      if (a{1,Scolumn+9} < 0) %  for spin down
                          Spin(ii,jj,ion,4)=Spin(ii,jj,ion,4)+a{1,Scolumn+9};   %yspindn
                      end
                   end
             %z
                   if ( kk>=1+3*(nions+1)+1 && kk<=1+3*(nions+1)+nions)
                      ion = kk - (1+3*(nions+1)) ;
                      a = textscan(tline,'%s %f %f %f %f %f %f %f %f %f %f');
                      if (a{1,Scolumn+9} >= 0) %  for spin up
                          Spin(ii,jj,ion,5)=Spin(ii,jj,ion,5)+a{1,Scolumn+9};   %zspinup
                      end
                      if (a{1,Scolumn+9} < 0) %  for spin down
                          Spin(ii,jj,ion,6)=Spin(ii,jj,ion,6)+a{1,Scolumn+9};   %zspindn
                      end
                   end
        end
     tline = fgetl(fidr);
  end
  tline = fgetl(fidr);
end
Full(:,:,:,1) = Orb(:,:,:,1);   %S orb
Full(:,:,:,2) = Orb(:,:,:,2)+Orb(:,:,:,3)+Orb(:,:,:,4); %P orb
Full(:,:,:,3) = Orb(:,:,:,5)+Orb(:,:,:,6)+Orb(:,:,:,7)+Orb(:,:,:,8)+Orb(:,:,:,9);   %D orb

%OAM Projection Calculation

fprintf(' Calculating OAM ');
fprintf('\n');
  % Orbital order: [s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]
  %                 1   2   3   4    5    6    7     8     9
OAMz = zeros(nkpoints, nbands, nions, 3); %1 = p, 2 = d, 3 = total%

m_p = [+1, 0, -1];
m_d = [-2, -1, 0, +1, +2];

for ii = 1:nkpoints
  disp(ii);
  for jj = 1:nbands
        Lz = 0;
        for i = 1:nions
            % p-orbital contributions (orbitals 2–4)
            p_proj = squeeze(Orb(ii, jj, i, 2:4));
            OAMz(ii,jj,i,1) = sum(m_p .* (abs(p_proj).^2)');
            
            % d-orbital contributions (orbitals 5–9)
            d_proj = squeeze(Orb(ii, jj, i, 5:9));
            OAMz(ii,jj,i,2) = sum(m_d .* (abs(d_proj).^2)');

            % Total atomic contribution
            OAMz(ii,jj,i,3) = Lz + OAMz(ii,jj,i,1) + OAMz(ii,jj,i,2);
        end
   end
end
kspace = kband;

%%
%Atomic selector
%input
%     atoms = [7,17];
%     atoms = [22,11];
%     atoms = [20,35];
%     atoms = [4,12];
%     atoms = [2,3,7,10];
%     atoms = [7,8,11,12]; 
%     atoms = [8,10,22,11]; %top surface
%     atoms = [3,7,11,4,8,12];
%     atoms = [3,7,11];
%     atoms = [2,6,10];
%     atoms = [2,6,10,1,5,9];
%     atoms = [1,5,9]; %bottom surface
    atoms = [1:nions]; %all atoms

%     atoms = [6,7,8]; %PdTe2 on Bi2Se3

% Orbital order: [s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]
%                 1   2   3   4    5    6    7     8     9

%Preallocate
Sband= zeros(nkpoints, nbands);
Pyband= zeros(nkpoints, nbands);
Pzband= zeros(nkpoints, nbands);
Pxband= zeros(nkpoints, nbands);
Dxyband= zeros(nkpoints, nbands);
Dyzband= zeros(nkpoints, nbands);
Dz2band= zeros(nkpoints, nbands);
Dxzband= zeros(nkpoints, nbands);
Dx2band= zeros(nkpoints, nbands);

xSpinup = zeros(nkpoints, nbands);
xSpindn = zeros(nkpoints, nbands);
ySpinup = zeros(nkpoints, nbands);
ySpindn = zeros(nkpoints, nbands);
zSpinup = zeros(nkpoints, nbands);
zSpindn = zeros(nkpoints, nbands);

OAMtotal = zeros(nkpoints, nbands);

SurfaceOrb = zeros(nkpoints, nbands, length(atoms),  norbs);
SurfaceSpin = zeros(nkpoints, nbands, length(atoms),  3);

for ll = 1:length(atoms)
    ion = atoms(ll);
    Sband(:,:) = Sband(:,:) + Orb(:,:,ion,1);
    Pyband(:,:) = Pyband(:,:) + Orb(:,:,ion,2);
    Pzband(:,:) = Pzband(:,:) + Orb(:,:,ion,3);
    Pxband(:,:) = Pxband(:,:) + Orb(:,:,ion,4);
    Dxyband(:,:) = Dxyband(:,:) + Orb(:,:,ion,5);
    Dyzband(:,:) = Dyzband(:,:) + Orb(:,:,ion,6);
    Dz2band(:,:) = Dz2band(:,:) + Orb(:,:,ion,7);
    Dxzband(:,:) = Dxzband(:,:) + Orb(:,:,ion,8);
    Dx2band(:,:) = Dx2band(:,:) + Orb(:,:,ion,9);
    
    xSpinup(:,:) = xSpinup(:,:) + Spin(:,:,ion,1);
    xSpindn(:,:) = xSpindn(:,:) + Spin(:,:,ion,2);
    ySpinup(:,:) = ySpinup(:,:) + Spin(:,:,ion,3);
    ySpindn(:,:) = ySpindn(:,:) + Spin(:,:,ion,4);
    zSpinup(:,:) = zSpinup(:,:) + Spin(:,:,ion,5);
    zSpindn(:,:) = zSpindn(:,:) + Spin(:,:,ion,6);
    
    OAMtotal(:,:) = OAMtotal(:,:) + OAMz(:,:,ion,3);
    
    SurfaceOrb(:,:,ion,:) = Orb(:,:,ion,:);
    SurfaceSpin(:,:,ion,1) = Spin(:,:,ion,1)+Spin(:,:,ion,2);
    SurfaceSpin(:,:,ion,2) = Spin(:,:,ion,3)+Spin(:,:,ion,4);
    SurfaceSpin(:,:,ion,3) = Spin(:,:,ion,5)+Spin(:,:,ion,6);
end
OAMup = max(OAMtotal,0);
OAMdn = min(OAMtotal,0);
Ptotal = Pxband + Pyband + Pzband;
Pxy = Pxband + Pyband;
Dtotal = Dx2band + Dxyband  + Dxzband + Dyzband + Dz2band;

kspace = kband;


%% 
%Total Density of States Plotting
figure, 
box on, hold on; 

plot(EDOS,TDOS)

% for i = 1:length(STRUCTURE.coords(:,1))
%     hold on
% %     plot(PDOS(:,1,i))
% end


%%
%ARPES Projections
    %inputs
      %Plotting details
        ymin = -3;
        ymax = 1;
        fidelity = 25000;
        sigma = 0.04;          %Gaussian Broadening factor
        gamma = 0.1;            %Lorentzian Broadening factor
        temperature = 15;       %Temperature in Kelvin
        photonE = 11;           %Photon energy in eV
        IncidentTheta = 45;    %Angle between light source and surface normal
        IncidentPhi = 0;       %Angle between and light source and x direction in-plane
        PolarizationAngle = 0;       %Angle of light polarization between 0 's' and 90 'p', for 'LAP'
        ExpEfShift = 0.8;       %Angle of light polarization between 0 's' and 90 'p', for 'LAP'
        LSscale = 0.01;

      %Polazization  
        polarization = 'unpolarized';
%         polarization = 'LHP';       %p incident polarization
%         polarization = 'LVP';       %s incident polarization
%         polarization = 'RCP';       %right handed circular polarization
%         polarization = 'LCP';       %left handed circular polarization

%         polarization = 'LAP';     %for arbitrarily angled linearly


%         PESold = PES;
%Simple ARPES simulation
%         [PES1,E_range] = ARPES_simulation_Basic(eigenbands, SurfaceOrb, ExpEfShift, sigma, fidelity, temperature, photonE);
%         [PES1,E_range] = ARPES_simulation_Basicplus(eigenbands, SurfaceOrb, ExpEfShift, sigma, fidelity, temperature, photonE);
%         [BASIC,~,~]=zscore_normalize_global(PES1);
%         PES = BASIC;
%         
%Advanced ARPES simulation
%         [PES2,E_range] = ARPES_simulation_Expert(eigenbands, SurfaceOrb, ExpEfShift, sigma, gamma, fidelity, temperature, photonE, polarization, IncidentTheta, IncidentPhi, PolarizationAngle);
        [PES2,E_range] = ARPES_simulation_Advanced(eigenbands, SurfaceOrb, ExpEfShift, sigma, fidelity, temperature, photonE, polarization, IncidentTheta, IncidentPhi, PolarizationAngle);
% %         [PES2,E_range] = ARPES_simulation_SOC(eigenbands, SurfaceOrb, SurfaceSpin, ExpEfShift, sigma, gamma, fidelity, temperature, photonE, polarization, IncidentTheta, IncidentPhi, PolarizationAngle, LSscale);
        [Advanced,~,~]=zscore_normalize_global(PES2);        
        PES = Advanced;

%         PES = (BASIC+Advanced) / 2;
        [PES,~,~]=zscore_normalize_global(PES);

%Advanced ARPES simulation Circular Dichroism
%         polarization = 'RCP';       %right handed circular polarization
%         [RCP,E_range] = ARPES_simulation_Expert(eigenbands, SurfaceOrb, ExpEfShift, sigma, gamma, fidelity, temperature, photonE, polarization, IncidentTheta, IncidentPhi, PolarizationAngle);
%         polarization = 'LCP';       %left handed circular polarization
%         [LCP,E_range] = ARPES_simulation_Expert(eigenbands, SurfaceOrb, ExpEfShift, sigma, gamma, fidelity, temperature, photonE, polarization, IncidentTheta, IncidentPhi, PolarizationAngle);
%         PES = RCP-LCP;

fprintf('Plotting photoemission spectrum \n')
        [PES,~,~]=zscore_normalize_global(PES);
% Plot: energy vs k-point index
        figure, box on, hold on; 
        imagesc(1:nkpoints, E_range, PES');
        
    % Set custom x-ticks and labels
        interval = KDATA.num_kpts;  % Define the interval for symmetry points
        k_labels = KDATA.k_labels;
        k_points = 0:interval:(interval * (length(k_labels)-1)); % Positions for Gamma, M, K, Gamma
        set(gca,  'XTick', k_points, 'XTickLabel', k_labels);
        xlabel('Momentum (k)');
    % Set custom y-ticks and labels
        set(gca, 'YMinorTick', 'on','YDir','normal');
        set(gca, 'FontSize', 14, 'linewidth', 2, 'TickDir', 'out');
        ylabel('Energy (eV)');
        title('Simulated ARPES / Photoemission Spectrum');
        colormap('gray');
        set(gca,'CLim',[0 5])
        colorbar;
        axis([0, nkpoints, ymin, ymax]) 

% imagesc(PES'); colorbar; xlabel('k-point'); ylabel('band index');
% title('ARPES intensity with matrix elements & smearing');    
%


%%
%Band Structure Plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INPUTS
%         figure, hold on, box on, axis square; 
        figure, box on, hold on; 
        
        %colorbar('vertical');
        fprintf('Plot band \n')
        ymin = double(-3);
        ymax = double(1);
        ytickformat('%.1f')
        size = 250;     %for spin and atoms
        OAMsize = 3000;  %for OAM

        %Set x-ticks and labels for symmetry points at intervals
        interval = KDATA.num_kpts;  % Define the interval for symmetry points
        k_labels = KDATA.k_labels;
%         k_labels = {'M','Y', 'G','X','M'}; % Labels for symmetry points
        k_points = 0:interval:(interval * (length(k_labels)-1)); % Positions for Gamma, M, K, Gamma
        kband = kspace;
        
        %Set x-ticks and labels for k-space using crystal structure
%         interval = KDATA.num_kpts;  % Define the interval for symmetry points
%         k_labels = zeros(KDATA.segments+1,1);
%         klat = [norm(STRUCTURE.reclattice(1,:)),norm(STRUCTURE.reclattice(2,:)),norm(STRUCTURE.reclattice(3,:))];
%         for j = 1:length(KDATA.k_labels)
%             for k = 1:3
%                 kpt = norm(KDATA.kpts(j,1));
%                 k_labels(j) = norm(klat*kpt);
%             end
%             for k = 1:nbands
%                 kband(:,j) = linespace;
%             end
%         end

% 
%         k_points = 0:interval:(interval * (length(k_labels)-1)); % Positions for Gamma, M, K, Gamma
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plots
      
% %         Total Orbitals 
%             scatter(kband(:), bands(:)-Ef, abs(Ptotal(:)/3 * size) + eps, 'r', 'filled')        
%             scatter(kband(:), bands(:)-Ef, abs(Dtotal(:)/5 * size) + eps, 'r', 'filled')    
% 
% %         Px,y,z Orbitals
%             scatter(kband(:), bands(:)-Ef, abs(Pxy(:)/2 * size) + eps, 'b', 'filled')        
%         
%             scatter(kband(:), bands(:)-Ef, abs(Pxband(:) * size) + eps, 'r', 'filled')
%             scatter(kband(:), bands(:)-Ef, abs(Pyband(:) * size) + eps, 'b', 'filled')
%             scatter(kband(:), bands(:)-Ef, abs(Pzband(:) * size) + eps, 'g', 'filled')
            

%             scatter(kband(:), bands(:)-Ef, abs(Dtotal(:)/5 * size) + eps, 'r', 'filled')
%             scatter(kband(:), bands(:)-Ef, abs(Pxy(:)/2 * size) + eps, 'b', 'filled')  
%                     
% %         D Orbitals
%             scatter(kband(:), bands(:)-Ef, abs(Dx2band(:) * size) + eps, 'c', 'filled')
%             scatter(kband(:), bands(:)-Ef, abs(Dxyband(:) * size) + eps, 'r', 'filled')
%             scatter(kband(:), bands(:)-Ef, abs(Dxzband(:) * size) + eps, 'b', 'filled')
%             scatter(kband(:), bands(:)-Ef, abs(Dyzband(:) * size) + eps, 'g', 'filled')
%             scatter(kband(:), bands(:)-Ef, abs(Dz2band(:) * size) + eps, 'm', 'filled')
%            
% %         Atom Specific Bands
%             C = {'r','b','m','m','o','c','k'};  %Array of color options for iterative plotting
%             for i=1:length(atoms(:))
%                 Atomband = Full(:,:,atoms(i),4);
%                 scatter(kband(:), bands(:)-Ef, abs(Atomband(:) * size) + eps, C{i}, 'filled')
%             end  
%             
% %         Spin Up/Down x
%             scatter(kband(:), bands(:)-Ef, abs(xSpinup(:) * size) + eps, 'r', 'filled')
%             scatter(kband(:), bands(:)-Ef, abs(xSpindn(:) * size) + eps, 'b', 'filled')
% 
%         Spin Up/Down y
%             scatter(kband(:), bands(:)-Ef, abs(ySpinup(:) * size) + eps, 'r', 'filled')
%             scatter(kband(:), bands(:)-Ef, abs(ySpindn(:) * size) + eps, 'b', 'filled')
%                         
% %         Spin Up/Down z
%             scatter(kband(:), bands(:)-Ef, abs(zSpinup(:) * size) + eps, 'g', 'filled')
%             scatter(kband(:), bands(:)-Ef, abs(zSpindn(:) * size) + eps, 'm', 'filled')
% 
% %         OAMz
%             scatter(kband(:), bands(:)-Ef, abs(OAMup(:) * OAMsize) + eps, 'g', 'filled')
%             scatter(kband(:), bands(:)-Ef, abs(OAMdn(:) * OAMsize) + eps, 'm', 'filled')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Graph settings
        axis([0, nkpoints, ymin, ymax])    
        ytickformat('%.1f')
        % Set custom x-ticks and labels
        set(gca, 'XTick', k_points, 'XTickLabel', k_labels);
        % Set custom y-ticks and labels
        set(gca, 'YMinorTick', 'on');
        
        % Set axis properties
        set(gca, 'FontSize', 14, 'linewidth', 2, 'TickDir', 'out')

        % Set axis labels
        xlabel('Momentum (k)');
        ylabel('Energy (eV)');
        
        % ylim([ymin ymax])

        % Add horizontal line at Fermi energy (zero energy level)
%         line(xlim(), [0, 0], 'LineStyle', '-.', 'LineWidth', 1, 'Color', 'k')
        
        % Add vertical lines at specific k-points
        for k_point = k_points
            line([k_point k_point], [ymin ymax], 'LineStyle', '-', 'LineWidth', 1, 'Color', 'k')
        end

        % Plot data
        for i = 1:nbands
           plot(eigenbands(:, i), 'color', 'Black', 'LineWidth', 1)
        end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%