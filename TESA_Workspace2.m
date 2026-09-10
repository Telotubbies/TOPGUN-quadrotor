%% TESA_Manual_Simulation.m (For Manual Gain Testing and Trajectory Plotting)
clc; clear; close all;

%% ===============================
%  LOAD REFERENCE DATA
% ===============================
load data.mat;

%% ===============================
%  DRONE / PLANT PARAMETERS
% ===============================
m    = 4.34; 
g    = 9.81;
dIdt = zeros(3,3);
I    = eye(3,3);
I(1,1) = 0.0820;
I(2,2) = 0.0845;
I(3,3) = 0.1377;
L      = 0.315;
cf = 8.004e-4;

K = [ 1   1   1   1;
      0  -L   0   L;
      L   0  -L   0;
     -cf  cf -cf  cf ];

assignin('base','m',m);
assignin('base','g',g);
assignin('base','dIdt',dIdt);
assignin('base','I',I);
assignin('base','L',L);
assignin('base','cf',cf);
assignin('base','K',K);

%% ===============================
%  INPUT STRUCTURE FOR SIMULINK
% ===============================
time = 0:timestep:SimulationTime;
inputStructure.time = time';

sig = {X,Y,Z,dX,dY,dZ,ddX,ddY,ddZ, ...
       dddX,dddY,dddZ,ddddX,ddddY,ddddZ,Psi,dPsi,ddPsi};

for k = 1:18
    inputStructure.signals(k).values = sig{k}';
    inputStructure.signals(k).dimensions = 1;
end
assignin('base','inputStructure',inputStructure);

%% ===============================
%  CONTROLLER BASE GAINS
% ===============================
Kp_base = diag([107.5, 107.5, 107.0]);    
Kv_base = diag([107.0, 107.0, 107.0]);    
Kr_base = diag([102.0, 12.0, 5.0]);   
Kw_base = diag([0.8, 0.8, 1.2]);    

%% ===============================
%  MANUAL GAIN SCALES (USER INPUT)
%  คุณสามารถเปลี่ยนค่าในบรรทัดถัดไปเพื่อทดสอบ Gain ใหม่ได้เลย
% ===============================
%Scale for Kp | Scale for Kv | Scale for Kr | Scale for Kw
s = [0.1000 0.4468 0.1000 0.1000]; 
%*********************************************************

s1 = s(1);
s2 = s(2);
s3 = s(3);
s4 = s(4);

Kp_test = s1*Kp_base;
Kv_test = s2*Kv_base;
Kr_test = s3*Kr_base;
Kw_test = s4*Kw_base;

disp("=== Starting Manual Simulation ===");
fprintf("Testing Scales: [Kp=%.4f, Kv=%.4f, Kr=%.4f, Kw=%.4f]\n", s1, s2, s3, s4);

%% ===============================
%  SIMULATION AND PLOTTING
% ===============================

assignin('base','Kp',Kp_test);
assignin('base','Kv',Kv_test);
assignin('base','Kr',Kr_test);
assignin('base','Kw',Kw_test);

% Run Simulation
out = sim('quadrotorsmodel2.slx');

% Extract Results
Xe = out.yout{1}.Values.Data(:,1)';
Ye = out.yout{1}.Values.Data(:,2)';
Ze = out.yout{1}.Values.Data(:,3)';

Len = min(numel(X), numel(Xe));
Xe = Xe(1:Len); Ye = Ye(1:Len); Ze = Ze(1:Len);

% Calculate Final Error (Optional: เพื่อให้รู้ว่า Error ที่ได้คือเท่าไหร่)
e3d = sqrt((X(1:Len)-Xe(1:Len)).^2 + (Y(1:Len)-Ye(1:Len)).^2 + (Z(1:Len)-Ze(1:Len)).^2);
mean_error = mean(e3d);
fprintf("Mean 3D Position Error: %.4f meters\n", mean_error);
disp("================================");

% Plot Trajectory Comparison
figure;
plot3(X, Y, Z, 'k--', 'LineWidth', 1.5); hold on;
plot3(Xe, Ye, Ze, 'LineWidth', 1.5);
set(gca,'ZDir','reverse'); grid on; axis equal;
xlabel('x (m)'); ylabel('y (m)'); zlabel('-z (m)');
legend('Reference Trajectory','Actual Drone Path (Manual Gain)','Location','best');
title(sprintf('Reference vs Actual Trajectory (Manual Test, Error: %.2f m)', mean_error));
saveas(gcf, 'Manual_Trajectory_Comparison.png');