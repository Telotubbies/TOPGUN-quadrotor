%% TESA_AutoTune_Parallel_10k_fixed.m
clc; clear; close all;

%% -------------------------------
% Load Reference Trajectory
%% -------------------------------
load('data.mat');  % trajectory X,Y,Z,...

%% -------------------------------
% Drone / Plant Parameters
%% -------------------------------
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

%% -------------------------------
% Base Gains
%% -------------------------------
Kp_base = diag([1.5, 1.5, 1.0]);    
Kv_base = diag([1.0, 1.0, 1.0]);    
Kr_base = diag([1.0, 1.0, 1.0]);   
Kw_base = diag([0.8, 0.8, 1.2]);    

%% -------------------------------
% GA Parameters
%% -------------------------------
nVars = 4;               % 4 scale factors for Kp,Kv,Kr,Kw
lb = [0.1 0.1 0.1 0.1];  % lower bound
ub = [2 2 2 2];          % upper bound

% Start parallel pool if not already running
if isempty(gcp('nocreate'))
    parpool('local', 12); % ใช้ 12 workers
end

options = optimoptions('ga',...
    'PopulationSize',50,...
    'MaxGenerations',100,...
    'UseParallel',true,...
    'Display','iter',...
    'PlotFcn',{@gaplotbestf});

%% -------------------------------
% Fitness Function
%% -------------------------------
fitnessFunc = @(s) simulateQuadScaleParallel(s,Kp_base,Kv_base,Kr_base,Kw_base,X,Y,Z);

%% -------------------------------
% Run GA 10,000 times
%% -------------------------------
results = zeros(10000,4);
errors  = zeros(10000,1);

for i = 1:10000
    fprintf('=== GA Run %d / 10000 ===\n', i);
    [bestScale,bestErr] = ga(fitnessFunc,nVars,[],[],[],[],lb,ub,[],options);
    results(i,:) = bestScale;
    errors(i) = bestErr;

    % Save periodically
    if mod(i,10)==0
        save('GA_10k_Results.mat','results','errors');
    end
end

%% -------------------------------
% Fitness Function for Parallel GA
%% -------------------------------
function mean_error = simulateQuadScaleParallel(s,Kp_base,Kv_base,Kr_base,Kw_base,X,Y,Z)
    s1 = s(1); s2 = s(2); s3 = s(3); s4 = s(4);

    Kp_test = real(s1*Kp_base);
    Kv_test = real(s2*Kv_base);
    Kr_test = real(s3*Kr_base);
    Kw_test = real(s4*Kw_base);

    % สร้าง Simulink simulation input
    simIn = Simulink.SimulationInput('quadrotorsmodel2.slx');
    simIn = simIn.setVariable('Kp',Kp_test);
    simIn = simIn.setVariable('Kv',Kv_test);
    simIn = simIn.setVariable('Kr',Kr_test);
    simIn = simIn.setVariable('Kw',Kw_test);

    try
        % รัน simulation
        out = sim(simIn,'SaveOutput','on');
        Xe = out.yout{1}.Values.Data(:,1)';
        Ye = out.yout{1}.Values.Data(:,2)';
        Ze = out.yout{1}.Values.Data(:,3)';

        Len = min([numel(X),numel(Xe)]);
        Xe = Xe(1:Len); Ye = Ye(1:Len); Ze = Ze(1:Len);

        e3d = sqrt((X(1:Len)-Xe).^2 + (Y(1:Len)-Ye).^2 + (Z(1:Len)-Ze).^2);
        mean_error = mean(e3d);

        if ~isreal(mean_error) || ~isfinite(mean_error)
            mean_error = 1e6;  % penalize invalid simulation
        end
    catch
        mean_error = 1e6;  % penalize crash
    end
end
