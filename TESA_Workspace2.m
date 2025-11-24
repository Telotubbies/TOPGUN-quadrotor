%% TESA_Train_Evo_fromBase.m
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
Kp_base = diag([1.5, 1.5, 1.0]);    
Kv_base = diag([2.0, 2.0, 2.0]);    
Kr_base = diag([8.0, 8.0, 10.0]);   
Kw_base = diag([0.8, 0.8, 1.2]);    

%% ===============================
%  EVO "TRAINING" PARAMETERS
% ===============================
popSize = 40;     % จำนวน candidate ต่อรุ่น
gens    = 30;     % จำนวนรุ่น (epoch)
eliteK  = 6;      % เก็บตัวดีที่สุดไว้กี่ตัว
sigma   = 0.25;   % mutation strength (log-space)

% init population in log-space
pop = log([ ...
    0.05 + 0.95*rand(popSize,1), ...
    0.05 + 0.95*rand(popSize,1), ...
    0.05 + 0.95*rand(popSize,1), ...
    0.05 + 0.95*rand(popSize,1) ...
]);

bestHist = zeros(gens,1);
bestScaleHist = zeros(gens,4);

disp("=== Start Evolution Training ===");

for gen = 1:gens
    errs = zeros(popSize,1);

    for i = 1:popSize
        s = exp(pop(i,:));  % scales

        Kp = s(1)*Kp_base;
        Kv = s(2)*Kv_base;
        Kr = s(3)*Kr_base;
        Kw = s(4)*Kw_base;

        assignin('base','Kp',Kp);
        assignin('base','Kv',Kv);
        assignin('base','Kr',Kr);
        assignin('base','Kw',Kw);

        try
            out = sim('quadrotorsmodel2.slx');
            Xe = out.yout{1}.Values.Data(:,1)';
            Ye = out.yout{1}.Values.Data(:,2)';
            Ze = out.yout{1}.Values.Data(:,3)';

            Len = min(numel(X), numel(Xe));
            e3d = sqrt((X(1:Len)-Xe(1:Len)).^2 + ...
                       (Y(1:Len)-Ye(1:Len)).^2 + ...
                       (Z(1:Len)-Ze(1:Len)).^2);

            errs(i) = mean(e3d);
            if any(~isfinite(e3d)), errs(i)=1e6; end

        catch
            errs(i) = 1e6;
        end
    end

    % sort (min error best)
    [errs,idx] = sort(errs);
    pop = pop(idx,:);

    bestHist(gen) = errs(1);
    bestScaleHist(gen,:) = exp(pop(1,:));

    fprintf("Gen %02d | best err=%.4f | scales=[%.3f %.3f %.3f %.3f]\n", ...
        gen, errs(1), bestScaleHist(gen,1),bestScaleHist(gen,2),bestScaleHist(gen,3),bestScaleHist(gen,4));

    % elitism
    elite = pop(1:eliteK,:);

    % mutate to make next gen
    newPop = elite;
    while size(newPop,1) < popSize
        parent = elite(randi(eliteK),:);
        child  = parent + sigma*randn(1,4);
        newPop = [newPop; child];
    end
    pop = newPop(1:popSize,:);
end

% best scales after training
best_scales = exp(pop(1,:));
best_err    = bestHist(end);

disp("===== TRAINING DONE =====");
disp(best_scales);
disp(best_err);

% plot training curve
figure;
plot(bestHist,'LineWidth',1.5); grid on;
xlabel('Generation'); ylabel('Best Mean 3D Error (m)');
title('Evolution Training Curve');

% simulate best and plot
Kp_best = best_scales(1)*Kp_base;
Kv_best = best_scales(2)*Kv_base;
Kr_best = best_scales(3)*Kr_base;
Kw_best = best_scales(4)*Kw_base;

assignin('base','Kp',Kp_best);
assignin('base','Kv',Kv_best);
assignin('base','Kr',Kr_best);
assignin('base','Kw',Kw_best);

out = sim('quadrotorsmodel2.slx');
Xe = out.yout{1}.Values.Data(:,1)';
Ye = out.yout{1}.Values.Data(:,2)';
Ze = out.yout{1}.Values.Data(:,3)';

Len = min(numel(X), numel(Xe));
Xe = Xe(1:Len); Ye = Ye(1:Len); Ze = Ze(1:Len);

figure;
plot3(X, Y, Z, 'k--', 'LineWidth', 1.5); hold on;
plot3(Xe, Ye, Ze, 'LineWidth', 1.5);
set(gca,'ZDir','reverse'); grid on; axis equal;
xlabel('x'); ylabel('y'); zlabel('-z');
legend('Reference','Best (Evo)','Location','best');
title('Reference vs Best (Evolution Training)');

