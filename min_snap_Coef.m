function [trajectory_coef] = min_snap_Coef(waypoints,timepoints)


%% Minimum snap trajectoty to quadratic programming
%% P(t) = c1 + c2*t.^1 + c3*t.^2 + c3*t.^3 + ... + cqt.^(q-1) ; with q the max. power
%% P(t) = SUM(i=1 to q)(ci*t(i-1))
%% dnP(t)/dtn = SUM(i=n to q)(ci * (i-1)! / (i-n)! * t.^(i-n)) 
%% dnP(t)i*dnP(t)j = (i-1)!/(i-n)!*(j-1)!/(j-n)!*t(i+j-2n)
%% int dnP(t)i*dnP(t)j = (i-1)!/(i-n)!*(j-1)!/(j-n)!/(i+j-2n+1)*t(i+j-2n+1)
%% Element in H matrix = (i-1)!/(i-n)!/(i-n+1) * (j-1)!/(j-n)!/(j-n+1) * t.^(i+j-2n+2) *ci*cj
%% for minimum snap trajectory max. power of coef q_xyz = 8
%% for minimum snap trajectory max. power of coef q_psi = 4


%% quadratic programming in form (c' H c + f' c)
%% c is (3*q_xyz + q_psi)*m vector --> [x y z psi]


q_xyz = 8;
q_psi = 4;
m = length(timepoints) - 1;
H = zeros((3*q_xyz + q_psi)*m,(3*q_xyz + q_psi)*m);


order_diff_xyz = 4;
order_diff_psi = 2;

T =0;
for iter = 1 : m
    
    t1 = timepoints(iter);
    t2 = timepoints(iter+1);
    
    H_xyz = zeros(3*q_xyz,3*q_xyz);
    H_psi = zeros(q_psi,q_psi);

    %% Find H_xyz
    for i = order_diff_xyz  : q_xyz
        for j = order_diff_xyz  : q_xyz
            H_xyz(i,j) = factorial(i-1) / factorial(i - order_diff_xyz) * factorial(j-1) / factorial(j-order_diff_xyz)...
                /(i+j-2*order_diff_xyz+1).*(t2.^(i+j-2*order_diff_xyz+1) - t1.^(i+j-2*order_diff_xyz+1));

                 
            H_xyz(i + (1*q_xyz), j + (1*q_xyz)) = H_xyz(i,j);
            H_xyz(i + (2*q_xyz), j + (2*q_xyz)) = H_xyz(i,j);
          
        
        end



    end
    %% Find H_psi
    for i = order_diff_psi : q_psi
        for j = order_diff_psi : q_psi
            H_psi(i,j) = factorial(i-1) / factorial(i - order_diff_psi) * factorial(j-1) / factorial(j-order_diff_psi)...
                /(i+j-2*order_diff_psi+1).*(t2.^(i+j-2*order_diff_psi+1) - t1.^(i+j-2*order_diff_psi+1));
 
     
        end
    end

    H( (iter - 1)*(3*q_xyz + q_psi) + 1 : (iter - 1)*(3*q_xyz + q_psi) + 3*q_xyz , (iter - 1)*(3*q_xyz + q_psi) + 1 : (iter - 1)*(3*q_xyz + q_psi) + 3*q_xyz ) = H_xyz;
    H((iter - 1)*(3*q_xyz + q_psi) + 3*q_xyz + 1 : (iter - 1)*(3*q_xyz + q_psi) + (3*q_xyz + q_psi) , (iter - 1)*(3*q_xyz + q_psi) + 3*q_xyz + 1 : (iter - 1)*(3*q_xyz + q_psi) + (3*q_xyz + q_psi)) = H_psi;    

end

% clear H_xyz H_psi

%% Insert Boundary condition to constraint Aeq*c = beq


Aeq = [];
beq = [];



%% Insert boundary conditions for intermidiate timepoints -- summary 28 constraints for each intervall
for it = 2 : length(timepoints) - 1
    constraint = zeros(28 ,(3*q_xyz + q_psi)*m);
    polynomial_xyz = [];
    polynomial_psi = [];
    t = timepoints(it);
    for power = 0 : q_xyz-1
        polynomial_xyz = [t.^power polynomial_xyz];
    end
    for power = 0 : q_psi-1
        polynomial_psi = [t.^power polynomial_psi];
    end
    
    polynomial_xyz = fliplr(polynomial_xyz);
    polynomial_psi = fliplr(polynomial_psi);

    constraint(1, (it-2)*(3*q_xyz + q_psi)+1 : (it-2)*(3*q_xyz + q_psi)+8) = polynomial_xyz; %% x -axis
    constraint(2, (it-2)*(3*q_xyz + q_psi)+9 : (it-2)*(3*q_xyz + q_psi)+16) = polynomial_xyz; %% y -axis
    constraint(3, (it-2)*(3*q_xyz + q_psi)+17 : (it-2)*(3*q_xyz + q_psi)+24) = polynomial_xyz; %% z -axis
    constraint(4, (it-2)*(3*q_xyz + q_psi)+25 : (it-2)*(3*q_xyz + q_psi)+28) = polynomial_psi;

    constraint(5, (it-1)*(3*q_xyz + q_psi)+1 : (it-1)*(3*q_xyz + q_psi)+8) = polynomial_xyz;
    constraint(6, (it-1)*(3*q_xyz + q_psi)+9 : (it-1)*(3*q_xyz + q_psi)+16) = polynomial_xyz;
    constraint(7, (it-1)*(3*q_xyz + q_psi)+17 : (it-1)*(3*q_xyz + q_psi)+24) = polynomial_xyz;
    constraint(8, (it-1)*(3*q_xyz + q_psi)+25 : (it-1)*(3*q_xyz + q_psi)+28) = polynomial_psi;

    for diff = 1 : 2*(order_diff_xyz-1)
        polynomial_xyz = fliplr(polynomial_xyz);
        polynomial_xyz = polyder(polynomial_xyz);
        polynomial_xyz = fliplr(polynomial_xyz);
    
        constraint(3*(diff-1)+9 , (it-2)*(3*q_xyz + q_psi)+1+diff: (it-2)*(3*q_xyz + q_psi)+8) = polynomial_xyz;
        constraint(3*(diff-1)+9 , (it-1)*(3*q_xyz + q_psi)+1+diff : (it-1)*(3*q_xyz + q_psi)+8) = (-1).*polynomial_xyz;
        constraint(3*(diff-1)+10 , (it-2)*(3*q_xyz + q_psi)+9+diff : (it-2)*(3*q_xyz + q_psi)+16) = polynomial_xyz;
        constraint(3*(diff-1)+10 , (it-1)*(3*q_xyz + q_psi)+9+diff : (it-1)*(3*q_xyz + q_psi)+16) = (-1).*polynomial_xyz;
        constraint(3*(diff-1)+11 , (it-2)*(3*q_xyz + q_psi)+17+diff : (it-2)*(3*q_xyz + q_psi)+24) = polynomial_xyz;
        constraint(3*(diff-1)+11 , (it-1)*(3*q_xyz + q_psi)+17+diff : (it-1)*(3*q_xyz + q_psi)+24) = (-1).*polynomial_xyz;
    end

    for diff = 1 : 2*(order_diff_psi -1)

        polynomial_psi = fliplr(polynomial_psi);
        polynomial_psi = polyder(polynomial_psi);
        polynomial_psi = fliplr(polynomial_psi);

        constraint((diff-1)+27 , (it-2)*(3*q_xyz + q_psi)+25+diff : (it-2)*(3*q_xyz + q_psi)+28) = polynomial_psi;
        constraint((diff-1)+27 , (it-1)*(3*q_xyz + q_psi)+25+diff : (it-1)*(3*q_xyz + q_psi)+28) = (-1).*polynomial_psi;

    end
    Aeq = [Aeq; constraint];
    beq = [beq; waypoints(it,:).'];
    beq = [beq; waypoints(it,:).'];
    beq = [beq; zeros(20,1)];

end

%% Insert boundary at beginpoint
constraint = zeros(14 ,(3*q_xyz + q_psi)*m);

polynomial_xyz = [];
polynomial_psi = [];
t = timepoints(1);
for power = 0 : q_xyz-1
    polynomial_xyz = [t.^power polynomial_xyz];
end
for power = 0 : q_psi-1
    polynomial_psi = [t.^power polynomial_psi];
end

polynomial_xyz = fliplr(polynomial_xyz);
polynomial_psi = fliplr(polynomial_psi);

constraint(1,1:8) = polynomial_xyz;
constraint(2,9:16) = polynomial_xyz;
constraint(3,17:24) = polynomial_xyz;
constraint(4,25:28) = polynomial_psi;

for diff = 1 : order_diff_xyz-1

    polynomial_xyz = fliplr(polynomial_xyz);
    polynomial_xyz = polyder(polynomial_xyz);
    polynomial_xyz = fliplr(polynomial_xyz);
    
    constraint(3*(diff-1)+5,1+diff:8) = polynomial_xyz;
    constraint(3*(diff-1)+6,9+diff:16) = polynomial_xyz;
    constraint(3*(diff-1)+7,17+diff:24) = polynomial_xyz;
end

for diff = 1 : order_diff_psi -1
    polynomial_psi = fliplr(polynomial_psi);
    polynomial_psi = polyder(polynomial_psi);
    polynomial_psi = fliplr(polynomial_psi);

    constraint(14,25+diff:28) = polynomial_psi;

end

Aeq = [Aeq;constraint];
beq = [beq; waypoints(1,:).'];
beq = [beq; zeros(10,1)];



%% Insert boundary at endpoint
constraint = zeros(14 ,(3*q_xyz + q_psi)*m);

polynomial_xyz = [];
polynomial_psi = [];
t = timepoints(length(timepoints));
for power = 0 : q_xyz-1
    polynomial_xyz = [t.^power polynomial_xyz];
end
for power = 0 : q_psi-1
    polynomial_psi = [t.^power polynomial_psi];
end

polynomial_xyz = fliplr(polynomial_xyz);
polynomial_psi = fliplr(polynomial_psi);

constraint(1,(m-1)*(3*q_xyz+q_psi)+1 : (m-1)*(3*q_xyz+q_psi)+8) = polynomial_xyz;
constraint(2,(m-1)*(3*q_xyz+q_psi)+9 : (m-1)*(3*q_xyz+q_psi)+16) = polynomial_xyz;
constraint(3,(m-1)*(3*q_xyz+q_psi)+17 : (m-1)*(3*q_xyz+q_psi)+24) = polynomial_xyz;
constraint(4,(m-1)*(3*q_xyz+q_psi)+25 : (m-1)*(3*q_xyz+q_psi)+28) = polynomial_psi;

for diff = 1 : order_diff_xyz-1

    polynomial_xyz = fliplr(polynomial_xyz);
    polynomial_xyz = polyder(polynomial_xyz);
    polynomial_xyz = fliplr(polynomial_xyz);
    
    constraint(3*(diff-1)+5,(m-1)*(3*q_xyz+q_psi)+1+diff:(m-1)*(3*q_xyz+q_psi)+8) = polynomial_xyz;
    constraint(3*(diff-1)+6,(m-1)*(3*q_xyz+q_psi)+9+diff:(m-1)*(3*q_xyz+q_psi)+16) = polynomial_xyz;
    constraint(3*(diff-1)+7,(m-1)*(3*q_xyz+q_psi)+17+diff:(m-1)*(3*q_xyz+q_psi)+24) = polynomial_xyz;
end

for diff = 1 : order_diff_psi -1
    polynomial_psi = fliplr(polynomial_psi);
    polynomial_psi = polyder(polynomial_psi);
    polynomial_psi = fliplr(polynomial_psi);

    constraint(14,(m-1)*(3*q_xyz+q_psi)+25+diff:(m-1)*(3*q_xyz+q_psi)+28) = polynomial_psi;

end

Aeq = [Aeq;constraint];
beq = [beq; waypoints(length(timepoints),:).'];
beq = [beq; zeros(10,1)];

% clear polynomial_psi polynomial_xyz

trajectory_coef = quadprog(H,[],[],[],Aeq,beq);



trajectory_coef = reshape(trajectory_coef.', length(trajectory_coef)/m,m);


end