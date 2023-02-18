
function GJ8_Demo()
%% This function was created to test GJ8 using orbit propagation
% units: length [km], time [s], mass [kg]
mu = 398600.4415;  % value of (GM)
%Initialize State Vector
r0 = [7000,0,0];  % initial position: 7000 km from COM (about 600 km above surface of Earth)
v0 = [0,sqrt(mu./sqrt(sum(r0.^2))),0];  % speed required for circular orbit (rv^2 = GM)
%Compute initial acceleration from state vector
a0 = SecondOrderODE(0,r0',v0',mu);  % x' = transpose of x (turn into column vectors)
t0 = 0;
tf = 58290;
h = 60;
PosVel_0 = [r0';v0'];

options = odeset('RelTol',1e-13,'AbsTol',1e-15);
%% Use ODE113 as a reference point
tic;
[t,PosVel] = ode113(@FirstOrderODE,[t0:h:tf],PosVel_0,options,mu);
disp(['ODE113 Took ', num2str(toc),' seconds']);
%% Call Gauss Jackson 8
tic;
[t,Pos,Vel] = GJ8(@FirstOrderODE,@SecondOrderODE,[t0 tf],h,r0',v0',a0,options,mu);
disp(['Gauss-Jackson Took ', num2str(toc),' seconds']);

%% Now compute the truth!
tic;

[r,~] = keplerUniversal(repmat(r0',[1 length(t)]),repmat(v0',[1 length(t)]),t,mu);

disp(['Analytic Solution Took ', num2str(toc),' seconds']);


figure('color',[1 1 1]);
plot(t./60,sqrt(sum((Pos-r).^2,1)).*1e6,'k','linewidth',2); hold on;
plot(t./60,sqrt(sum((PosVel(:,1:3)'-r).^2,1)).*1e6,'r','linewidth',2);
xlabel('Time (min)'); ylabel('Position Magnitude Absolute Error (mm)');
title('Gauss-Jackson Eighth Order vs ODE 113 (2-Body Problem)');
grid on; axis square; legend('GJ8','ODE113','location','NW');

end

function eta = FirstOrderODE(t,posvel,mu)
eta = NaN(size(posvel));
rMag = sqrt(sum(posvel(1:3,:).^2));
nuR3 = -mu./rMag.^3;
eta(1,:) = posvel(4,:);
eta(2,:) = posvel(5,:);
eta(3,:) = posvel(6,:);
eta(4,:) = nuR3.*posvel(1,:);
eta(5,:) = nuR3.*posvel(2,:);
eta(6,:) = nuR3.*posvel(3,:);
end

function a = SecondOrderODE(t,r,v,mu)
a = NaN(size(r));
rMag = sqrt(sum(r.^2));
nuR3 = -mu./rMag.^3;
a(1,:) = nuR3.*r(1,:);
a(2,:) = nuR3.*r(2,:);
a(3,:) = nuR3.*r(3,:);
end