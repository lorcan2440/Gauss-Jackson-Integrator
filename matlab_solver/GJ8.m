function [t,y,dy,ddy] = GJ8(FirstOrderODE,SecondOrderODE,tspan,h,y0,dy0,ddy0,options,varargin)
%% Purpose:
% Gauss Jackson Eighth Order Solver for all ODE solver applications ...
% This function was created with orbit propagation specifically in mind,
% but written generally enough where you shouldn't have a problem running
% other second order ODE problems.
% 
% SEQUENCE:
% 1). Startup
% Adams-Bashforth-Moulton (ode113) is run initially at startup in order to
% predict the first 8 points (-4 from t0 and +4 from t0)
% a). The solution of ode113 is evaluated and checked to insure that the
% error tolerance specified by odeset('RelError') is respected
% 
% 2). Predictor
% The next point corresponding to t0+nh is predicted for y, dy, and ddy
%
% 3). Corrector
% The point corresponding to step two is evaluated to insure solution convergence
% for both dy and y. If convergence is not acheived, the force function
% (y''(t)) is re-evaluated and the values obtained from the predictor are
% updated and checked again for convergence
%
%
%% Inputs:
%---------
%FirstOrderODE          function Handle        This function is the first
%                                              order ODE which is needed to
%                                              run the single step predictor
%                                              This must be in the same
%                                              form needed to run a MATLAB
%                                              ODE solver as the ODE solver
%                                              is directly calling this
%                                              handle.
%
%SecondOrderODE         function Handle        This function is the second
%                                              order ODE which is needed to
%                                              run the Gauss Jackson
%                                              routine, it is slightly
%                                              different from the first
%                                              order ode function handle as
%                                              it does not need combined
%                                              inputs for y, and dy.  Each
%                                              input must be specified
%                                              separately, even if they are
%                                              not used.
%
%tspan                  [1 x 2]                Initial Time and Final Time
%                                              in which the propagation is
%                                              desired
%                                              [t0 tf]
%
%h                      double                 Timestep ... t_n = t0 + nh                  
%
%
%y0                      [M x 1]               Initial States for the First
%                                              Order ODE Solver
%                                              y(0)  = ??
%
%dy0                     [M x 1]               Initial States for the First
%                                              Order ODE Solver
%                                              y'(0) = ??
%
%ddy0                    [O x 1]               Initial States for the
%                                              Second Order ODE Solver
%                                               y''(0) = ??
%
%options                struct                 Use the same options
%                                              as required by any other ODE
%                                              solver
%
%varargin                                      Any other information in which
%                                              the FirstOrder and
%                                              SecondOrder ODEs need to run
%
%% Outputs:
%----------
% t                     [N x 1]                 Time array of any one
%                                               dimension
%
% y                     [N x M]                 y(t)
%
%dy                     [N x M]                 dy(t)
%
%ddy                    [N x M]                 ddy(t)
%
%% References:
%-------------
% M. Berry, Healy, L. "Implementation of Gauss-Jackson Integration for
% Orbit Propagation". The Journal of Astronautical Sciences. V 52. N 3.
% Sept. 2004. pp 331-357.
%
%%-------------------------------------------------------------------------
% Programmed by Darin C. Koblick 05-03-2012
%%-------------------------------------------------------------------------

format longg;
fprintf('.25g')

%% Constant Declarations
% Initialize Coefficient Tables
% Eighth-Order Gauss-Jackson Coefficients in Ordinate Form ajk
%-4                           -3                            -2                           -1                            0                            +1                             +2                          +3                              +4  
a(1,1) = 3250433/53222400;    a(1,2) = 572741/5702400;      a(1,3) = -8701681/39916800;  a(1,4) = 4026311/13305600;    a(1,5) = -917039/3193344;   a(1,6) = 7370669/39916800;      a(1,7) = -1025779/13305600;  a(1,8) = 754331/39916800;     a(1,9) = -330157/159667200;
a(2,1) = -330157/159667200;   a(2,2) = 530113/6652800;      a(2,3) = 518887/19958400;    a(2,4) = -27631/623700;       a(2,5) = 44773/1064448;     a(2,6) = -531521/19958400;      a(2,7) = 109343/9979200;     a(2,8) = -1261/475200;        a(2,9) = 45911/159667200;
a(3,1) = 45911/159667200;     a(3,2) = -185839/39916800;    a(3,3) = 171137/1900800;     a(3,4) = 73643/39916800;      a(3,5) = -25775/3193344;    a(3,6) = 77597/13305600;        a(3,7) = -98911/39916800;    a(3,8) = 24173/39916800;      a(3,9) =  -3499/53222400;
a(4,1) = -3499/53222400;      a(4,2) = 4387/4989600;        a(4,3) = -35039/4989600;     a(4,4) = 90817/950400;        a(4,5) = -20561/3193344;    a(4,6) = 2117/9979200;          a(4,7) = 2059/6652800;       a(4,8) = -317/2851200;        a(4,9) = 317/22809600;
a(5,1) = 317/22809600;        a(5,2) = -2539/13305600;      a(5,3) = 55067/39916800;     a(5,4) = -326911/39916800;    a(5,5) = 14797/152064;      a(5,6) = -326911/39916800;      a(5,7) = 55067/39916800;     a(5,8) = -2539/13305600;      a(5,9) = a(4,9);
a(6,1) = a(5,1);              a(6,2) = -317/2851200;        a(6,3) = 2059/6652800;       a(6,4) = 2117/9979200;        a(6,5) = a(4,5);            a(6,6) = 90817/950400;          a(6,7) = -35039/4989600;     a(6,8) = 4387/4989600;        a(6,9) = a(3,9);
a(7,1) = a(4,1);              a(7,2) = 24173/39916800;      a(7,3) = -98911/39916800;    a(7,4) = 77597/13305600;      a(7,5) = a(3,5);            a(7,6) = 73643/39916800;        a(7,7) = 171137/1900800;     a(7,8) =-185839/39916800;     a(7,9) = a(2,9);
a(8,1) = a(3,1);              a(8,2) = -1261/475200;        a(8,3) = 109343/9979200;     a(8,4) = -531521/19958400;    a(8,5) = a(2,5);            a(8,6) = -27631/623700;         a(8,7) = 518887/19958400;    a(8,8) = 530113/6652800;      a(8,9) = a(1,9);
a(9,1) = a(2,1);              a(9,2) = 754331/39916800;     a(9,3) = -1025779/13305600;  a(9,4) = 7370669/39916800;    a(9,5) = a(1,5);            a(9,6) = 4026311/13305600;      a(9,7) = -8701681/39916800;  a(9,8) = 572741/5702400;      a(9,9) = 3250433/53222400;
a(10,1) =a(1,1);              a(10,2) = -11011481/19958400;a(10,3) = 6322573/2851200;    a(10,4) = -8660609/1663200;  a(10,5) = 25162927/3193344; a(10,6) = -159314453/19958400;  a(10,7) = 18071351/3326400;   a(10,8) = -24115843/9979200; a(10,9) = 103798439/159667200;
% Eighth-Order Summed-Adams Coefficients in Ordinate Form bjk
%-4                           -3                            -2                           -1                            0                            +1                             +2                          +3                              +4  
b(1,1) = 19087/89600;    b(1,2) = -427487/725760;     b(1,3) = 3498217/3628800;     b(1,4) = -500327/403200;    b(1,5) = 6467/5670;      b(1,6) =-2616161/3628800;      b(1,7) = 24019/80640;        b(1,8) = -263077/3628800;      b(1,9) = 8183/1036800;
b(2,1) = 8183/1036800;   b(2,2) = 57251/403200;       b(2,3) = -1106377/3628800;    b(2,4) = 218483/725760;     b(2,5) = -69/280;        b(2,6) = 530177/3628800;       b(2,7) = -210359/3628800;    b(2,8) = 5533/403200;          b(2,9) = -425/290304;
b(3,1) = -425/290304;    b(3,2) = 76453/3628800;      b(3,3) = 5143/57600;          b(3,4) = -660127/3628800;   b(3,5) = 661/5670;       b(3,6) = -4997/80640;          b(3,7) = 83927/3628800;      b(3,8) = -19109/3628800;       b(3,9) =  7/12800;
b(4,1) = 7/12800;        b(4,2) = -23173/3628800;     b(4,3) = 29579/725760;        b(4,4) = 2497/57600;        b(4,5) = -2563/22680;    b(4,6) = 172993/3628800;       b(4,7) = -6463/403200;       b(4,8) = 2497/725760;          b(4,9) = -2497/7257600;
b(5,1) = -2497/7257600;  b(5,2) = 1469/403200;        b(5,3) = -68119/3628800;      b(5,4) = 252769/3628800;    b(5,5) = 0;              b(5,6) = -252769/3628800;      b(5,7) = 68119/3628800;      b(5,8) = -1469/403200;         b(5,9) = -b(4,9);
b(6,1) = -b(5,1);        b(6,2) = -2497/725760;       b(6,3) = 6463/403200;         b(6,4) = -172993/3628800;   b(6,5) = -b(4,5);        b(6,6) = -2497/57600;          b(6,7) = -29579/725760;      b(6,8) = 23173/3628800;        b(6,9) = -b(3,9);
b(7,1) = -b(4,1);        b(7,2) = 19109/3628800;      b(7,3) = -83927/3628800;      b(7,4) = 4997/80640;        b(7,5) = -b(3,5);        b(7,6) = 660127/3628800;       b(7,7) = -5143/57600;        b(7,8) =-76453/3628800;        b(7,9) = -b(2,9);
b(8,1) = -b(3,1);        b(8,2) = -5533/403200;       b(8,3) = 210359/3628800;      b(8,4) = -530177/3628800;   b(8,5) = -b(2,5);        b(8,6) = -218483/725760;       b(8,7) = 1106377/3628800;    b(8,8) = -57251/403200;        b(8,9) = -b(1,9);
b(9,1) = -b(2,1);        b(9,2) = 263077/3628800;     b(9,3) = -24019/80640;        b(9,4) = 2616161/3628800;   b(9,5) = -b(1,5);        b(9,6) = 500327/403200;        b(9,7) = -3498217/3628800;   b(9,8) = 427487/725760;        b(9,9) = -19087/89600;
b(10,1) = 25713/89600;  b(10,2) = -9401029/3628800;  b(10,3) = 5393233/518400;     b(10,4) = -9839609/403200;  b(10,5) = 167287/4536;   b(10,6) = -135352319/3628800;  b(10,7) = 10219841/403200;   b(10,8) = -40987771/3628800;   b(10,9) = 3288521/1036800;
%Initialize Other Constants
t0 = tspan(1); 
tf = tspan(end);
MaxIter = 1000;
h2 = h^2;
%% Startup ...
%Find the time correspoinding to -4 before epoch and +4 after epoch
t_n_n4p4 = [t0-(4*h):h:t0-1,t0+h:h:t0+(h*4)];
%initialized time array
t = [t_n_n4p4(1:4),t0:h:tf];
%run ode113 as a single step integrator in order to provide an initial
%estimate for the first eight points (4 before epoch, and 4 after epoch)
[~,y_dy_n] = ode113(FirstOrderODE,[0,fliplr(t_n_n4p4(1:4))],[y0;dy0],options,varargin{:});
y_dy_n(1,:) = []; y_dy_n = flipud(y_dy_n);
[~,y_dy_p] = ode113(FirstOrderODE,[0,t_n_n4p4(5:8)],[y0;dy0],options,varargin{:});
y_dy_p(1,:) = []; y_dy = [y_dy_n; y_dy_p];
%Assemble the surrounding positions and velocities
y_p = [y_dy(1:4,1:end/2); y0' ;y_dy(5:8,1:end/2)];
dy_p = [y_dy(1:4,end/2+1:end); dy0' ;y_dy(5:8,end/2+1:end)];
%Evaluate the Nine ddy's from y and dy above
ddy_p = SecondOrderODE([t_n_n4p4(1:4),0,t_n_n4p4(5:8)],y_p',dy_p',varargin{:});
%Fix the epoch
ddy_p(:,5) = ddy0;
%Check ddy for convergence ...
ddy_1 = ddy_p; dy_1 = dy_p'; y_1 = y_p';
ddy_2 = NaN(size(ddy_1)); dy_2 = NaN(size(dy_1));  y_2 = NaN(size(y_1));
ddy_2(:,5) = ddy_p(:,5);
RelErr = Inf;
Iter = 0;

while any(RelErr > options.RelTol) && Iter <= MaxIter
    Iter = Iter + 1;
    %Eqn. (73) Calculate C1
    C1 = dy_1(:,5)./h - sum(bsxfun(@times,b(0+5,:),ddy_1),2) + ddy_1(:,5)./2;
    C1_prime = C1-ddy_1(:,5)./2;
    %Eqn. (75) Calculate sn
    sn = zeros(3,9); sn(:,0+5) = C1_prime;
    % For n < 0
    for n = -1:-1:-4; sn(:,n+5) = sn(:,n+1+5) - (ddy_1(:,n+1+5) + ddy_1(:,n+5))/2; end
    % For n > 0
    for n = 1:4; sn(:,n+5) = sn(:,n-1+5) + (ddy_1(:,n-1+5) + ddy_1(:,n+5))/2; end
    %Eqn. (85) Calculate Sn
    C2 = y_1(:,5)./(h2) - sum(bsxfun(@times,a(0+5,:),ddy_1),2) + C1;
    %Calculate S_n+1 (step 4, Eqn.86)
    Sn = zeros(3,9); Sn(:,0+5) = C2-C1;
    % For n < 0
    for n = -1:-1:-4; Sn(:,n+5) = Sn(:,n+1+5)-sn(:,n+1+5) + ddy_1(:,n+1+5)/2; end
    % For n > 0
    for n = 1:4; Sn(:,n+5) = Sn(:,n-1+5) + sn(:,n-1+5) + ddy_1(:,n-1+5)/2; end
    %Re-Calculate dy_p (74) and y_p (87) ... do not consider n == 0
    for n=1:9
        dy_2(:,n) = h.*(sn(:,n) +  sum(bsxfun(@times,b(n,:),ddy_1),2));
        y_2(:,n) = (h.^2).*(Sn(:,n) + sum(bsxfun(@times,a(n,:),ddy_1),2));
    end
    %Evaluate Updated ddy by calling y''(t)
    ddy_2 = SecondOrderODE([t_n_n4p4(1:4),0,t_n_n4p4(5:8)],y_2,dy_2,varargin{:});
    ddy_2(:,5) = ddy_1(:,5); dy_2(:,5) = dy_1(:,5); y_2(:,5) = y_1(:,5);
    RelErr = max(abs(ddy_2-ddy_1)); ddy_1 = ddy_2; dy_1 = dy_2; y_1 = y_2;
end
%Re-scale Sn and sn to allow for storage of all values over the full interval
Sn = [Sn,NaN(3,length(t)-9)]; sn = [sn,NaN(3,length(t)-9)];
%Re-scale ddy, dy, and y to allow for storage of all values over the full
%interval
ddy = [ddy_2,NaN(3,length(t)-9)]; dy = [dy_2,NaN(3,length(t)-9)]; y = [y_2,NaN(3,length(t)-9)];

%% Predictor/Corrector ...
for i=9:size(Sn,2)-1
    %Initialize a_5k and b_5k upfront
    addyn54 = sum(bsxfun(@times,a(5+5,:),ddy(:,i-8:i)),2);
    bddyn54 = sum(bsxfun(@times,b(5+5,:),ddy(:,i-8:i)),2);
    if i>9
        %Corrector
        %----------------------------------------------------------------------
        Iter = 0;
        %Compute a_4k*ddy and b_4k*ddy upfront
        addyn43 = sum(bsxfun(@times,a(4+5,1:8),ddy(:,i-8:i-1)),2);
        bddyn43 = sum(bsxfun(@times,b(4+5,1:8),ddy(:,i-8:i-1)),2);
        while Iter <= MaxIter
            Iter = Iter + 1;
            sn(:,i) = sn(:,i-1) + (ddy(:,i-1) + ddy(:,i))./2;
            yc = (h2).*(Sn(:,i) + addyn43 + a(4+5,9).*ddy(:,i));
            dyc = h.*(sn(:,i) + bddyn43 + b(4+5,9).*ddy(:,i));
            RelErr = [abs(yc-y(:,i)),abs(dyc-dy(:,i))];
            if all(RelErr(:) <= options.RelTol)
                break;
            end
            y(:,i) = yc;
            dy(:,i) = dyc;
            %Evaluate
            ddy(:,i) = SecondOrderODE(t(i),y(:,i),dy(:,i),varargin{:});
        end
    end
    %Predictor
    %----------------------------------------------------------------------
    %Eqn.86 Calculate S_n+1
    Sn(:,i+1) = Sn(:,i) + sn(:,i) + ddy(:,i)./2;
    y(:,i+1) = (h2).*(Sn(:,i+1) + addyn54);
    dy(:,i+1) = h.*(sn(:,i) + ddy(:,i)./2 + bddyn54);
    %Evaluate
    ddy(:,i+1) = SecondOrderODE(t(i+1),y(:,i+1),dy(:,i+1),varargin{:});
end

%Trim Arrays to appropriate length
t = t(5:end); y = y(:,5:end); dy = dy(:,5:end); ddy = ddy(:,5:end);

end