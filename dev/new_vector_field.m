%%
clear; clc;

[x,y] = meshgrid(-2:0.1:2, -1:0.1:5);

u = zeros(61,41);
v = zeros(61,41);

%% Inside field
% X is controlled with a Gaussian
x_in_sigma = 0.1;
x_in_mu = 0;
% Y is controlled with a sigmoid
y_in_sigma = 0.1;
y_in_mu = -1;

%% Outside field
center = [0,1];
% X is controlled with a sigmoid
x_out_sigma = [0.1,0.1];
x_out_mu = [-0.5,0.5];
% Y is controlled with a sigmoid
y_out_sigma = [0.1,0.1];
y_out_mu = [-1,0.2];

%% Make field
for i=[1:61]
    for j=[1:41]
        % Inside
        u(i,j) = u(i,j) + 0;
        v(i,j) = v(i,j) + 1/x_in_sigma/sqrt(2*pi) * exp(-1/2 * ((x(i,j)-(center(1)+x_in_mu))/x_in_sigma)^2) * 1 / (1+exp(-(y(i,j)-(center(2)+y_in_mu))/y_in_sigma));
        % Outside
        d = center - [x(i,j), y(i,j)];
        unit = d / norm(d);
        
        x_bounds = center(1) + x_out_mu;
        y_bounds = center(2) + y_out_mu;
        
        x_val = [0,0];
        x_val(1) = 1 / (1 + exp( (-x(i,j) + x_bounds(1)) / x_out_sigma(1)) );
        x_val(2) = 1 / (1 + exp( -(-x(i,j) + x_bounds(2)) / x_out_sigma(2)) );
        
        y_val = [0,0];
        y_val(1) = 1 / (1 + exp( (-y(i,j) + y_bounds(1)) / y_out_sigma(1)) );
        y_val(2) = 1 / (1 + exp( -(-y(i,j) + y_bounds(2)) / y_out_sigma(2)) );
        
        acc = unit * 5 / (norm(d)+1)^2 * x_val(1) * x_val(2) * y_val(1) * y_val(2);
        u(i,j) = u(i,j) + acc(1);
        v(i,j) = v(i,j) + acc(2);
    end
end

%% Plot
t = [-2:0.1:2];
h = ones(1,length(t))*4;

figure
quiver(x,y,u,v)
hold on
plot(t,h)
hold off










