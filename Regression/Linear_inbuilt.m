% Description: Linear regression with different weight functions
linear_data = load('linear_data.mat');
xdata = linear_data.xData;
ydata = linear_data.yData;

%% Linear regression
mean_x = mean(xdata);
mean_y = mean(ydata);
[n,~] = size(xdata);
xdata(:,2) = ones(n,1);

b(:,1) = (xdata' * xdata) \ (xdata' * ydata);
h(:,1) = xdata(:,1) * b(1,1) + b(2,1);
mse_robust(1,1) = mean((ydata - h(:,1)) .^ 2);
mae_robust(1,1) = mean(abs(ydata - h(:,1)));

%% 
b(:,2) = robustfit(xdata(:,1), ydata, 'cauchy');
h(:,2) = xdata(:,1) * b(2,2) + b(1,2);
mse_robust(1,2) = mean((ydata - h(:,2)) .^ 2);
mae_robust(1,2) = mean(abs(ydata - h(:,2)));


b(:,3) = robustfit(xdata(:,1), ydata, 'fair');
h(:,3) = xdata(:,1) * b(2,3) + b(1,3);
mse_robust(1,3) = mean((ydata - h(:,3)) .^ 2);
mae_robust(1,3) = mean(abs(ydata - h(:,3)));

b(:,4) = robustfit(xdata(:,1), ydata, 'huber');
h(:,4) = xdata(:,1) * b(2,4) + b(1,4);
mse_robust(1,4) = mean((ydata - h(:,4)) .^ 2);
mae_robust(1,4) = mean(abs(ydata - h(:,4)));

b(:,5) = robustfit(xdata(:,1), ydata, 'talwar');
h(:,5) = xdata(:,1) * b(2,5) + b(1,5);
mse_robust(1,5) = mean((ydata - h(:,5)) .^ 2);
mae_robust(1,5) = mean(abs(ydata - h(:,5)));

w_huber = b(2,4);
b_huber = b(1,4);

%% Plot

scatter(xdata(:,1), ydata, 'filled'); hold on;
plot(xdata(:,1), h, 'LineWidth',2)
legend('Data', 'OLS', 'Cauchy','Fair','Huber','Talwar')
title('Comparison of OLS, Cauchy, Fair, Huber and Talwar methods');
xlabel('x-data');
ylabel('y-data');

