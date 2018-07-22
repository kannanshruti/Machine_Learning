% Description: OLS and Ridge regression (Implementation)
quad_data = load('quad_data.mat');
xtrain = quad_data.xtrain;
ytrain = quad_data.ytrain;
xtest = quad_data.xtest;
ytest = quad_data.ytest;

[poly_ols, poly_olst] = deal(zeros(21, 14), zeros(201, 14));
[mse_ols, mse_olst] = deal(zeros(14,1), zeros(14,1));
k = -25:5;
[poly_ridge, poly_ridget] = deal(zeros(21, 31), zeros(201, 31));
[mse_ridge, mse_ridget] = deal(zeros(length(k),1), zeros(length(k),1));
degree = [2,6,10,14];

for i = 1:14
    xtrain(:,i) = xtrain(:,1).^i;
    xtest(:,i) = xtest(:,1).^i;
end

%% OLS Train and Test
for i = 1:14
    b0 = ridge(ytrain, xtrain(:,1:i), 0, 0);
    poly_ols(:,i) = b0(1) + (xtrain(:,1:i) * b0(2:end));
    mse_ols(i,1) = immse(poly_ols(:,i) , ytrain);
    
    poly_olst(:,i) = b0(1) + (xtest(:,1:i) * b0(2:end));
    mse_olst(i,1) = immse(poly_olst(:,i) , ytest);
end

%% Ridge Train and Test
for i = 1:length(k)
    b0 = ridge(ytrain, xtrain(:,1:10), exp(k(i)),0);
    poly_ridge(:,i) = b0(1) + (xtrain(:,1:10) * b0(2:end));
    mse_ridge(i,1) = immse(poly_ridge(:,i) , ytrain);
    
    poly_ridget(:,i) = b0(1) + (xtest(:,1:10) * b0(2:end));
    mse_ridget(i,1) = immse(poly_ridget(:,i) , ytest);
end

[~, indtr] = min(mse_ridge);
[~, indt] = min(mse_ridget);

%% Ridge coefficients of degree 4
for i = 1:length(k)
    b4(i,:) = ridge(ytrain, xtrain(:,1:4), exp(k(i)),0);    
end

%% Plots
% 7.3 a i
figure;
scatter(xtrain(:,1), ytrain, 'filled'); hold on;
for i = 1:length(degree)
    plot(xtrain(:,1), poly_ols(:,degree(i))); hold on;
end
legend('Data', '2', '6', '10', '14');
title('Polynomials of degrees 2,6,10,14 fitted across data points (OLS) (7.3.a.i)');
xlabel('X- Data');
ylabel('Y- Data');

% 7.3 a ii
figure;
plot(1:14, mse_ols); hold on;
plot(1:14, mse_olst);
legend('MSE Train', 'MSE Test');
title('Mean Sq Error Vs Degree of polynomial being fit (OLS) (7.3.a.ii)');
xlabel('Degree');
ylabel('Mean Squared Error');

% 7.3 b i
figure;
plot(k,mse_ridge); hold on;
plot(k, mse_ridget);
legend('MSE Train', 'MSE Test');
title('Mean Sq Error Vs log(k) (7.3.b.i)');
xlabel('log(k)');
ylabel('Mean Squared Error');

% 7.3 b ii
figure;
scatter(xtest(:,1), ytest, 'filled'); hold on;
plot(xtest(:,1), poly_ridget(:,indt), 'LineWidth',2); hold on;
plot(xtest(:,1), poly_olst(:,10), 'LineWidth',2); hold on;
legend('Data', 'OLS', 'Ridge');
title('Polynomials of degree 10- Ridge (7.3.b.ii)');
xlabel('Data');
ylabel('Labels');

% 7.3 c
figure;
plot(k, b4, 'LineWidth',2);
legend('w0', 'w1', 'w2', 'w3', 'w4');
title('Ridge Coefficients (degree=4) vs ln lambda (7.3.b.iii)');
xlabel('ln lambda');
ylabel('Ridge Coefficients');
