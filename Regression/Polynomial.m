% Description: Polynomial vs Ridge regression (Implementation)
pros_data = load('prostateStnd.mat');
xtrain = pros_data.Xtrain;
ytrain = pros_data.ytrain;
xtest = pros_data.Xtest;
ytest = pros_data.ytest;
[n,d] = size(xtrain);
[a,c,] = deal(zeros(8,1), zeros(8,1));
lambda = -5:10;
[poly_ridge, poly_ridget] = deal(zeros(67, 8), zeros(30, 8));
[mse_ridge, mse_ridget] = deal(zeros(16,1), zeros(16,1));
w = zeros(16,8);
w_new = zeros(16,8);

xtrain1 = xtrain - repmat(mean(xtrain),67,1);
ytrain1 = ytrain - repmat(mean(ytrain),67,1);
xtest1 = xtest - repmat(mean(xtest),30,1);
ytest1 = ytest - repmat(mean(ytest),30,1); 

%% 7.4.a,b
t=1;
for lam = 1:length(lambda)
    for t = 1:100
       for k = 1:d
          a(k) = 2 * (xtrain1(:,k)' * xtrain1(:,k)); % 1x67 * 67x1 = 1x1
          c(k) = 2 * (xtrain1(:,k)' * (ytrain1 - xtrain1*w(lam,:)' + w(lam,k).*xtrain1(:,k))); % 1x67 * (67x1 - 67x8*8x1 + 1x1*67x1)
          w(lam,k) = wthresh((c(k)/a(k)),'s',(exp(lambda(lam))/a(k)));
       end
    end
end

plot(lambda, w)
legend(pros_data.names(1:8));
title('Features Vs log(lambda)');
xlabel('log(lambda)');
ylabel('Features');


for i = 1:length(lambda)
    poly_ridge(:,i) = xtrain1 * w(i,:)';
    mse_ridge(i,1) = mean((poly_ridge(:,i) - ytrain1) .^ 2);
    
    poly_ridget(:,i) = xtest1 * w(i,:)';
    mse_ridget(i,1) = mean((poly_ridget(:,i) - ytest1) .^ 2);
end

figure;
plot(lambda,mse_ridge); hold on;
plot(lambda, mse_ridget);
legend('MSE Train', 'MSE Test');
title('Mean Sq Error Vs log(lambda)');
xlabel('log(lambda)');
ylabel('Mean Squared Error');

%% 7.4.c,d

k = -5:10;
[poly_ridge_4, poly_ridget_4] = deal(zeros(67, length(k)), zeros(30, length(k)));
[mse_ridge_4, mse_ridget_4] = deal(zeros(length(k),1), zeros(length(k),1));
for i = 1:length(k)
    b0 = ridge(ytrain, xtrain, exp(k(i)));
    wr(i,:) = b0;
    poly_ridge_4(:,i) = xtrain * b0;
    mse_ridge_4(i,1) = mean((poly_ridge_4(:,i) - ytrain) .^ 2);
    
    poly_ridget_4(:,i) = xtest * b0;
    mse_ridget_4(i,1) = mean((poly_ridget_4(:,i) - ytest) .^ 2);
end
plot(lambda, wr)
legend(pros_data.names(1:8));
title('Features Vs log(lambda)');
xlabel('log(lambda)');
ylabel('Features');

figure;
plot(k,mse_ridge_4); hold on;
plot(k, mse_ridget_4);
legend('MSE Train', 'MSE Test');
title('Mean Sq Error Vs log(lambda)');
xlabel('log(lambda)');
ylabel('Mean Squared Error');
