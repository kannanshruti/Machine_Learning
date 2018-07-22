m = 39; % no of classes
[n, no_feat] = size(feat); % no_samples, no_features : 878049 x 41
w = zeros(m,no_feat); % each theta= 1 x 41
[grad_y, grad_f] = deal(zeros(m,no_feat), zeros(m,no_feat));
[e_wx, e_wx1, no_y, e_wx2, sum_x] = deal(zeros(n,1), zeros(m,1), zeros(m,1), zeros(m,no_feat), zeros(m,no_feat));
[A, B, no_iter, lambda] = deal(0,0,1000, 1000);
[f, ccr2, logloss, theta] = deal(zeros(no_iter, 1), zeros(no_iter, 1), zeros(no_iter, 1), zeros(m, no_feat, 1000));
[ind, val] = deal(zeros(n, 1), zeros(n, 1));
step_size = 10^-5; % Step size (eta)

num = ceil(0.6*n); % 60% samples
row_ind_train = 1:num; % Storing 60% train data 
row_ind_test = num+1 : n; % Storing remaining 40% test data
n_test = length(row_ind_test); % No of test samples

y_train = Category(row_ind_train, 1);
y_test = histval(row_ind_test,4); % Ground truth

for y = 1:m
    no_y(y,1) = sum(histval(row_ind_train,4) == y); % No of labels per class
    label = find(histval(row_ind_train,4) == y); % Rows corresponding to each label
    sum_x(y,:) = sum(feat(label,:)); % Sum of feat vec values corresponding to each class
end

for t = 1:no_iter
    e_wx2 = w; % w
    e_wx2 = e_wx2 * feat(row_ind_train,:)'; % w*x
    e_wx2 = exp(e_wx2); % e^(w*x) 
    NLL = sum(log(sum(e_wx2)),2) - sum(sum(w.*sum_x,2));
    grad_y = bsxfun(@rdivide,e_wx2,sum(e_wx2)) * feat(row_ind_train,:) - sum_x;
    f(t,1) = NLL + (lambda/2) * (norm(w)^2); % f(theta)
    grad_f = grad_y + lambda .* w; % Gradient of f(theta)
    
    e_wx3 = w;
    e_wx3 = e_wx3 * feat(row_ind_test,:)'; 
    e_wx3 = exp(e_wx3); % e^(w*x) 39 x 878049
    py_x = bsxfun(@rdivide,e_wx3,sum(e_wx3));
    
    p= zeros(1, n_test);
    py_x = log(py_x);
    for i = 1:n_test;
        p(1,i) = p(1,i) + py_x(y_test(i),i);
    end
    logloss(t,1) = (-1/n_test) * sum(p,2); % Logloss
    
    w = w - (step_size .* grad_f); % Updating the weights
    
    w_x  = feat(row_ind_test,:)*w';
    [~,y_predict] = max(w_x, [], 2);
    ccr2(t,1) = numel(find(histval(row_ind_test,4) == y_predict))/length(y_predict); % CCR
end

%% Plots
figure;
plot(1:no_iter,f); % ANS
title('Objective Function (f(theta)) Vs Number of Iterations (t)');
xlabel('t');
ylabel('f(theta)');

figure;
plot(1:no_iter,logloss); % ANS
title('Log loss Vs Number of Iterations (t)');
xlabel('t');
ylabel('logloss');

figure;
plot(1:no_iter,ccr2); % ANS
title('Correct Classification Rate (CCR) Vs Number of Iterations (t)');
xlabel('t');
ylabel('CCR');

