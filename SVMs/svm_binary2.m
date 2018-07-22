%% Pre-processing
train_data = load('train.data');
train_label = load('train.label');
test_data = load('test.data');
test_label = load('test.label');

[vocabulary] = textread('vocabulary.txt', '%s');
[stopwords] = textread('stoplist.txt', '%s');
unique_docs = unique(train_data(:,1));
[vocab_len,~] = size(vocabulary);

c = 2.^(-5:15);
rbf_sigma = 2.^(-13:3);
c_r = zeros(length(c)*length(rbf_sigma),2);
temp = 1;
for i = 1:length(c)
    for j = 1:length(rbf_sigma)
        c_r(temp,1) = c(i);
        c_r(temp,2) = rbf_sigma(j);
        temp = temp + 1;
    end
end

cv_ccr = zeros(length(c_r),1);

valid_rows = find(~ismember(vocabulary, stopwords));
train_rows = find(ismember(train_data(:,2), valid_rows));
new_train = train_data(train_rows,:); 
test_rows = find(ismember(test_data(:,2), valid_rows));
new_test = test_data(test_rows,:); 

uniq_docs = unique(new_train(:,1));
uniq_docs_old = unique(train_data(:,1));
labelrem = setdiff(uniq_docs_old, uniq_docs);
train_label(labelrem, :) = [];
test_label(labelrem, :) = [];
uniq_test = unique(new_test(:,1));

%% Feature vectors
x = zeros(length(uniq_docs), length(valid_rows));
for i = 1:length(new_train)
    a = find(valid_rows == new_train(i,2));
    b = find(uniq_docs == new_train(i,1));
    x(b, a) = x(b, a) + new_train(i,3);
    
end
x = bsxfun(@rdivide,x,sum(x,2));

xt = zeros(length(uniq_test), length(valid_rows));
for i = 1:length(new_test)
    a = find(valid_rows == new_test(i,2));
    b = find(uniq_test == new_test(i,1));
    xt(b, a) = xt(b, a) + new_test(i,3);
end
xt = bsxfun(@rdivide,xt,sum(xt,2));

%% Class 1 and 20
lab1_20_row = find(train_label == 1 | train_label == 20); % Doc samples with label 1 or 20
lab1_20 = train_label(lab1_20_row,1);
x1 = zeros(length(lab1_20_row), length(valid_rows));
x1 = x(lab1_20_row,:);

lab1_20_rowt = find(test_label == 1 | test_label == 20); % Doc samples with label 1 or 20
lab1_20t = test_label(lab1_20_rowt,1);
x1t = zeros(length(lab1_20_rowt), length(valid_rows));
x1t = xt(lab1_20_rowt,:);

%% SVM CV on train
N = length(lab1_20);
cv_indices = crossvalind('Kfold', N, 5);
cv_ccr = zeros(length(c), length(rbf_sigma));

for i = 1:length(c)
    for k = 1:length(rbf_sigma)
        for j = 1:5
            test_ind = find(cv_indices(:,1) == j);
            train_ind = find(~(cv_indices(:,1) == j));
            svmstruct = svmtrain(x1(train_ind,:), lab1_20(train_ind,:), 'autoscale', false,...
                'boxconstraint', c(i)*ones(length(train_ind),1), 'kernel_function', 'rbf', 'rbf_sigma', rbf_sigma(k));
            group = svmclassify(svmstruct, x1(test_ind,:));
            cv_ccr(i,k) = cv_ccr(i,k) + numel(find(lab1_20(test_ind,1)==group))/ numel(group);
        end
        cv_ccr(i,k) = cv_ccr(i,k)/5;
    end
end

best_ccr = max(max(cv_ccr));
[best_c, best_r]  = find(cv_ccr == best_ccr); 

%% SVM on the test set

svmstruct = svmtrain(x1, lab1_20, 'autoscale', false,...
    'boxconstraint', best_c*ones(length(lab1_20),1), 'kernel_function', 'rbf', 'rbf_sigma', best_r);
group = svmclassify(svmstruct, x1t);
cv_ccrt1 = numel(find(lab1_20t==group))/ numel(group);

%% Plots

contour(log(c), log(rbf_sigma), cv_ccr);
xlabel('Boxconstraint');
ylabel('Rbf_sigma');
title('Boxconstraint Vs Rbf_sigma');
