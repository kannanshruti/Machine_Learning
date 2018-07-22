%% Pre-processing
train_data = load('train.data');
train_label = load('train.label');
test_data = load('test.data');
test_label = load('test.label');

[vocabulary] = textread('vocabulary.txt', '%s'); % Removing stop words
[stopwords] = textread('stoplist.txt', '%s');
unique_docs = unique(train_data(:,1));
[vocab_len,~] = size(vocabulary);

c = 2.^(-5:15);
cv_ccr = zeros(length(c),1);
p = zeros(length(c),1);
r = zeros(length(c),1);
f = zeros(length(c),1);
cv_ccr1 = zeros(length(c),5);
p1 = zeros(length(c),5);
r1 = zeros(length(c),5);
f1 = zeros(length(c),5);

valid_rows = find(~ismember(vocabulary, stopwords)); % Updating Train and test data
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
x = zeros(length(uniq_docs), length(valid_rows)); % Train features
for i = 1:length(new_train)
    a = find(valid_rows == new_train(i,2));
    b = find(uniq_docs == new_train(i,1));
    x(b, a) = x(b, a) + new_train(i,3);
    
end
x = bsxfun(@rdivide,x,sum(x,2));
x = sparse(x);
xt = zeros(length(uniq_test), length(valid_rows)); % Test features
for i = 1:length(new_test)
    a = find(valid_rows == new_test(i,2));
    b = find(uniq_test == new_test(i,1));
    xt(b, a) = xt(b, a) + new_test(i,3);
end
xt = bsxfun(@rdivide,xt,sum(xt,2));
xt = sparse(xt);

%% Class 17 Vs Others
lab17_row = find(train_label == 17); % Doc samples with label 17
lab17 = train_label(lab17_row,1);
lab1_row = find(~(train_label == 17)); % Doc samples with label other than 17
train_label(lab1_row) = 1;
lab17_rowt = find(test_label == 17); % Doc samples with label 17
lab17t = test_label(lab17_rowt,1);
lab1_rowt = find(~(test_label == 17)); % Doc samples with label other than 17
test_label(lab1_rowt) = 1;

%% SVM CV on train
N = length(train_label);
cv_indices = crossvalind('Kfold', N, 5);
for i = 1:length(c)
    for j = 1:5
        test_ind = find(cv_indices(:,1) == j);
        train_ind = find(~(cv_indices(:,1) == j));
        svmstruct = svmtrain(x(train_ind,:), train_label(train_ind,:), 'autoscale', false,...
            'boxconstraint', c(i), 'kernel_function', 'linear', 'kernelcachelimit', 100000);
        group = svmclassify(svmstruct, x(test_ind,:));        
        cm1 = confusionmat(train_label(test_ind,1), group);
        cv_ccr1(i,j) = trace(cm1) / sum(sum(cm1));
        p1(i,j) = cm1(2,2)/sum(cm1(2,:),2);
        r1(i,j) = cm1(2,2)/sum(cm1(:,2),1);
        f1(i,j) = 2*(p1(i,j) * r1(i,j))/(p1(i,j) + r1(i,j));
    end
end

cv_ccr = mean(cv_ccr1,2);
p = mean(p1,2);
r = mean(r1,2);
f = mean(f1,2);

[best_ccr, ind] = max(cv_ccr);
best_c = c(ind); % Best C- maximum CCR
[best_r, ind] = max(r);
best_cr = c(ind); % Best C- maximum Recall
[best_f, ind] = max(f);
best_cf = c(ind); % Best C- maximum F-score

%% SVM on the test set
svmstruct = svmtrain(x, train_label, 'autoscale', false,... 
    'boxconstraint', best_c, 'kernel_function', 'linear', 'kernelcachelimit', 100000);
group = svmclassify(svmstruct, xt); % C with best CCR
cv_ccrt = numel(find(test_label==group))/ numel(group);
cm = confusionmat(test_label, group);

svmstruct = svmtrain(x, train_label, 'autoscale', false,...
    'boxconstraint', best_cr, 'kernel_function', 'linear', 'kernelcachelimit', 100000);
group = svmclassify(svmstruct, xt); % C with best Recall
cm2 = confusionmat(test_label, group);
cv_ccrt2 = trace(cm2) / sum(sum(cm2));

svmstruct = svmtrain(x, train_label, 'autoscale', false,...
    'boxconstraint', best_cf, 'kernel_function', 'linear', 'kernelcachelimit', 100000);
group = svmclassify(svmstruct, xt);  % C with best F-score
cm3 = confusionmat(test_label, group);
cv_ccrt3 = trace(cm3) / sum(sum(cm3));

%% Plots
figure;
plot(log(c), cv_ccr) % log(c) Vs CV-CCR
xlabel('log c');
ylabel('CV-CCR');
title('log(c) Vs CV-CCR');

figure;
plot(log(c), p, log(c), r, log(c), f); % log(c) vs P, R, F
legend( 'p', 'r', 'f');
xlabel('log c');
ylabel('CV Parameters');
title('log(c) Vs Precision(p), Recall(r), F-score(f)');