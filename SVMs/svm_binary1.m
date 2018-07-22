train_data = load('train.data');
train_label = load('train.label');
test_data = load('test.data');
test_label = load('test.label');

[vocabulary] = textread('vocabulary.txt', '%s'); % Removing stop words
[stopwords] = textread('stoplist.txt', '%s');
unique_docs = unique(train_data(:,1));
[vocab_len,~] = size(vocabulary);

c = 2.^(-1:15);
cv_ccr = zeros(length(c),1);

valid_rows = find(~ismember(vocabulary, stopwords)); % Updating train and test data: removing stop word rows
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
x = zeros(length(uniq_docs), length(valid_rows)); % Train data features
for i = 1:length(new_train)
    a = find(valid_rows == new_train(i,2));
    b = find(uniq_docs == new_train(i,1));
    x(b, a) = x(b, a) + new_train(i,3);
    
end
x = bsxfun(@rdivide,x,sum(x,2));

xt = zeros(length(uniq_test), length(valid_rows)); % Test data features
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
for i = 1:length(lab1_20_row)
    traind1_20 = find(train_data(train_rows,1) == lab1_20(i)); % Train data rows associated with documents labeled 1 or 20
    x1(i,:) = x(lab1_20_row(i),:);
end

lab1_20_rowt = find(test_label == 1 | test_label == 20); % Doc samples with label 1 or 20
lab1_20t = test_label(lab1_20_rowt,1);
x1t = zeros(length(lab1_20_rowt), length(valid_rows));
for i = 1:length(lab1_20_rowt)
    testd1_20 = find(test_data(test_rows,1) == lab1_20t(i)); % Test data rows associated with documents labeled 1 or 20
    x1t(i,:) = xt(lab1_20_rowt(i),:);
end

%% SVM CV on train
N = length(lab1_20);
cv_indices = crossvalind('Kfold', N, 5); 

for i = 1:length(c)
    for j = 1:5
        test_ind = find(cv_indices(:,1) == j);
        train_ind = find(~(cv_indices(:,1) == j));
        svmstruct = svmtrain(x1(train_ind,:), lab1_20(train_ind,:), 'autoscale', false,...
            'boxconstraint', c(i)*ones(length(train_ind),1), 'kernel_function', 'linear');
        group = svmclassify(svmstruct, x1(test_ind,:));
        cv_ccr(i,1) = cv_ccr(i,1) + numel(find(lab1_20(test_ind,1)==group))/ numel(group)
    end
    cv_ccr(i,1) = cv_ccr(i,1)/5;
end

[best_ccr, ind] = max(cv_ccr);
best_c = c(ind) % ANS (a.ii) Best C- maximum CCR

%% SVM on the test set
svmstruct = svmtrain(x1, lab1_20, 'autoscale', false,...
    'boxconstraint', best_c*ones(length(lab1_20),1), 'kernel_function', 'linear');
group = svmclassify(svmstruct, x1t);
cv_ccrt = numel(find(lab1_20t==group))/ numel(group)

%% Plots
plot(log(c), cv_ccr) % ANS (a.i)
xlabel('log c');
ylabel('CV-CCR');
title('log(c) Vs CV-CCR');
