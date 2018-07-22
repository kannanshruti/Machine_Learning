%% Pre-processing
train_data = load('train.data');
train_label = load('train.label');
test_data = load('test.data');
test_label = load('test.label');

[vocabulary] = textread('vocabulary.txt', '%s');
[stopwords] = textread('stoplist.txt', '%s');
unique_docs = unique(train_data(:,1));
[vocab_len,~] = size(vocabulary);

valid_rows = find(~ismember(vocabulary, stopwords));
train_rows = find(ismember(train_data(:,2), valid_rows));
new_train = train_data(train_rows,:); % TRAIN DATA WITHOUT STOP WORDS
test_rows = find(ismember(test_data(:,2), valid_rows));
new_test = test_data(test_rows,:); % TEST DATA WITHOUT STOP WORDS

uniq_docs = unique(new_train(:,1)); % Unique documents in the updated train data
uniq_docs_old = unique(train_data(:,1));
labelrem = setdiff(uniq_docs_old, uniq_docs); 
train_label(labelrem, :) = [];
test_label(labelrem, :) = [];
uniq_test = unique(new_test(:,1));% Unique documents in the updated test data

%% Feature vectors
x = zeros(length(uniq_docs), length(valid_rows)); % Train data
for i = 1:length(new_train)
    a = find(valid_rows == new_train(i,2));
    b = find(uniq_docs == new_train(i,1));
    x(b, a) = x(b, a) + new_train(i,3);
    
end
x = bsxfun(@rdivide,x,sum(x,2));
x = sparse(x);

xt = zeros(length(uniq_test), length(valid_rows)); % Test data
for i = 1:length(new_test)
    a = find(valid_rows == new_test(i,2));
    b = find(uniq_test == new_test(i,1));
    xt(b, a) = xt(b, a) + new_test(i,3);
end
xt = bsxfun(@rdivide,xt,sum(xt,2));
xt = sparse(xt);

%% Pairs of Classes
temp = 1;
for i = 1:20
    for j = (i+1):20
        pairs(temp,1) = i;
        pairs(temp,2) = j;
        temp = temp + 1;
    end
end
cv_ccr = zeros(length(test_label), length(pairs(:,1))); 

%% Linear SVM
for j = 1:length(pairs(:,1))
    lab1_20_row = find(train_label == pairs(j,1) | train_label == pairs(j,2)); % Doc samples with label as per jth class pair
    lab1_20 = train_label(lab1_20_row,1);
    x1 = zeros(length(lab1_20_row), length(valid_rows));
    x1 = x(lab1_20_row,:); % Features corresponding to jth class pair
    tic;
    svmstruct = svmtrain(x1, lab1_20, 'autoscale', false, 'kernel_function', 'rbf'); % Training of SVM with jth class pair
    a(j,1) = toc;
    tic;
    group = svmclassify(svmstruct, xt); % Prediction of test labels as per jth class pair
    a(j,2) = toc;
    cv_ccr(:,j) = group; % Storing predictions 
end
y_predict = mode(cv_ccr, 2);
cm = confusionmat(test_label, y_predict);
time_train = sum(a(:,1));
time_test = sum(a(:,2));
overall_ccr = trace(cm)/sum(sum(cm));