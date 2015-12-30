function y = classify(Model1, X)

cellsize1 = 4; %For HOG feature extraction
cellsize2 = 8; %For HOG feature extraction - only for SVM

%%%Feature extraction
for i=1:1:size(X,1)
    im =im2single(reshape(X(i,:),32,32,3));
    hog_feat_test1(i,:)=hog_extract_feature(im, cellsize1)';
    hog_feat_test2(i,:)=hog_extract_feature(im, cellsize2)';
end

%%%Probabilities of the predictions given the class
[~, prob_knn] = classify_knn(double(Model1.X_knn)/1e4, hog_feat_test1', Model1.labels_knn, Model1.num_k);
[~, prob_svm] = classify_svm(double(Model1.X_svm)/1e4, double(Model1.alpha)/1e4, double(hog_feat_test2), Model1.labels_svm, Model1.kernel_sigma, Model1.epsilon);

prob_total = prob_svm+prob_knn;
[~,y] = max(prob_total,[],2);
y = uint8(y - 1);

end


function [y_ans, prob_knn] = classify_knn(features_train, features_test, batch_train_labels, num_k)

%%%Implement a kNN classifier based on Euclidean distance

dist_to_cluster = bsxfun(@plus, dot(features_train, features_train, 1)'...
    , dot(features_test, features_test, 1)) - 2*(features_train'*features_test);

dist_to_cluster_orig = dist_to_cluster;
test_labels = unique(batch_train_labels);

for i=1:1:num_k
    [~,y_index(i,:)] = min(dist_to_cluster,[],1);
    y_labels(i,:) = batch_train_labels(y_index(i,:),1);


    mat_subtract = dist_to_cluster*0;
    for j=1:1:size(mat_subtract,2)
        mat_subtract(y_index(i,j),j) = 1e6;
    end
    dist_to_cluster = dist_to_cluster + mat_subtract;
end

for i=1:1:size(y_labels,2)
    y_count = zeros(length(unique(batch_train_labels)),1);
    y_dist = zeros(1,length(unique(batch_train_labels)));
    unique_labels = unique(y_labels(:,i));
    for j=1:1:length(unique_labels)
        y_count(unique_labels(j)+1,1) = length(find(y_labels(:,i)==unique_labels(j)));
    end

    for j=1:1:length(y_count)
        if(y_count(j)>0)
            dist_index = find(y_labels(:,i)==(j-1));
            y_dist(1,j) = mean(dist_to_cluster_orig(y_index(dist_index,i),i))/length(dist_index);
        end
    end
%     [~,y_dist_min_index] = min(y_dist);
%     y_ans(1,i) = unique_labels(y_dist_min_index);

    y_prob = [];
    y_temp = y_dist(unique_labels+1);
    for j=1:1:length(y_temp)
      if (sum(y_temp>0) || length(y_temp)>1)
        y_prob(1,j) = sum(y_temp([1:length(y_temp)]~=j))/((length(y_temp)-1)*sum(y_temp));
      else
        y_prob(1,j)=1;
      end

        if(isinf(y_prob(1,j))==1)
            y_prob(1,j) = 1;
        end
    end

    y_dist(unique_labels+1) = y_prob;
    prob_knn(i,:) = y_dist;

end
[~,y_count_max_index] = max(prob_knn,[],2);
y_ans = test_labels(y_count_max_index);
end

function [svm_index, prob_svm] = classify_svm(batch_train_data_all, alpha_train, batch_test_data_all, batch_train_labels, kernel_sigma, epsilon)

%%%Train SVM weights and classify
train_labels = unique(batch_train_labels);
y_ans = zeros(size(batch_test_data_all,1),length(train_labels));
g = zeros(size(batch_test_data_all,1),length(train_labels));
K_test = svm_kernel(batch_test_data_all', batch_train_data_all', kernel_sigma);


for i=1:1:length(train_labels)
    svm_labels = double(batch_train_labels == train_labels(i));
    index_zeros = find(svm_labels==0);
    svm_labels(index_zeros) = -1;

    %%%Get support vector indices
    sv_index = find(alpha_train(:,i)>epsilon);
    [y_ans_temp,g_temp] = svm_classify(alpha_train(sv_index,i), K_test(sv_index,:), svm_labels(sv_index,1));

    y_ans(:,i) = y_ans_temp;
    g(:,i) = g_temp;


%     svm_labels_test = double(batch_test_labels == train_labels(i));
%     index_zeros = find(svm_labels_test==0);
%     svm_labels_test(index_zeros) = -1;
%
end

prob_svm = g;
prob_svm(find(prob_svm<=0))=0;
for i=1:size(prob_svm,1)
    prob_svm(i,:)=prob_svm(i,:)/sum(prob_svm(i,:));
end

[~,svm_index] = max(g,[],2);
svm_index = svm_index - 1;

end

function [K] = svm_kernel(x,l,sigma )

%%%RBF Kernel
svm_dist = bsxfun(@plus, dot(l, l, 1)', dot(x, x, 1)) - 2*(l'*x);
K = exp(-svm_dist/(2*sigma*sigma));

end

function [y_ans,g] = svm_classify(alpha, K_test, y_train)

%%%Write a linear/non-linear SVM classifier - degree 2
%%%RBF
g = sum(repmat(alpha.*y_train,1,size(K_test,2)).*K_test,1)';
y_ans = -1*ones(size(K_test,2),1);
y_ans(g>0) = 1;
end


function hog_features = hog_extract_feature(image, cellsize)
hog = vl_hog(image, cellsize);
hog_features = hog(:);
end
