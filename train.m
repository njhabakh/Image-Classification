function Model = train()

batch_train.data1 = [];
batch_train.data2 = [];
batch_train.labels = [];

for n=1:1:5
    %%Load a single training batch
    batch_load = load(['small_data_batch_',num2str(n),'.mat']);
    for i=1:1:size(batch_load.data,1)
        %Converting the image to single
        im =im2single(reshape(batch_load.data(i,:),32,32,3));
        %Extracting hog features of different cell sizes
        hog_feat_train1(i,:)=hog_extract_feature(im,8)';
        hog_feat_train2(i,:)=hog_extract_feature(im,4)';
    end
    
    batch_train.data1 = [batch_train.data1; hog_feat_train1];
    batch_train.data2 = [batch_train.data2; hog_feat_train2];
    batch_train.labels = [batch_train.labels; batch_load.labels];
end

batch_train.data1 = double(batch_train.data1);

%%%Train SVM parameters
train_labels = unique(batch_train.labels);
K_train = svm_kernel(batch_train.data1', batch_train.data1', 1.25);

for i=1:1:length(train_labels)
    svm_labels = double(batch_train.labels == train_labels(i));
    index_zeros = find(svm_labels==0);
    svm_labels(index_zeros) = -1;
    
    %%%Get training kernel and alphas
    alpha_train(:,i) = svm_weights(K_train, svm_labels);
end  


Model.X_svm = uint16(batch_train.data1*1e4);
Model.X_knn = uint16(batch_train.data2(1:4000,:)'*1e4);
Model.alpha = uint16(alpha_train*1e4);
Model.labels_svm = batch_train.labels;
Model.labels_knn = batch_train.labels(1:4000);
Model.kernel_sigma = 1.25;
Model.epsilon = 1e-6;
Model.labels = unique(batch_train.labels);

end

function hog_features = hog_extract_feature(image,cellSize)
hog = vl_hog(image, cellSize);
hog_features = hog(:);
end

function K = svm_kernel(x,l,sigma )

%%%RBF Kernel
svm_dist = bsxfun(@plus, dot(l, l, 1)', dot(x, x, 1)) - 2*(l'*x);
K = exp(-svm_dist/(2*sigma*sigma));

end

function alpha = svm_weights(K, y)

%%%Calculate linear/non-linear SVM weights 
H = diag(y)*K*diag(y);
f = -1*ones(length(y),1);

lower_alpha = zeros(size(y,1),1);
upper_alpha = 5000*ones(size(y,1),1);

[alpha,~,~,~] = quadprog(H,f,[],[],y',0,lower_alpha,upper_alpha);

end