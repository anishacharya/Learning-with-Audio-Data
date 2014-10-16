%% while not Using PCA %%
y=dataset_FDR_sorted(:,end);
X=dataset_FDR_sorted(:,1:N);% here N is the number of top feature u want to include

%% while using PCA %% 
% x=dataset(:,1:end-1);
% y=dataset(:,end);
% B=zscore(x);
% [COEFF SCORE LATENT]=princomp(B);
% X1=B*COEFF;
% E=cumsum(var(SCORE))/sum(var(SCORE));
% X=X1(:,1:24);

%% Monte Carlo Loop %%

k=1; 
for i=1:1:500 
% [x y] = shuffleData(x,y);
% [Xtr Xte Ytr Yte] = splitData(x,y, .60);  
%%when using PCA
[X y] = shuffleData(X,y);
[Xtr Xte Ytr Yte] = splitData(X,y, .60);

%% lda %% 

%% linear 
%Fits a multivariate normal density to each group,with a pooled estimate of covariance

lda = classify(Xte,Xtr,Ytr);
Err_lda(k,1)=sum(abs(lda-Yte)>0)/length(Yte);

%% qda %%
% 'quadratic' � Fits multivariate normal densities with covariance estimates stratified by group.

qda = classify(Xte,Xtr,Ytr,'quadratic');
Err_qda(k,1)=sum(abs(qda-Yte)>0)/length(Yte);

%% naive bayes %%
%Similar to 'linear', but with a diagonal covariance matrix estimate (naive Bayes classifiers)

naive = classify(Xte,Xtr,Ytr,'diaglinear');
Err_naive(k,1)=sum(abs(naive-Yte)>0)/length(Yte);

%% naive_quad %%
%Similar to 'quadratic', but with a diagonal covariance matrix estimate (naive Bayes classifiers)

naive_quad = classify(Xte,Xtr,Ytr,'diagquadratic');
Err_naive_quad(k,1)=sum(abs(naive_quad-Yte)>0)/length(Yte);

%% moholanbis distance %%
%Uses Mahalanobis distances with stratified covariance estimates.

moholan = classify(Xte,Xtr,Ytr,'mahalanobis');
Err_moholanbis(k,1)=sum(abs(moholan-Yte)>0)/length(Yte);









%% knn %%
%% euclidian 
%Euclidean distance (default)
% for j=1:1:20    
% knn_euclid = knnclassify(Xte,Xtr,Ytr,j);
% Err_knn_euclidian(j,k)=sum(abs(knn_euclid-Yte)>0)/length(Yte);
% end
%Err_knn_euclidian_best(k,1)=min(Err_knn_euclidian(:,1));
% %% cityblock
% % Sum of absolute differences
% for j=1:1:20    
% knn_abs = knnclassify(Xte,Xtr,Ytr,j,'cityblock');
% Err_knn_abs(j,1)=sum(abs(knn_abs-Yte)>0)/length(Yte);
% end
% Err_knn_abs_best(k,1)=min(Err_knn_abs(:,1));
% %% cosine
% %One minus the cosine of the included angle between points (treated as
% %vectors)
% for j=1:1:20    
% knn_cosine = knnclassify(Xte,Xtr,Ytr,j,'cosine');
% Err_knn_cosine(j,1)=sum(abs(knn_cosine-Yte)>0)/length(Yte);
% end
% Err_knn_cosine_best(k,1)=min(Err_knn_cosine(:,1));
%% correlation
%One minus the sample correlation between points (treated as sequences of
%values)

% for j=1:1:20    
% knn_corr = knnclassify(Xte,Xtr,Ytr,j,'correlation');
% Err_knn_corr(j,1)=sum(abs(knn_corr-Yte)>0)/length(Yte);
% end
% Err_knn_corr_best(k,1)=min(Err_knn_corr(:,1));






%% SVM 
%% linear kernel

%% Quadtratic Programming
svm_linear_kernel_QP=svmtrain(Xtr,Ytr,'Method','QP');
svm_linear = svmclassify(svm_linear_kernel_QP,Xte);
Err_svm_linear_QP(k,1)=sum(abs(svm_linear-Yte)>0)/length(Yte);
%% SMO
svm_linear_kernel_SMO=svmtrain(Xtr,Ytr,'Method','SMO');
svm_linear_SMO = svmclassify(svm_linear_kernel_SMO,Xte);
Err_svm_linear_SMO(k,1)=sum(abs(svm_linear-Yte)>0)/length(Yte);
%% LS
svm_linear_kernel_LS=svmtrain(Xtr,Ytr,'Method','LS');
svm_linear_LS = svmclassify(svm_linear_kernel_LS,Xte);
Err_svm_linear_LS(k,1)=sum(abs(svm_linear_LS-Yte)>0)/length(Yte);

%% quadratic kernel

%% QP
svm_quad_kernel_QP=svmtrain(Xtr,Ytr,'Kernel_Function','quadratic','Method','QP');
svm_quad_QP = svmclassify(svm_quad_kernel_QP,Xte);
Err_svm_quad_QP(k,1)=sum(abs(svm_quad_QP-Yte)>0)/length(Yte);
%% SMO
svm_quad_kernel_SMO=svmtrain(Xtr,Ytr,'Kernel_Function','quadratic','Method','SMO');
svm_quad_SMO = svmclassify(svm_quad_kernel_SMO,Xte);
Err_svm_quad_SMO(k,1)=sum(abs(svm_quad_SMO-Yte)>0)/length(Yte);
%% LS
svm_quad_kernel_LS=svmtrain(Xtr,Ytr,'Kernel_Function','quadratic','Method','LS');
svm_quad_LS = svmclassify(svm_quad_kernel_LS,Xte);
Err_svm_quad_LS(k,1)=sum(abs(svm_quad_LS-Yte)>0)/length(Yte);


%% rbf kernel
%% QP
for j=1:1:10
svm_rbf_kernel=svmtrain(Xtr,Ytr,'Kernel_Function','rbf','RBF_Sigma',j,'Method','QP');
svm_rbf = svmclassify(svm_rbf_kernel,Xte);
Err_svm_rbf_QP(j,1)=sum(abs(svm_rbf-Yte)>0)/length(Yte);
end
Err_svm_rbf_QP_best(k,1)=min(Err_svm_rbf_QP(:,1));
%% SMO
for j=1:1:10
svm_rbf_kernel=svmtrain(Xtr,Ytr,'Kernel_Function','rbf','RBF_Sigma',j,'Method','SMO');
svm_rbf = svmclassify(svm_rbf_kernel,Xte);
Err_svm_rbf_SMO(j,1)=sum(abs(svm_rbf-Yte)>0)/length(Yte);
end
Err_svm_rbf_SMO_best(k,1)=min(Err_svm_rbf_SMO(:,1));
%% LS
for j=1:1:10
svm_rbf_kernel=svmtrain(Xtr,Ytr,'Kernel_Function','rbf','RBF_Sigma',j,'Method','LS');
svm_rbf = svmclassify(svm_rbf_kernel,Xte);
Err_svm_rbf_LS(j,1)=sum(abs(svm_rbf-Yte)>0)/length(Yte);
end
Err_svm_rbf_LS_best(k,1)=min(Err_svm_rbf_LS(:,1));
% %% polynomial kernel
% %% QP
% for k=1:1:10
% svm_poly_kernel=svmtrain(Xtr,Ytr,'Kernel_Function','polynomial','polyorder',k,'Method','QP');
% svm_poly = svmclassify(svm_poly_kernel,Xte);
% Err_svm_poly_QP(k,1)=sum(abs(svm_poly-Yte)>0)/length(Yte);
% end
% %% SMO
% for k=1:1:10
% svm_poly_kernel=svmtrain(Xtr,Ytr,'Kernel_Function','polynomial','polyorder',k,'Method','SMO');
% svm_poly = svmclassify(svm_poly_kernel,Xte);
% Err_svm_poly_SMO(k,1)=sum(abs(svm_poly-Yte)>0)/length(Yte);
% end
% %% LS
% for k=1:1:10
% svm_poly_kernel=svmtrain(Xtr,Ytr,'Kernel_Function','polynomial','polyorder',k,'Method','LS');
% svm_poly = svmclassify(svm_poly_kernel,Xte);
% Err_svm_poly_LS(k,1)=sum(abs(svm_poly-Yte)>0)/length(Yte);
% end 
k=k+1;
end 
Err_svm_rbf_LS=mean(Err_svm_rbf_LS_best(:,1))
Err_svm_rbf_SMO=mean(Err_svm_rbf_SMO_best(:,1))
Err_svm_rbf_QP=mean(Err_svm_rbf_QP_best(:,1))
Err_svm_quad_LS=mean(Err_svm_quad_LS(:,1))
Err_svm_quad_SMO=mean(Err_svm_quad_SMO(:,1))
Err_svm_quad_QP=mean(Err_svm_quad_QP(:,1))
Err_svm_linear_LS=mean(Err_svm_linear_LS(:,1))
Err_svm_linear_SMO=mean(Err_svm_linear_SMO(:,1))
Err_svm_linear_QP=mean(Err_svm_linear_QP(:,1))
% Err_knn_corr=mean(Err_knn_corr_best(:,1))
% Err_knn_cosine=mean(Err_knn_cosine_best(:,1))
% Err_knn_abs=mean(Err_knn_abs_best(:,1))
%Err_knn_euclidian_avg=mean(transpose(Err_knn_euclidian))
Err_moholanbis=mean(Err_moholanbis(:,1))
Err_naive_quad=mean(Err_naive_quad(:,1))
Err_naive=mean(Err_naive(:,1))
Err_qda=mean(Err_qda(:,1))
Err_lda=mean(Err_lda(:,1))
