clear all;

%% Creating the Dataset %%
cd MUSIC_DATA
cd blues
dataset1=create_new();
dataset1(:,end+1)=ones(100,1);
X=dataset1(:,1:end-1);
Y=dataset1(:,end);
[X Y] = shuffleData(X,Y);
[Xtr Xte Ytr Yte] = splitData(X,Y, .90);
training=cat(2,Xtr,Ytr);
test=cat(2,Xte,Yte);

cd ..
cd classical/
dataset1=create_new();
dataset1(:,end+1)=2.*ones(100,1);
X=dataset1(:,1:end-1);
Y=dataset1(:,end);
[X Y] = shuffleData(X,Y);
[Xtr Xte Ytr Yte] = splitData(X,Y, .90);
training1=cat(2,Xtr,Ytr);
test1=cat(2,Xte,Yte);
training=cat(1,training,training1);
test=cat(1,test,test1);

cd ..
cd country/
dataset1=create_new();
dataset1(:,end+1)=3.*ones(100,1);
X=dataset1(:,1:end-1);
Y=dataset1(:,end);
[X Y] = shuffleData(X,Y);
[Xtr Xte Ytr Yte] = splitData(X,Y, .90);
training1=cat(2,Xtr,Ytr);
test1=cat(2,Xte,Yte);
training=cat(1,training,training1);
test=cat(1,test,test1);
cd ..
cd disco/
dataset1=create_new();
dataset1(:,end+1)=4.*ones(100,1);
X=dataset1(:,1:end-1);
Y=dataset1(:,end);
[X Y] = shuffleData(X,Y);
[Xtr Xte Ytr Yte] = splitData(X,Y, .90);
training1=cat(2,Xtr,Ytr);
test1=cat(2,Xte,Yte);
training=cat(1,training,training1);
test=cat(1,test,test1);
cd ..
cd hiphop/
dataset1=create_new();
dataset1(:,end+1)=5.*ones(100,1);
X=dataset1(:,1:end-1);
Y=dataset1(:,end);
[X Y] = shuffleData(X,Y);
[Xtr Xte Ytr Yte] = splitData(X,Y, .90);
training1=cat(2,Xtr,Ytr);
test1=cat(2,Xte,Yte);
training=cat(1,training,training1);
test=cat(1,test,test1);
cd ..
cd jazz/
dataset1=create_new();
dataset1(:,end+1)=6.*ones(100,1);
X=dataset1(:,1:end-1);
Y=dataset1(:,end);
[X Y] = shuffleData(X,Y);
[Xtr Xte Ytr Yte] = splitData(X,Y, .90);
training1=cat(2,Xtr,Ytr);
test1=cat(2,Xte,Yte);
training=cat(1,training,training1);
test=cat(1,test,test1);
cd ..
cd metal/
dataset1=create_new();
dataset1(:,end+1)=7.*ones(100,1);
X=dataset1(:,1:end-1);
Y=dataset1(:,end);
[X Y] = shuffleData(X,Y);
[Xtr Xte Ytr Yte] = splitData(X,Y, .90);
training1=cat(2,Xtr,Ytr);
test1=cat(2,Xte,Yte);
training=cat(1,training,training1);
test=cat(1,test,test1);
cd ..
cd pop/
dataset1=create_new();
dataset1(:,end+1)=8.*ones(100,1);
X=dataset1(:,1:end-1);
Y=dataset1(:,end);
[X Y] = shuffleData(X,Y);
[Xtr Xte Ytr Yte] = splitData(X,Y, .90);
training1=cat(2,Xtr,Ytr);
test1=cat(2,Xte,Yte);
training=cat(1,training,training1);
test=cat(1,test,test1);
cd ..
cd reggae/
dataset1=create_new();
dataset1(:,end+1)=9.*ones(100,1);
X=dataset1(:,1:end-1);
Y=dataset1(:,end);
[X Y] = shuffleData(X,Y);
[Xtr Xte Ytr Yte] = splitData(X,Y, .90);
training1=cat(2,Xtr,Ytr);
test1=cat(2,Xte,Yte);
training=cat(1,training,training1);
test=cat(1,test,test1);
cd ..
cd rock/
dataset1=create_new();
dataset1(:,end+1)=10.*ones(100,1);
X=dataset1(:,1:end-1);
Y=dataset1(:,end);
[X Y] = shuffleData(X,Y);
[Xtr Xte Ytr Yte] = splitData(X,Y, .90);
training1=cat(2,Xtr,Ytr);
test1=cat(2,Xte,Yte);
training=cat(1,training,training1);
test=cat(1,test,test1);
cd ..
cd ..
cd MUSIC' PAPER'/

%% Feature Ranking Using J2 and J3 %%
n=size(training,2)-1; % number of features
J03=zeros(n,1);
J02=zeros(n,1);
for i=1:n
    [B W]=scattermat(training(:,i),training(:,end));
    J02(i,1)=abs(B+W)./abs(W);
end

for i=1:n
    [B W]=scattermat(training(:,i),training(:,end));
    J03(i,1)=abs(B)./abs(W);
end

[J2,I2]=sort(J02,'descend');
[J3,I3]=sort(J03,'descend');
