
y=Sheet1(:,end);
x=Sheet1(:,2:end-1);

%% find unique classes and corresponding feature rows
UniqueClass = unique(y);
ClassA = (y == UniqueClass(1));
ClassB = (y == UniqueClass(2));
nA     = sum(double(ClassA));
nB     = sum(double(ClassB));

%% calculate FDR
FDR = ...
    abs(mean(x(ClassA,:)) - mean(x(ClassB,:)))  ./  ...
    sqrt(var(x(ClassA,:)) ./ nA + var(x(ClassB,:)) ./ nB);
R=transpose(FDR);
[S(:,1),I]=sort(R,'descend'); % S has the sorted values and I has feature index

%% create sorted dataset
dataset_FDR_sorted=zeros(197,25);
dataset_FDR_sorted(:,25)=Sheet1(:,end);

for i=1:24
    m=I(i);
 dataset_FDR_sorted(:,i)=Sheet1(:,m+1);
end

