function [CA F P Recall nmi AR]= evalResults_multiview_K(H_final, truth,varargin )

if length(varargin)>0
    kk = varargin{1};
end

nClass = length(unique(truth));

if iscell(H_final)
    H_final = H_final{:};
end

for i = 1:20,
    options = [];
    options.k = 5;
    options.WeightMode = 'HeatKernel';
    options.t = 1;
    W = constructW(H_final', options);
    clusNum = nClass;
    
    idx_spec = SpectralClustering(W,clusNum);
    a = idx_spec;
    b = truth;
    nmii(i) = compute_nmi(a,b);
    CAi(i) = 1-compute_CE(idx_spec, truth); % clustering accuracy
    [Fi(i),Pi(i),Ri(i)] = compute_f(truth,idx_spec); % F1, precision, recall
    nmii(i) = compute_nmi(truth,idx_spec);
    ARi(i) = rand_index(truth,idx_spec);  
end
CA(1) = mean(CAi); CA(2) = std(CAi);
F(1) = mean(Fi); F(2) = std(Fi);
P(1) = mean(Pi); P(2) = std(Pi);
Recall(1) = mean(Ri); Recall(2) = std(Ri);
nmi(1) = mean(nmii); nmi(2) = std(nmii);
AR(1) = mean(ARi); AR(2) = std(ARi);
end

