load('Threesources');
X = cell(1,nviews);
for v = 1:nviews
    X{1,v} = fea{v,1}';
    X{1,v} = NormalizeFea(X{1,v}, 0);
end 
fea = X;
%% Parameter setting
maxiteration = 200;
eta = 80; %graph regularization para
beta = 100; %diversity parameter
graph_k = 100; %neighbourhood size
layers = [100 50];   
%% build similarity graph
options = [];
options.k = graph_k;
options.WeightMode = 'HeatKernel';
for v_ind = 1:nviews        
    A_graph{v_ind} = constructA(X{v_ind}', options);   
end
Aopt = OptimalManifold(A_graph', nviews);
Dopt = constructD(Aopt);
%%
tic
[Z, H, dnorm , H_final] = ODD_NMF_function(maxiteration, Aopt, Dopt, fea, layers, gnd, beta, eta, graph_k);
time = toc;
rand('twister',5489);
%% Use spectral clustering to obtain clusters
if ~(any(any(isnan(H_final))) || any(any(isinf(H_final))))
    [CA F P Recall nmi AR] = evalResults_multiview_K(H_final, gnd);
    disp(['    NMI and std:       ',num2str(nmi(1)), ' , ', num2str(nmi(2))]);
    disp(['    Accuracy and std:  ',num2str(CA(1)), ' , ', num2str(CA(2))]);
    disp(['    F-score and std:   ',num2str(F(1)), ' , ', num2str(F(2))]);
end
