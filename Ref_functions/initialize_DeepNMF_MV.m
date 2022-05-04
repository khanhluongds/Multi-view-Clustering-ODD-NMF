function [Z, H] = initialize_DeepNMF_MV(fea, layers)
% fea size is 1 x nviews
% fea{1,1} size is nsamp x mfea
nviews = size(fea,2);
% [a,b] = size(fea{1,1});
nlayers = length(layers);
label = cell(nviews,1);

for t = 1:nviews
    fea_new{1,t} = fea{1,t}';
    for layeri = 1:nlayers%eg 2 layers 100 50
        rand('twister',5489);
        label{t} = litekmeans(fea_new{1,t}, layers(layeri));  % doc
        for i = 1:layers(layeri) 
            H{t,layeri}(label{t} ==i, i) = 1;
        end
        H{t,layeri} = (H{t,layeri}+0.2)'; %H{view1,layer1} size 100 x nsamp
        mfea_viewt = size(fea_new{1,t},2);
        if layeri ==1
            Z{t,layeri} = ones(mfea_viewt, layers(layeri));
        else
            Z{t,layeri} = ones(layers(layeri-1), layers(layeri));
        end    
    end
end

