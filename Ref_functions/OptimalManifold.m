    %% Calculate AOpt and Lcompatible
function [AOpt] = OptimalManifold(A, m)
    for i = 1:m
        A{i,1} = full(A{i,1});
    end
    
    for i = 1:m
        AOpt(:,:,i) = A{i,1};
    end
AOpt = min(AOpt,[],3);
