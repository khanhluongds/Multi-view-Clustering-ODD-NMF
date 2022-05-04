function D = constructD(W)
	nSmp = size(W,1);
% 	if alpha > 0
		%the degree is sum of rows for W, and create diagonal matrix with degree values
% 		W = alpha*W;
		DCol = full(sum(W,2));
		D = spdiags(DCol,0,nSmp,nSmp);
% 	end