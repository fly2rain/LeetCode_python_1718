function rmse = rmse_endMember( w, w_hat)
% ** sid = sid_specInforDiverg( w, w_hat, epss )
% parameters **************************************************************
    %  W          is the standard endmember matrix, or a endmember(that is a column)
    %  W_hat      is the estimated endmember matrix, or a endmember(that is a column)
    %  SAD        is the spectral angle distance between every column of
    %  W and W_hat matrix by default, every column of W and W_hat is a endmember.
% *************************************************************************
    %  author : zhu feiyun
    %  time   : 2012-7-26
    %  version: 2.0
    
if( length(w) ~= length(w_hat))
    error('size error: the two input matrix must have same size\n');
end

if( min(w(:))<0 || min(w_hat(:))<0 )
    error('the elements in the two input matrix must be positive\n');
end

w       = w ./ max(w);
w_hat   = w_hat ./ max(w_hat); 

rmse =  sqrt(1/length(w) * norm(w - w_hat, 2));
end