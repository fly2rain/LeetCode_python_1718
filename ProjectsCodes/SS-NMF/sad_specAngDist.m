function sad = sad_specAngDist( w, w_hat, epss )
% ** SAD = SAD_specAngDist( W, W_hat )
% parameters **************************************************************
    %  W          is the standard endmember matrix, or a endmember(that is a column)
    %  W_hat      is the estimated endmember matrix, or a endmember(that is a column)
    %  SAD        is the spectral angle distance between every column of
    %  W and W_hat matrix by default, every column of W and W_hat is a endmember.
% *************************************************************************
    %  author : zhu feiyun
    %  time   : 2012-2-22
    %  version: 1.0
if( length(w) ~= length(w_hat))
    error('size error: the two input matrix must have same size\n');
end

if( min(w(:))<0 || min(w_hat(:))<0 )
    error('the elements in the two input matrix must be positive\n');
end

w       = max(w, epss);
w_hat   = max(w_hat, epss); 

w       = w ./ norm(w);
w_hat   = w_hat ./ norm(w_hat);

sad     = acos(sum(w .* w_hat) );
end