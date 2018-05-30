function match = euc_endMemMatch( W, W_hat )
% *match = euc_endMemMatch( W, W_hat )
% parameters **************************************************************
%  W          is the standard endmember matrix, or a endmember(that is a column)
%  W_hat      is the estimated endmember matrix, or a endmember(that is a column)
%  errs       is the eule distance between every column of
%  W and W_hat matrix by default, every column of W and W_hat is a endmember.
% *************************************************************************
%  author : zhu feiyun
%  time   : 2012-2-27
%  version: 1.0

epss = 1e-15;
% 废话 --------------------------------------------------------
[ nFea, nEnd ] = size( W );
[ nFea_hat, nEnd_hat] = size( W_hat );

if(nFea ~= nFea_hat || nEnd ~= nEnd_hat)
    error('size error: the two input matrix must have same size\n');
end

if( min(W(:))<0 || min(W_hat(:))<0 )
    error('the elements in the two input matrix must be positive\n');
end

if nFea < nEnd
    W = W.';
    [~, nEnd] = size( W );
end

if nFea_hat < nEnd_hat
    W_hat = W_hat.';
    %     [ row_hat, column_hat] = size( W_hat );
end
% ------------------------------------------------------------
% resacle each column of W to unit length
W = W * max( diag(sum(W.^2, 1).^-0.5), epss) ;
W_hat = W_hat * max( diag(sum(W_hat.^2, 1).^-0.5) , epss) ;

errs = zeros( nEnd );
for i = 1:nEnd
    for j = 1:nEnd
        errs(j, i) = norm( W_hat(:,i) - W(:,j) );
    end
end

match = zeros(nEnd, 2);
for i = 1:nEnd
    [match(i,1) match(i,2)] = find(errs == min(errs(:)));
    errs(match(i,1), :) = inf;
    errs(:, match(i,2)) = inf;
end
% 排序，将match的结果按照顺序排好
temp = match;
for i = 1:nEnd
   ta = find(temp(:, 1) == min(temp(:, 1)));
   match(i,:) = temp(ta,:);
   temp(ta,1) = inf;
end

end