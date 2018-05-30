function  [ H_hat, W_hat ]= rescale_WH( H_hat, W, W_hat )
    % [ H_hat, W_hat ]= rescale_WH( H_hat, W, W_hat )
    % parameters: ********************************************************
    %    W:        endMember matrix, each column is a endmember spectral
    %    H:        abundant matrix, each row is a kind of endmember's  abundant
    % ********************************************************************
    %    by :   zhu feiyun
    %    data:  2012-2-23
    %
    
    epss = 1e-15;
    [ row_H, col_H ] = size( H_hat );
    
    if( nargin == 1 )
        if( nargout ~= 1 )
            error('at the condition of 1 input argument, must 1 output argument.\n');
        end
        H_hat  = H_hat ./ repmat( sum( H_hat ), [ row_H, 1 ] );
        
    elseif(nargin == 3)
        [ row_W, col_W ] = size( W_hat );
        
        if( col_W ~= row_H )
            error('the number of column of W must be equal to the number of row of H.\n');
        end
        
     
        hMax_ratio = max( W ) ./ ( max( W_hat ) + epss );
        W_hat      = W_hat .* repmat( hMax_ratio, [ row_W, 1 ] );
        H_hat      = H_hat ./ repmat( sum( H_hat ), [ row_H, 1 ] );        
    else
        error('must be 1 or 3 input arguments.\n');
    end    
    %   D = zeros( end_num );
    %   for i = 1 : end_num
    %       D(i, i) = norm( W(:,i) );
    %   end
    %   W = W / D;
    %   H = D * H;
    
    
    
    