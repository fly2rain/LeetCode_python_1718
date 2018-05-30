function [ row, col ]= rescale_subplot( num_fig )
% [ row, col ]= rescale_subplot( end_num )
% function:
%           rescale the subplot to fill the window beautifully
% parameters: ********************************************************
  % input arguments:
    %     num_fig:    the number of the fig to be plotted
  % output arguments:
    %     row:        the row of subplot
    %     column:     the column of subplot
% ********************************************************************
    % by :   zhu feiyun 
    % data:  2012-2-23
    % 
    row = floor( num_fig ^ (1/2) );
    col = ceil( num_fig ^ (1/2) );
    if( row * col < num_fig )
        if( row == col )
            col = col + 1;
        else
            row = row + 1;
        end
    end
end
