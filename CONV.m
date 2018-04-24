function conv_filt = CONV(x,w)
% x is row of original data (1,3072 uint8)
% w is weight vector, b is bias
x_color_square = reshape(x,[32,32,3]);

filt_size = 4; % allows 8 full sweeps
stride = 2; % skip

K = (32 - filt_size)/stride + 1; % number of receptive fields

w = zeros(K^2,1,3);





% % make receptive layer into a column test
% %col = reshape(x_color_square(1:4,1:4,:),[48,1]);
% receptive_columns = zeros(filt_size^2*3,K^2);
% 
% % This matrix contains 1 column for each receptive field. The column is a
% % squished version of the filt_size*filt_size*3 field: first are red values
% % (layer 1), then green, then blue. The first filt_size values in the
% % vector are the first column of the red values; next filt_size values are
% % the second column, then third column. 
% for i = 0:K-1
%     for j = 0:K-1
%         receptive_columns(:,i*2+1 + j) = ...
%             reshape(x_color_square(i*2+1:i*2+4,j*2+1:j*2+4,:),[filt_size^2*3,1]);
%     end
% end

% not really what this will be
conv_filt = receptive_columns;