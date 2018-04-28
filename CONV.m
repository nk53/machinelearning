function [conv_filt,receptive_columns] = CONV(x_color_square,weights,filt_spec)
% x is row of original data (1,3072 uint8)
% w is weight vector, b is bias
% filt_spec - [f,p,s]

% x_color_square = reshape(x,[32,32,3]); % THIS ARRANGES BY COLUMNS INSTEAD OF ROWS!!
% x_color_square = permute(x_color_square,[2,1,3]); % this should fix it

width = size(x_color_square,1);

% filt_size = 8;
filt_size = filt_spec(1);
% stride = 2;
stride = filt_spec(3);

K = (size(x_color_square,1) - filt_size)/stride + 1; % number of receptive fields

receptive_columns = zeros(K^2,filt_size^2+1,3);

% Not sure about generalizability, but this works for 32x32 with 8,2
% Seems to generalize well, as long as no padding is needed
for i = 1:size(x_color_square,3)
    X = reshape(x_color_square(:,:,i),[width,width]);
    A = im2col(X,[filt_size,filt_size]);
    B = A(:,1:stride:end);
    C = [];
    for j = 1:2*K-1:size(B,2)
        C = [C B(:,j:j+K-1)];        
    end
    C = [C ; ones(1,size(C,2))]; % ADD BIAS TERMS
    receptive_columns(:,:,i) = C.'; % transpose for easy multiplication
end

conv_filt = zeros(K^2,1);
for i = 1:size(receptive_columns,3)
   conv_filt = conv_filt + receptive_columns(:,:,i) * weights; 
end

% conv_filt = reshape(conv_filt,[K,K]);

% Result (receptive fields)
%  -------------
% |1   ...   K+1|
% |2   ...   .  |
% |3   ...   .  |
% |... ...   .  |
% |K   ...   K^2|
%  -------------


% Correct for 32x32!
% X = reshape(1:32^2,[32,32]);
% A = im2col(X,[8,8]);
% B = A(:,1:2:625);
% C = [];
% for i = 1:25:313
% C = [C B(:,i:i+12)];
% end



