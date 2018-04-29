function prediction = test_conv_nn(X,weights,output_weights)



%% Parameters (eventually will be function arguments)
conv_nodes = 10;
output_nodes = 2;
eta = 0.00001;
num_classes = 10; % will eventually be = output_nodes, after testing

%% Initialize weights
% weights = randn(64+1,conv_nodes); % 8x8 conv output
% output_weights = rand(13*13*conv_nodes,output_nodes)*.00001;
conv_outputs = zeros(conv_nodes,13*13); % 13x13 output volume

%% CONV layer
x_color_square = reshape(X(1,:),[32,32,3]); % THIS ARRANGES BY COLUMNS INSTEAD OF ROWS!!
x_color_square = permute(x_color_square,[2,1,3]); % this should fix it

% Filter settings for cifar data
filt_spec = [8,0,2];

% NOTE - this is doing parameter sharing (weights shared by all layers)
for i = 1:conv_nodes
    [co,~] = CONV(x_color_square,weights(:,i),filt_spec);
    
    %% RELU
    co(co < 0) = 0;
    conv_outputs(i,:) = co;
end

%% FULLY CONNECTED OUTPUT LAYER
conv_vector = [];
for i = 1:conv_nodes
    conv_vector = [conv_vector conv_outputs(i,:)];
end

activations = 1 ./ (1 + exp(-conv_vector * output_weights));
[~,prediction] = max(activations);
prediction = prediction - 1;