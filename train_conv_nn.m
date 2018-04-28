function activations = train_conv_nn(X,y)



%% Parameters (eventually will be function arguments)
conv_nodes = 3;
output_nodes = 2;
eta = 0.00001;
num_classes = 10; % will eventually be = output_nodes, after testing

%% Vectorize y
Y = zeros(num_classes,1);
Y(y+1) = 1;

%% Initialize weights
weights = randn(64+1,conv_nodes); % 8x8 conv output
output_weights = rand(13*13*conv_nodes,output_nodes)*.00001;
conv_outputs = zeros(conv_nodes,13*13); % 13x13 output volume

%% CONV layer
x_color_square = reshape(X(1,:),[32,32,3]); % THIS ARRANGES BY COLUMNS INSTEAD OF ROWS!!
x_color_square = permute(x_color_square,[2,1,3]); % this should fix it

% Filter settings for cifar data
filt_spec = [8,0,2];

% NOTE - this is doing parameter sharing (weights shared by all layers)
for i = 1:conv_nodes
    [co,rec_cols] = CONV(x_color_square,weights(:,i),filt_spec);
    
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

%% START BACKPROP
% output nodes
delta_output = zeros(output_nodes,1);
for i = 1:output_nodes
    delta_j = (activations(i) - Y(i)) * activations(i) * (1 - activations(i));
    output_weights(:,i) = output_weights(:,i) - eta * conv_vector.' * delta_j;
    delta_output(i) = delta_j;
end

% conv nodes
delta_j = sum(output_weights .* delta_output.',2) .* (conv_vector > 0).';
delta_j = reshape(delta_j,[169,3]);


weight_updates = weights * 0;
for j = 1:3 % 3 is the depth dimension
    for i = 1:conv_nodes
        weight_updates(:,i) = weight_updates(:,i) + sum(delta_j(:,i) .* rec_cols(:,:,j)).';
    end
end

weight_updates = weight_updates/3; % 3 is the depth dimension
weights = weights - eta * weight_updates;


