function [weights,output_weights] = drive_conv_nn_sgd(X,Y)

conv_nodes = 10;
output_nodes = 10;
eta = 0.00001;
num_classes = 10; % will eventually be = output_nodes, after testing

weights = randn(64+1,conv_nodes); % 8x8 conv output
output_weights = rand(13*13*conv_nodes,output_nodes)*.00001;

for i = 1:10000
    j = randi(size(X,1));
    [weights,output_weights] = train_conv_nn(X(j,:),Y(j),weights,output_weights);
end