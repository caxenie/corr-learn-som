%% SIMPLE IMPLEMENTATION OF THE UNSUPERVISED LEARNING OF RELATIONS NETWORK USING SOMs
%% PREPARE ENVIRONMENT
clear all; clc; close all;
%% INIT SIMULATION
% enables dynamic visualization on network runtime
DYN_VISUAL      = 1;
% number of populations in the network
N_SOM           = 2;
% number of neurons in each population
N_NEURONS       = 200;
% max MAX_EPOCHS for SOM relaxation
MAX_EPOCHS = 1000;
% decay factors
ETA = 1.0; % activity decay
XI = 1e-2; % weights decay
%%% INIT INPUT DATA - RELATION IS EMBEDDED IN THE INPUT DATA PAIRS
% set up the interval of interest (i.e. +/- range)
sensory_data.range  = 1.0;
% setup the number of random input samples to generate
sensory_data.num_vals = N_NEURONS;
% generate NUM_VALS random samples in the given interval
sensory_data.x  = -sensory_data.range + rand(sensory_data.num_vals, 1)*(2*sensory_data.range);
sensory_data.y = sensory_data.x.^3;
% generate NUM_VALS consecutive samples in the given interval
% sensory_data.x  = linspace(sensory_data.range, sensory_data.range, sensory_data.num_vals);
% sensory_data.y = x.^3;
%% CREATE NETWORK AND INITIALIZE
% create a network of SOMs given the simulation constants
populations = create_init_network(N_SOM, N_NEURONS);
% init activity vector
act_cur = zeros(N_NEURONS, 1);
% init neighborhood function
hwi = zeros(N_NEURONS, 1);
% learning params
t0 = 1;
tf = MAX_EPOCHS;
% init width of neighborhood function
sigma0 = N_NEURONS/2;
sigmaf = 0.5;
learning_params.sigmat = parametrize_learning_law(sigma0, sigmaf, t0, tf, 'invtime');
% init learning rate
alpha0 = 0.1;
alphaf = 0.001;
learning_params.alphat = parametrize_learning_law(alpha0, alphaf, t0, tf, 'invtime');
%% NETWORK SIMULATION LOOP
fprintf('Started training sequence ...\n');
% present each entry in the dataset for MAX_EPOCHS epochs to train the net
for t = 1:MAX_EPOCHS
    for didx = 1:length(sensory_data.x)
        % loop through populations
        for pidx = 1:N_SOM
            % pick a new sample from the dataset and feed it to the current layer
            if(pidx==1)
                input_sample = sensory_data.x(didx);
            else
                input_sample = sensory_data.y(didx);
            end
            % compute new activity given the current input sample
            for idx = 1:populations(pidx).lsize
                act_cur(idx) = (1/sqrt(2*pi*abs(populations(pidx).s(idx))))*...
                    exp(-(input_sample - populations(pidx).Winput(idx))^2/(2*populations(pidx).s(idx)));
            end
            % normalize the activity vector of the population
            act_cur = act_cur./sum(act_cur);
            % update the activity for the next iteration
            populations(pidx).a = (1-ETA)*populations(pidx).a + ETA*act_cur;
            % competition step: find the winner in the population given the input data
            % the winner is the neuron with the highest activity elicited
            % by the input sample
            [win_act, win_pos] = max(populations(pidx).a);
            for idx = 1:N_NEURONS % go through each neuron in the population
                % cooperation step: compute the neighborhood kernell
                hwi(idx) = exp(-norm(idx - win_pos)^2/(2*learning_params.sigmat(t)^2));
                % compute the weight update
                populations(pidx).Winput(idx) = populations(pidx).Winput(idx) + ...
                    learning_params.alphat(t)*hwi(idx)*(input_sample - populations(pidx).Winput(idx));
                % update the spread of the tuning curve for current neuron
                populations(pidx).s(idx) = 0.002025;
            end
        end % end for population pidx
    end % end samples in the dataset
end % end for training epochs
fprintf('Ended training sequence.\n');
fprintf('Start testing sequence ...\n');
for t = 1:MAX_EPOCHS
    % generate NUM_VALS random samples in the given interval
    sensory_data.x  = -sensory_data.range + rand(sensory_data.num_vals, 1)*(2*sensory_data.range);
    sensory_data.y  = sensory_data.x.^2;
    
    % use the learned weights and compute activation
    % loop through populations
    for pidx = 1:N_SOM
        % pick a new sample from the dataset and feed it to the current layer
        input_sample = sensory_data.x(didx);
        
        % compute new activity given the current input sample
        for idx = 1:populations(pidx).lsize
            act_cur(idx) = (1/sqrt(2*pi*abs(populations(pidx).s(idx))))*...
                exp(-(input_sample - populations(pidx).Winput(idx))^2/(2*populations(pidx).s(idx)));
        end
        % normalize the activity vector of the population
        act_cur = act_cur./sum(act_cur);
        % update the activity for the next iteration
        populations(pidx).a = (1-ETA)*populations(pidx).a + ETA*act_cur;
    end
    
    % perform cross-modal Hebbian learning
    populations(1).Wcross = (1-XI)*populations(1).Wcross + XI*populations(1).a*populations(2).a';
    populations(2).Wcross = (1-XI)*populations(2).Wcross + XI*populations(2).a*populations(1).a';
end % end for training epochs
fprintf('Ended testing sequence. Presenting results ...\n');
% normalize weights between [0,1]
populations(1).Wcross = populations(1).Wcross ./ max(populations(1).Wcross(:));
populations(2).Wcross = populations(2).Wcross ./ max(populations(2).Wcross(:));
% visualize post-simulation weight matrices encoding learned relation
visualize_runtime(sensory_data, populations(pidx), length(sensory_data.x), learning_params);