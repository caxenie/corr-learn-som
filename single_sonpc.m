%% SIMPLE IMPLEMENTATION OF THE SONPC - SELF ORGANIZING NEURAL POPULATION CODE
clear all; clc; close all;
% number of neurons in each population
N_NEURONS  = 200;
% max MAX_EPOCHS for SOM relaxation
MAX_EPOCHS = 1000;
% decay factors
ETA = 1.0; % activity decay
%% INIT INPUT DATA - REL    ATION IS EMBEDDED IN THE INPUT DATA PAIRS
% set up the interval of interest (i.e. +/- range)
sensory_data.range  = 1.0;
% setup the number of random input samples to generate
sensory_data.num_vals = N_NEURONS;
% generate NUM_VALS random samples in the given interval
sensory_data.x  = -sensory_data.range + rand(sensory_data.num_vals, 1)*(2*sensory_data.range);
% generate NUM_VALS consecutive samples in the given interval
% sensory_data.x  = linspace(sensory_data.range, sensory_data.range, sensory_data.num_vals);
%% CREATE NETWORK AND INITIALIZE
% create a network of SOMs given the simulation constants
population = create_init_network(1, N_NEURONS);
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
% present each entry in the dataset for MAX_EPOCHS epochs to train the net
for t = 1:MAX_EPOCHS
    for didx = 1:length(sensory_data.x)
        % pick a new sample from the dataset and feed it to the current layer
        input_sample = sensory_data.x(didx);
        
        % compute new activity given the current input sample
        for idx = 1:population.lsize
            act_cur(idx) = (1/sqrt(2*pi*abs(population.s(idx))))*...
                exp(-(input_sample - population.Winput(idx))^2/(2*population.s(idx)));
        end
        % normalize the activity vector of the population
        act_cur = act_cur./sum(act_cur);
        % update the activity for the next iteration
        population.a = (1-ETA)*population.a + ETA*act_cur;
        
        % competition step: find the winner in the population given the input data
        % the winner is the neuron with the highest activity elicited
        % by the input sample
        [win_act, win_pos] = max(population.a);
        for idx = 1:N_NEURONS % go through each neuron in the population
            
            % cooperation step: compute the neighborhood kernell
            hwi(idx) = exp(-norm(idx - win_pos)^2/(2*learning_params.sigmat(t)^2));
            % compute the weight update
            population.Winput(idx) = population.Winput(idx) + ...
                learning_params.alphat(t)*hwi(idx)*(input_sample - population.Winput(idx));
            % update the spread of the tuning curve for current neuron
           population.s(idx) = 0.002025;
%             population.s(idx) = population.s(idx) + ...
%                 learning_params.alphat(t)*hwi(idx)*...
%                 ((input_sample - population.Winput(idx))^2 - population.s(idx));
%             population.s(idx) = population.s(idx) + ...
%                 learning_params.alphat(t)*...
%                 (0.5*(input_sample - population.Winput(idx))^2 - population.s(idx));
        end
    end % end samples in the dataset
end % end for training epochs
% visualize the learned tuning curves
present_tuning_curves(population, sensory_data);