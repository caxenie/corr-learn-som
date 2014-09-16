%% SIMPLE IMPLEMENTATION OF THE UNSUPERVISED LEARNING OF RELATIONS NETWORK USING SOMs
%% PREPARE ENVIRONMENT
clear all; clc; close all; format long;
%% INIT SIMULATION
% enables dynamic visualization on network runtime
DYN_VISUAL = 1;
% number of populations in the network
N_SOM      = 2;
% number of neurons in each population
N_NEURONS  = 200;
% max MAX_EPOCHS for SOM relaxation
MAX_EPOCHS = 200;
% number of data samples
N_SAMPLES = 4444;
% decay factors
ETA = 1.0; % activity decay
XI = 1e-2; % weights decay
%%% INIT INPUT DATA - RELATION IS EMBEDDED IN THE INPUT DATA PAIRS
% switch between power-law relations (TODO add a more flexible way)
exponent=2;
% set up the interval of interest (i.e. +/- range)hack the wowee
sensory_data.range  = 1.0;
% setup the number of random input samples to generate
sensory_data.num_vals = N_SAMPLES;
% choose between uniformly distributed data and non-uniform distribution
sensory_data.dist = 'non-uniform'; % {uniform, non-uniform}
% choose between random data / sequentially ordered data to present to net
sensory_data.order  = 'random';  % {random, ordered}
% non-unifrom distribution type {power-law, Gauss dist}
nufrnd_type       = 'plaw';
% generate training data
switch (sensory_data.dist)
    case 'uniform'
        switch(sensory_data.order)
            case 'random'
                % generate NUM_VALS random samples in the given interval
                sensory_data.x  = -sensory_data.range + rand(sensory_data.num_vals, 1)*(2*sensory_data.range);
                sensory_data.y = sensory_data.x.^exponent;
            case 'ordered'
                % generate NUM_VALS consecutive samples in the given interval
                sensory_data.x  = linspace(-sensory_data.range, sensory_data.range, sensory_data.num_vals);
                sensory_data.y = sensory_data.x.^exponent;
        end
    case 'non-uniform'
        switch(sensory_data.order)
            case 'random'
                switch nufrnd_type
                    case 'plaw'
                        % generate NUM_VALS random samples in the given interval
                        sensory_data.x = rand(sensory_data.num_vals, 1)*(sensory_data.range);
                        sensory_data.x = nufrnd_plaw(sensory_data.x, 0.000001, sensory_data.range, exponent);
                        sensory_data.y = sensory_data.x.^exponent;
                    case 'gauss'
                        % generate NUM_VALS random samples in the given interval
                        sensory_data.x  = randn(sensory_data.num_vals, 1)*(2*sensory_data.range/10);
                        sensory_data.y = sensory_data.x.^exponent;
                end
            case 'ordered'
                switch nufrnd_type
                    case 'plaw'
                        % generate NUM_VALS consecutive samples in the given interval
                        sensory_data.x  = linspace(0.000001, sensory_data.range, sensory_data.num_vals);
                        sensory_data.x = nufrnd_plaw(sensory_data.x, 0.000001, sensory_data.range, exponent);
                        sensory_data.y = sensory_data.x.^exponent;
                    case 'gauss'
                        sensory_data.x  = linspace(0.000001, sensory_data.range, sensory_data.num_vals);
                        sensory_data.x  = randn(sensory_data.num_vals, 1)*(2*sensory_data.range/10);
                        sensory_data.y = sensory_data.x.^exponent;
                end
        end
end
%% CREATE NETWORK AND INITIALIZE
% create a network of SOMs given the simulation constants
populations = create_init_network(N_SOM, N_NEURONS, N_SAMPLES);
% init activity vector
act_cur = zeros(N_NEURONS, 1);
% init neighborhood function
hwi = zeros(N_NEURONS, 1);
% learning params
t0 = 1;
tf_learn_in = MAX_EPOCHS/10;
tf_learn_cross = MAX_EPOCHS;
% init width of neighborhood kernel
sigma0 = N_NEURONS/2;
sigmaf = 1.0;
learning_params.sigmat = parametrize_learning_law(sigma0, sigmaf, t0, tf_learn_in, 'invtime');
% init learning rate
alpha0 = 0.1;
alphaf = 0.001;
learning_params.alphat = parametrize_learning_law(alpha0, alphaf, t0, tf_learn_in, 'invtime');
%% NETWORK SIMULATION LOOP
fprintf('Started training sequence ...\n');
% present each entry in the dataset for MAX_EPOCHS epochs to train the net
for t = 1:tf_learn_cross
    % learn the sensory space data distribution
    if(t<tf_learn_in)
        for didx = 1:sensory_data.num_vals
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
                    act_cur(idx) = (1/(sqrt(2*pi)*populations(pidx).s(idx)))*...
                        exp(-(input_sample - populations(pidx).Winput(idx))^2/(2*populations(pidx).s(idx)^2));
                end
                % normalize the activity vector of the population
                act_cur = act_cur./sum(act_cur);
                % update the activity for the next iteration
                populations(pidx).a = (1-ETA)*populations(pidx).a + ETA*act_cur;
                % competition step: find the winner in the population given the input data
                % the winner is the neuron with the highest activity elicited
                % by the input sample
                [win_act, win_pos] = max(populations(pidx).a);
                for idx = 1:N_NEURONS % go through neurons in the population
                    % cooperation step: compute the neighborhood kernell
                    hwi(idx) = exp(-norm(idx - win_pos)^2/(2*learning_params.sigmat(t)^2));
                    % learning step: compute the weight update
                    populations(pidx).Winput(idx) = populations(pidx).Winput(idx) + ...
                        learning_params.alphat(t)*hwi(idx)*(input_sample - populations(pidx).Winput(idx));
                    % update the spread of the tuning curve for current neuron
                    % at the moment we consider uniformly distributed values
                    % with the same spread of the neurons tuning curves
                    switch(sensory_data.dist)
                        case 'uniform'
                            populations(pidx).s(idx) = (N_NEURONS/N_SAMPLES);
                        case 'non-uniform'
                            populations(pidx).s(idx) = populations(pidx).s(idx) + ...
                                learning_params.alphat(t)*hwi(idx)* ...
                                ((input_sample - populations(pidx).Winput(idx))^2 - populations(pidx).s(idx)^2);
                    end
                end
            end % end for population pidx
        end % end samples in the dataset
    end % allow the som to learn the sensory space data distribution 
    % learn the cross-modal correlation
    for didx = 1:sensory_data.num_vals
        % use the learned weights and compute activation
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
                act_cur(idx) = (1/(sqrt(2*pi)*populations(pidx).s(idx)))*...
                    exp(-(input_sample - populations(pidx).Winput(idx))^2/(2*populations(pidx).s(idx)^2));
            end
            % normalize the activity vector of the population
            act_cur = act_cur./sum(act_cur);
            % update the activity for the next iteration
            populations(pidx).a = (1-ETA)*populations(pidx).a + ETA*act_cur;
        end
        
        % cross-modal Hebbian learning step: update the hebbian weights
        populations(1).Wcross = (1-XI)*populations(1).Wcross + XI*populations(1).a*populations(2).a';
        populations(2).Wcross = (1-XI)*populations(2).Wcross + XI*populations(2).a*populations(1).a';
    end
    
end % end for training epochs
fprintf('Ended training sequence. Presenting results ...\n');
present_tuning_curves(populations(1), sensory_data);
present_tuning_curves(populations(2), sensory_data);
% normalize weights between [0,1]
populations(1).Wcross = populations(1).Wcross ./ max(populations(1).Wcross(:));
populations(2).Wcross = populations(2).Wcross ./ max(populations(2).Wcross(:));
% visualize post-simulation weight matrices encoding learned relation
visualize_runtime(sensory_data, populations(pidx), length(sensory_data.x), learning_params);