%% SIMPLE IMPLEMENTATION OF THE UNSUPERVISED LEARNING OF RELATIONS NETWORK USING SOMs
%% PREPARE ENVIRONMENT
clear all; clc; close all; format long; pause(2);
%% INIT SIMULATION
% enables dynamic visualization on network runtime
DYN_VISUAL = 1;
% number of populations in the network
N_SOM      = 2;
% number of neurons in each population
N_NEURONS  = 100;
% max MAX_EPOCHS for SOM relaxation
MAX_EPOCHS = 400;
% number of data samples
N_SAMPLES = 2500;
% decay factors
ETA = 1.0; % activity decay
XI = 1e-3; % weights decay
%% INIT INPUT DATA - RELATION IS EMBEDDED IN THE INPUT DATA PAIRS
% switch between power-law relations (TODO add a more flexible way)
exponent=2;
% set up the interval of interest (i.e. +/- range)
sensory_data.range  = 1.0;
% setup the number of random input samples to generate
sensory_data.num_vals = N_SAMPLES;
% choose between uniformly distributed data and non-uniform distribution
sensory_data.dist = 'non-uniform'; % {uniform, non-uniform}
% generate observations distributed as some continous heavy-tailed distribution.
% options are decpowerlaw, incpowerlaw and Gauss
% distribution
nufrnd_type  = 'convex';
sensory_data.x = randnum_gen(sensory_data.dist, sensory_data.range, sensory_data.num_vals, nufrnd_type);
sensory_data.y = sensory_data.x.^exponent;
%% CREATE NETWORK AND INITIALIZE PARAMS
% create a network of SOMs given the simulation constants
populations = create_init_network(N_SOM, N_NEURONS);
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
% cross-modal learning rule type
cross_learning = 'covariance';    % {hebb - Hebbain, covariance - Covariance, oja - Oja's Local PCA}
% mean activities for covariance learning
avg1 = 0.0; avg2 = 0.0;
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
                    % populations(pidx).s(idx) = populations(pidx).s(idx) + ...
                    %    learning_params.alphat(t)*hwi(idx)* ...
                    %    ((input_sample - populations(pidx).Winput(idx))^2 - populations(pidx).s(idx)^2);
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
        % check which learning rule we employ
        switch(cross_learning)
            case 'hebb'
                % cross-modal Hebbian learning rule
                populations(1).Wcross = (1-XI)*populations(1).Wcross + XI*populations(1).a*populations(2).a';
                populations(2).Wcross = (1-XI)*populations(2).Wcross + XI*populations(2).a*populations(1).a';
            case 'covariance'
                % compute the mean value computation decay
                OMEGA = 0.002 + 0.998/(t+2);
                % compute the average activity for Hebbian covariance rule
                avg1 = (1-OMEGA)*avg1 + OMEGA*populations(1).a;
                avg2 = (1-OMEGA)*avg2 + OMEGA*populations(2).a;
                % cross-modal Hebbian covariance learning rule: update the synaptic weights
                populations(1).Wcross = (1-XI)*populations(1).Wcross + XI*(populations(1).a - avg1)*(populations(2).a - avg2)';
                populations(2).Wcross = (1-XI)*populations(2).Wcross + XI*(populations(2).a - avg2)*(populations(1).a - avg1)';
            case 'oja'
                % Oja's local PCA learning rule
                populations(1).Wcross = ((1-XI)*populations(1).Wcross + XI*populations(1).a*populations(2).a')/...
                    sqrt(sum(sum((1-XI)*populations(1).Wcross + XI*populations(1).a*populations(2).a')));
                populations(2).Wcross = ((1-XI)*populations(2).Wcross + XI*populations(2).a*populations(1).a')/...
                    sqrt(sum(sum((1-XI)*populations(2).Wcross + XI*populations(2).a*populations(1).a')));
        end
    end % end for values in dataset
end % end for training epochs
fprintf('Ended training sequence. Presenting results ...\n');
%% VISUALIZATION 
present_tuning_curves(populations(1), sensory_data);
present_tuning_curves(populations(2), sensory_data);
% normalize weights between [0,1] for display
populations(1).Wcross = populations(1).Wcross ./ max(populations(1).Wcross(:));
populations(2).Wcross = populations(2).Wcross ./ max(populations(2).Wcross(:));
% visualize post-simulation weight matrices encoding learned relation
lrn_fct = visualize_results(sensory_data, populations);