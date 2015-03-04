%% SIMPLE IMPLEMENTATION OF THE UNSUPERVISED LEARNING OF RELATIONS NETWORK USING SOMs
%% PREPARE ENVIRONMENT
clear all; clc; close all; format long; pause(2);
%% INIT SIMULATION
% enables dynamic visualization on network runtime
DYN_VISUAL = 0;
% number of populations in the network
N_SOM      = 2;
% number of neurons in each population
N_NEURONS  = 100;
% MAX_EPOCHS for SOM relaxation
MAX_EPOCHS_IN_LRN = 40;
MAX_EPOCHS_XMOD_LRN = 100;
% number of data samples
N_SAMPLES = 800;
% decay factors
ETA = 1.0; % activity decay
XI = 1e-3; % weights decay
%% INIT INPUT DATA - RELATION IS EMBEDDED IN THE INPUT DATA PAIRS
% set up the interval of interest (i.e. +/- range)ststr
sensory_data.range  = 1.0;
% setup the number of random input samples to generate
sensory_data.num_vals = N_SAMPLES;
% choose between uniformly distributed data and non-uniform distribution
sensory_data.dist = 'uniform'; % {uniform, non-uniform}
% generate observations distributed as some continous heavy-tailed distribution.
% options are decpowerlaw, incpowerlaw and Gauss
% distribution
sensory_data.nufrnd_type  = 'gauss';
sensory_data.x = randnum_gen(sensory_data.dist, sensory_data.range, sensory_data.num_vals, sensory_data.nufrnd_type);
% switch between power-law relations (TODO add a more flexible way)
exponent=3;
sensory_data.y = sensory_data.x.^exponent;
%% CREATE NETWORK AND INITIALIZE PARAMS
% create a network of SOMs given the simulation constants
populations = create_init_network(N_SOM, N_NEURONS);
% init activity vector
act_cur = zeros(N_NEURONS, 1);
% init neighborhood function
hwi = zeros(N_NEURONS, 1);
% learning params
learning_params.t0 = 1;
learning_params.tf_learn_in = MAX_EPOCHS_IN_LRN;
learning_params.tf_learn_cross = MAX_EPOCHS_XMOD_LRN;
% init width of neighborhood kernel
sigma0 = N_NEURONS;
sigmaf = 1.0;
learning_params.sigmat = parametrize_learning_law(sigma0, sigmaf, learning_params.t0, learning_params.tf_learn_in, 'invtime');
% init learning rate
alpha0 = 0.1;
alphaf = 0.001;
learning_params.alphat = parametrize_learning_law(alpha0, alphaf, learning_params.t0, learning_params.tf_learn_in, 'invtime');
% cross-modal learning rule type
cross_learning = 'covariance';    % {hebb - Hebbian, covariance - Covariance, oja - Oja's Local PCA}
% mean activities for covariance learning
avg_act=zeros(N_NEURONS, N_SOM);
%% NETWORK SIMULATION LOOP
fprintf('Started training sequence ...\n');
% present each entry in the dataset for MAX_EPOCHS epochs to train the net
for t = 1:learning_params.tf_learn_cross
    for didx = 1:sensory_data.num_vals
        % loop through populations
        for pidx = 1:N_SOM
            % pick a new sample from the dataset and feed it to the current layer
            switch pidx
                case 1
                    input_sample = sensory_data.x(didx);
                case 2
                    input_sample = sensory_data.y(didx);
            end
            % compute new activity given the current input sample
            for idx = 1:populations(pidx).lsize
                act_cur(idx) = (1/(sqrt(2*pi)*populations(pidx).s(idx)))*...
                    exp(-(input_sample - populations(pidx).Winput(idx))^2/(2*populations(pidx).s(idx)^2));
                % normalize the activity vector of the population
                act_cur = act_cur./sum(act_cur);
                % update the activity for the next iteration
                populations(pidx).a = (1-ETA)*populations(pidx).a + ETA*act_cur;
                % competition step: find the winner in the population given the input data
                % the winner is the neuron with the highest activity elicited
                % by the input sample
                [win_act, win_pos] = max(populations(pidx).a);
                % cooperation step: compute the neighborhood kernel
                % simple Gaussian kernel with no boundary compensation
                hwi(idx) = exp(-abs(idx-win_pos)^2/(2*learning_params.sigmat(t)^2));
                % learning step: compute the weight update
                populations(pidx).Winput(idx) = populations(pidx).Winput(idx) + ...
                    learning_params.alphat(t)*hwi(idx)*(input_sample - populations(pidx).Winput(idx));
                % update the shape of the tuning curve for current neuron
                populations(pidx).s(idx) = populations(pidx).s(idx) + ...
                    learning_params.alphat(t)*...
                    exp(-abs(idx-win_pos)^2/(2*learning_params.sigmat(t)^2))*...
                    ((input_sample - populations(pidx).Winput(idx))^2 - populations(pidx).s(idx)^2);
                % compute new activity given the current input sample
                act_cur(idx) = (1/(sqrt(2*pi)*populations(pidx).s(idx)))*...
                    exp(-(input_sample - populations(pidx).Winput(idx))^2/(2*populations(pidx).s(idx)^2));
            end
            % normalize the activity vector of the population
            act_cur = act_cur./sum(act_cur);
            % update the activity for the next iteration
            populations(pidx).a = (1-ETA)*populations(pidx).a + ETA*act_cur; 
            % compute the mean value computation decay
            OMEGA = 0.002 + 0.998/(t+2);
            % compute the average activity for Hebbian covariance rule
            avg_act(:, pidx) = (1-OMEGA)*avg_act(:, pidx) + OMEGA*populations(pidx).a;
            % cross-modal Hebbian covariance learning rule: update the synaptic weights
            populations(1).Wcross = (1-XI)*populations(1).Wcross + XI*(populations(1).a - avg_act(:, 1))*(populations(2).a - avg_act(:, 2))';
            populations(2).Wcross = (1-XI)*populations(2).Wcross + XI*(populations(2).a - avg_act(:, 2))*(populations(1).a - avg_act(:, 1))';
            
        end % end for population pidx
    end % end samples in the dataset
end % allow the som to learn the sensory space data distribution
fprintf('Ended training sequence. Presenting results ...\n');
%% VISUALIZATION
present_tuning_curves(populations(1), sensory_data);
present_tuning_curves(populations(2), sensory_data);
% normalize weights between [0,1] for display
populations(1).Wcross = populations(1).Wcross ./ max(populations(1).Wcross(:));
populations(2).Wcross = populations(2).Wcross ./ max(populations(2).Wcross(:));
% visualize post-simulation weight matrices encoding learned relation
lrn_fct = visualize_results(sensory_data, populations, learning_params);
% save runtime data in a file for later analysis
runtime_data_file = sprintf('runtime_data_%d_soms_%d_neurons_%d_samples_data_dist_%s_%s_train_epochs_%d.mat',...
    N_SOM, N_NEURONS, N_SAMPLES, sensory_data.dist, sensory_data.nufrnd_type, MAX_EPOCHS_XMOD_LRN);
save(runtime_data_file);