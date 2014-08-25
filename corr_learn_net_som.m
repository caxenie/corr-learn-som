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
% decay factor
ETA = 0.5;
%% INIT INPUT DATA - RELATION IS EMBEDDED IN THE INPUT DATA PAIRS
% set up the interval of interest
MIN_VAL         = -1.0;
MAX_VAL         = 1.0;
% setup the number of random input samples to generate
NUM_VALS        = 250;
% generate NUM_VALS random samples in the given interval
sensory_data.x  = MIN_VAL + rand(NUM_VALS, 1)*(MAX_VAL - MIN_VAL);
sensory_data.y  = sensory_data.x.^3;
%% CREATE NETWORK AND INITIALIZE
% create a network of SOMs given the simulation constants
populations = create_init_network(N_SOM, N_NEURONS, MIN_VAL, MAX_VAL);
%% NETWORK SIMULATION LOOP
% % present each entry in the dataset for MAX_EPOCHS epochs to train the net
for didx = 1:length(sensory_data.x)
    % pick a new sample from the dataset, make a pair and feed it to each layer
    for pidx = 1:N_SOM
        if pidx ==1
            input_data = sensory_data.x(didx);
        else
            input_data = sensory_data.y(didx);
        end
        
        % init width of neighborhood function
        sigma0 = populations(pidx).lsize/2+1;
        sigmaf = 0.5; sigmat = sigma0;
        % init learning rate
        alpha0 = 0.1; alphat = alpha0;
        alphaf = 0.001;
        % init neighborhood function
        hwi = zeros(populations(pidx).lsize, MAX_EPOCHS);
        % init activity vector
        act_cur = zeros(populations(pidx).lsize, 1);
        
        % som network relaxation loop
        for t = 1:MAX_EPOCHS % loop for MAX_EPOCHS until relaxation
            % compute new activity given the current input sample
            % compute the Gaussian centered on the current unit
            for idx = 1:populations(pidx).lsize
                act_cur(idx) = (1/sqrt(2*pi*abs(populations(pidx).s(idx))))*exp(-(input_data - populations(pidx).Winput(idx))^2/(2*populations(pidx).s(idx)));
            end
            populations(pidx).a = act_cur./sum(act_cur);
            
            % competition step: find the winner in the population given the input data
            % the winner is the neruon with the highest activity
            [win_act, win_pos] = max(populations(pidx).a);
            
            for idx = 1:populations(pidx).lsize % go through each neuron in the population
               
                % cooperation step: compute the neighborhood kernel
                hwi(idx,t) = exp(-(idx - win_pos)^2/(2*sigmat^2));
                
                % learning step: update the input weights - mean value, or
                % preferred value of a certain neuron in the population
                populations(pidx).Winput(idx) = populations(pidx).Winput(idx) + alphat*hwi(idx, t)*(input_data - populations(pidx).Winput(idx));
                
                % update the spread of the tuning curve for the current
                % neuron in the population
                populations(pidx).s(idx) = populations(pidx).s(idx) + alphat*hwi(idx, t)*((input_data - populations(pidx).Winput(idx))^2 - populations(pidx).s(idx));
            end
                            % adapt the learnig params
            alphat = alpha0 - (alpha0/(1+exp(-0.0002*(t-MAX_EPOCHS/320)))) + alphaf;
            sigmat = sigma0 - (sigma0/(1+exp(-0.0002*(t-MAX_EPOCHS/320)))) + sigmaf;
            
        end % end som relaxation
    end % end loop for each population
    % perform cross-modal Hebbian learning
    populations(1).Wcross = (1-ETA)*populations(1).Wcross + ETA*populations(1).a'*populations(2).a;
    populations(2).Wcross = (1-ETA)*populations(2).Wcross + ETA*populations(2).a'*populations(1).a;
    % normzalize cross-modal weights
    populations(1).Wcross = populations(1).Wcross./sum(populations(1).Wcross(:));
    populations(2).Wcross = populations(2).Wcross./sum(populations(2).Wcross(:));
    if(DYN_VISUAL==1)
        visualize_runtime(sensory_data, populations, didx);
    end
end % end of all samples in the training dataset
% visualize post-simulation data
visualize_runtime(sensory_data, populations, length(sensory_data.x));