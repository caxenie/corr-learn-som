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
MAX_EPOCHS = 500;
% decay factors
ETA = 1; % activity decay
XI = 1e-2; % weights decay
%% INIT INPUT DATA - RELATION IS EMBEDDED IN THE INPUT DATA PAIRS
% set up the interval of interest
sensory_data.min_val         = -1.0;
sensory_data.max_val         = 1.0;
% setup the number of random input samples to generate
sensory_data.num_vals        = 250;
% generate NUM_VALS random samples in the given interval
% sensory_data.x  = sensory_data.min_val + rand(sensory_data.num_vals, 1)*(sensory_data.max_val - sensory_data.min_val);
% sensory_data.y  = sensory_data.x.^3;
% generate NUM_VALS consecutive samples in the given interval
sensory_data.x  = linspace(sensory_data.min_val, sensory_data.max_val, sensory_data.num_vals);
sensory_data.y  = sensory_data.x.^3;
%% CREATE NETWORK AND INITIALIZE
% create a network of SOMs given the simulation constants
populations = create_init_network(N_SOM, N_NEURONS, sensory_data.min_val, sensory_data.max_val);
% init activity vector
act_cur = zeros(populations(1).lsize, 1);
% init neighborhood function
hwi = zeros(populations(1).lsize, 1);
% init width of neighborhood function
sigma0 = populations(1).lsize/2;
sigmaf = 0.5;
sigmat = sigma0;
% init learning rate
alpha0 = 0.01;
alphaf = 0.001;
alphat = alpha0;
%% NETWORK SIMULATION LOOP
% % present each entry in the dataset for MAX_EPOCHS epochs to train the net
for didx = 1:length(sensory_data.x)
    % pick a new sample from the dataset and feed it to the current layer
    for pidx = 1:N_SOM
        if pidx == 1
            input_sample = sensory_data.x(didx);
        else
            input_sample = sensory_data.y(didx);
        end
        
        % re-init activity vector
        act_cur = zeros(populations(1).lsize, 1);
        
        % som network relaxation loop
        for t = 1:MAX_EPOCHS % loop for MAX_EPOCHS until relaxation
            
            % compute new activity given the current input sample
            for idx = 1:populations(pidx).lsize
                act_cur(idx) = (1/sqrt(2*pi*populations(pidx).s(idx)))*exp(-(input_sample - populations(pidx).Winput(idx))^2/(2*populations(pidx).s(idx)));
            end
            act_cur = act_cur./sum(act_cur);
            populations(pidx).a = (1-ETA)*populations(pidx).a + ETA*act_cur;
            
            % competition step: find the winner in the population given the input data
            % the winner is the neuron with the highest activity elicited
            % by the input sample
            [win_act, win_pos] = max(populations(pidx).a);
            
            % re-init neighborhood function
            hwi = zeros(populations(1).lsize, 1);
            
            for idx = 1:populations(pidx).lsize % go through each neuron in the population
                
                % cooperation step: compute the neighborhood kernel
                hwi(idx) = exp(-(idx - win_pos)^2/(2*sigmat^2));
                
                % learning step: update the input weights /
                % preferred value of a certain neuron in the population
                populations(pidx).Winput(idx) = populations(pidx).Winput(idx) + alphat*hwi(idx)*(input_sample - populations(pidx).Winput(idx));
                
                % update the spread of the tuning curve for the current
                % neuron in the population
                populations(pidx).s(idx) = populations(pidx).s(idx) + alphat*hwi(idx)*((input_sample - populations(pidx).Winput(idx))^2 - populations(pidx).s(idx));
            end
            
            % adapt learnig parameters
            alphat = alpha0 - (alpha0/(1+exp(-0.002*(t-MAX_EPOCHS/320)))) + alphaf;
            sigmat = sigma0 - (sigma0/(1+exp(-0.002*(t-MAX_EPOCHS/320)))) + sigmaf;
             

        end % end som relaxation
        
    end % end loop for each population
    
%     populations(1).a = population_encoder(sensory_data.x(didx), max(sensory_data.x(:)),  N_NEURONS);
%     populations(2).a = population_encoder(sensory_data.y(didx), max(sensory_data.y(:)),  N_NEURONS);
%     populations(1).a = populations(1).a./sum(populations(1).a);
%     populations(2).a = populations(2).a./sum(populations(2).a);
   
    % perform cross-modal Hebbian learnin
    populations(1).Wcross = (1-XI)*populations(1).Wcross + XI*populations(1).a*populations(2).a';
    populations(2).Wcross = (1-XI)*populations(2).Wcross + XI*populations(2).a*populations(1).a';
     
    % visualization
    if(DYN_VISUAL==1)
        visualize_runtime(sensory_data, populations, didx);
    end
    
end % end of all samples in the training dataset

% visualize post-simulation weight matrices encoding learned relation
visualize_runtime(sensory_data, populations, length(sensory_data.x));