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
MAX_EPOCHS = 10000;
% decay factors
ETA = 0.5; % activity decay
XI = 1e-4; % weights decay
% enable/ disable boundry effect correction {1/0}
FIX_BOUNDS = 0;
%% INIT INPUT DATA - RELATION IS EMBEDDED IN THE INPUT DATA PAIRS
% set up the interval of interest
sensory_data.min_val         = -1.0;
sensory_data.max_val         = 1.0;
% setup the number of random input samples to generate
sensory_data.num_vals        = 1000;
% generate NUM_VALS random samples in the given interval
sensory_data.x  = sensory_data.min_val + rand(sensory_data.num_vals, 1)*(sensory_data.max_val - sensory_data.min_val);
sensory_data.y  = sensory_data.x.^3;
% generate NUM_VALS consecutive samples in the given interval
% sensory_data.x  = linspace(sensory_data.min_val, sensory_data.max_val, sensory_data.num_vals);
% sensory_data.y  = sensory_data.x.^3;
%% CREATE NETWORK AND INITIALIZE
% create a network of SOMs given the simulation constants
populations = create_init_network(N_SOM, N_NEURONS, sensory_data.min_val, sensory_data.max_val);
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
learning_params.sigmat = parametrize_learning_law(sigma0, sigmaf, t0, tf, 'sigmoid');
% init learning rate
alpha0 = 0.1;
alphaf = 0.001;
learning_params.alphat = parametrize_learning_law(alpha0, alphaf, t0, tf, 'sigmoid');
%% NETWORK SIMULATION LOOP
% % present each entry in the dataset for MAX_EPOCHS epochs to train the net
for t = 1:MAX_EPOCHS
    for didx = 1:length(sensory_data.x)
        % pick a new sample from the dataset and feed it to the current layer
        for pidx = 1:N_SOM
            
            % check the input depending on the population chose for update
            if pidx == 1
                input_sample = sensory_data.x(didx);
            else
                input_sample = sensory_data.y(didx);
            end
            
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
            for idx = 1:N_NEURONS % go through each neuron in the population
                % check if we compensate for boundry effects
                if (FIX_BOUNDS==1)
                    % compute the reflected mean (Zhou, Dudek, Shi - IJCNN 2011)
                    if (win_pos==1)
                        mw = 0.75*populations(pidx).Winput(1) + 0.25*populations(pidx).Winput(2);
                    end
                    if (win_pos==N_NEURONS)
                        mw = 0.75*populations(pidx).Winput(N_NEURONS) + 0.25*populations(pidx).Winput(N_NEURONS-1);
                    end
                    if(win_pos>1)
                        mw = populations(pidx).Winput(win_pos);
                    end
                    if(win_pos<N_NEURONS)
                        mw = populations(pidx).Winput(win_pos);
                    end
                    % compute the border means (both extremities of the population)
                    mr1 = 2*populations(pidx).Winput(1) - mw;
                    mrn = 2*populations(pidx).Winput(N_NEURONS) - mw;
                    % compute the border neighborhood kernel values
                    h2nwi = exp(-(idx - (2*N_NEURONS - win_pos))^2/(2*learning_params.sigmat(t)^2));
                    h2wi  = exp(-(idx - (2-win_pos))^2/(2*learning_params.sigmat(t)^2));
                    % learning step: update the input weights /
                    % preferred value of a certain neuron in the population
                    populations(pidx).Winput(idx) = populations(pidx).Winput(idx) + ...
                        learning_params.alphat(t)*(input_sample-populations(pidx).Winput(idx)) + ...
                        alphaf*h2nwi*(mrn - populations(pidx).Winput(idx)) + ...
                        alphaf*h2wi*(mr1 - populations(pidx).Winput(idx));
                else
                    % cooperation step: compute the neighborhood kernel
                    hwi(idx) = exp(-(idx - win_pos)^2/(2*learning_params.sigmat(t)^2));
                    % compute the weight update
                    populations(pidx).Winput(idx) = populations(pidx).Winput(idx) + learning_params.alphat(t)*hwi(idx)*(input_sample - populations(pidx).Winput(idx));
                end
                
                % update the spread of the tuning curve for current neuron
                % populations(pidx).s(idx) = 0.002025;
                populations(pidx).s(idx) = populations(pidx).s(idx) + learning_params.alphat(t)*hwi(idx)*((input_sample - populations(pidx).Winput(idx))^2 - populations(pidx).s(idx));
                %populations(pidx).s(idx) = populations(pidx).s(idx) + learning_params.alphat(t)*(0.5*(input_sample - populations(pidx).Winput(win_pos))^2 - populations(pidx).s(idx));
            end
        end % end for population loop
        
        % visualization
        if(DYN_VISUAL==1)
            visualize_runtime(sensory_data, populations, didx, learning_params);
        end
        
    end % end for samples in training dataset
    
    %         % test sequence
    %         % sample population coded input
    %         populations(1).a = population_encoder(sensory_data.x(didx), max(sensory_data.x(:)),  N_NEURONS);
    %         populations(2).a = population_encoder(sensory_data.y(didx), max(sensory_data.y(:)),  N_NEURONS);
    %         populations(1).a = populations(1).a./sum(populations(1).a);
    %         populations(2).a = populations(2).a./sum(populations(2).a);
    %         % perform cross-modal Hebbian learning
    %         populations(1).Wcross = (1-XI)*populations(1).Wcross + XI*populations(1).a*populations(2).a';
    %         populations(2).Wcross = (1-XI)*populations(2).Wcross + XI*populations(2).a*populations(1).a';
    
    % perform cross-modal Hebbian learning
    populations(1).Wcross = (1-XI)*populations(1).Wcross + XI*populations(1).a*populations(2).a';
    populations(2).Wcross = (1-XI)*populations(2).Wcross + XI*populations(2).a*populations(1).a';
    
end % end for training epochs
% normalize weights between [0,1]
populations(1).Wcross = populations(1).Wcross ./ max(populations(1).Wcross(:));
populations(2).Wcross = populations(2).Wcross ./ max(populations(2).Wcross(:));
% visualize post-simulation weight matrices encoding learned relation
visualize_runtime(sensory_data, populations, length(sensory_data.x), learning_params);