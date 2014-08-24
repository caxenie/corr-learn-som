%% SIMPLE IMPLEMENTATION OF THE UNSUPERVISED LEARNING OF RELATIONS NETWORK
% based on the self organizing-maps-network
%% PREPARE ENVIRONMENT
clear all; clc; close all;
%% INIT SIMULATION
% enables dynamic visualization on network runtime
DYN_VISUAL      = 1;
% verbose in standard output
VERBOSE         = 0;
% number of populations in the network
N_SOM           = 2;
% number of neurons in each population
N_NEURONS       = 200;
% max range value @ init for weights and activities in the population
MAX_INIT_RANGE  = 1;
%% INIT INPUT DATA - RELATION IS EMBEDDED IN THE INPUT DATA PAIRS
% set up the interval of interest
MIN_VAL         = -1.0; 
MAX_VAL         = 1.0;
% setup the number of random input samples to generate
NUM_VALS        = 250;
% generate NUM_VALS random samples in the given interval
sensory_data.x  = MIN_VAL + rand(NUM_VALS, 1)*(MAX_VAL - MIN_VAL);
sensory_data.y  = sensory_data.x.^3;
DATASET_LEN     = length(sensory_data.x);
%% INIT NETWORK DYNAMICS

%% CREATE NETWORK AND INITIALIZE
% create a network of SOMs given the simulation constants
populations = create_init_network(N_SOM, N_NEURONS, MAX_INIT_RANGE);

%% NETWORK SIMULATION LOOP
% % present each entry in the dataset for MAX_EPOCHS epochs to train the net
for didx = 1:DATASET_LEN
    % pick a new sample from the dataset and feed it to the input (noiseless input)
    % population in the network (in this case X -> A -> | <- B <- Y)
    X = population_encoder(sensory_data.x(didx), max(sensory_data.x(:)),  N_NEURONS);
    Y = population_encoder(sensory_data.y(didx), max(sensory_data.y(:)),  N_NEURONS);
    % normalize input such that the activity in all units sums to 1.0
    X = X./sum(X);
    Y = Y./sum(Y);
    % clamp input to neural populations
    populations(1).a = X;
    populations(2).a = Y;
    while(1)
       
    end
end % end of all samples in the training dataset
% visualize post-simulation data
visualize_runtime(sensory_data, populations, 1, t, DATASET_LEN);