% crate the network composed of N_POP populations of
% N_NEURONS neurons implementing a SOM
% and init each struct weight and activity matrices
function populations = create_init_network(N_POP, N_NEURONS, MIN_INIT_RANGE, MAX_INIT_RANGE)
    wcross = rand(N_NEURONS, N_NEURONS)*MAX_INIT_RANGE;
    for pop_idx = 1:N_POP
        populations(pop_idx) = struct(...
            'lsize', N_NEURONS, ...
            'Winput', MIN_INIT_RANGE + rand(N_NEURONS, 1)*(MAX_INIT_RANGE - MIN_INIT_RANGE), ...
            's', ones(N_NEURONS, 1), ...
            'Wcross', wcross./sum(wcross(:)), ...
            'a', zeros(N_NEURONS, 1));
    end
end