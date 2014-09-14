% function to plot the learned tuning curves
function present_tuning_curves(pop, sdata)
figure; set(gcf, 'color', 'w');
subplot(2, 1, 1);
neurons_idx = 1:pop.lsize;
plot(neurons_idx, zeros(pop.lsize, 1), 'ok', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k', 'MarkerSize', 5); hold on;
% for each neuron in the current population
for idx = 1:pop.lsize
    % extract the preferred values (wight vector) of each neuron
    v_pref = pop.Winput(idx);
    % compute the tuning curve of the current neuron in the population
    x = -sdata.range:2*1/pop.lsize:sdata.range;
    fx = exp(-(x - v_pref).^2/(2*pop.s(idx)));
    % visualize results
    plot(fx, 'LineWidth', 3); hold all;
end
% adjust axes
axis([0, pop.lsize, 0, 1]); box off;
xlabel('neuron index'); ylabel('learned tuning curves');
subplot(2, 1, 2);
% plot the uniformly distributed profile
plot(neurons_idx, zeros(pop.lsize, 1), 'ok', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k', 'MarkerSize', 5); hold on;
% for each neuron in the current population
for idx = 1:pop.lsize
    % extract the preferred values (wight vector) of each neuron
    v_pref = -sdata.range + (idx-1)*(sdata.range/((pop.lsize)/2));
    % compute the tuning curve of the current neuron in the population
    x = -sdata.range:2*1/pop.lsize:sdata.range;
    fx = exp(-(x-v_pref).^2/(2*pop.s(idx)));
    plot(fx, 'LineWidth', 3); hold all;
end
% adjust axes
axis([0, pop.lsize, 0, 1]); box off;
xlabel('neuron index'); ylabel('uniform tuning curves');
end