% function to plot the learned tuning curves and the probability density
% function of the input data
function present_tuning_curves(pop, sdata)
figure; set(gcf, 'color', 'w');
subplot(2,1,1);
% plot the probability distribution of the input data to motivate the
% density of the learned tuning curves
if pop.idx == 1
    hist(sdata.x); box off;
else
    hist(sdata.y); box off;
end
xlabel(sprintf('input data population %d ', pop.idx)); ylabel('values distribution');
subplot(2, 1, 2);
neurons_idx = 1:pop.lsize;
plot(neurons_idx, zeros(pop.lsize, 1), 'ok', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k', 'MarkerSize', 5); hold on;
% for each neuron in the current population
for idx = 1:pop.lsize
    % extract the preferred values (wight vector) of each neuron
    v_pref = pop.Winput(idx);
    % compute the tuning curve of the current neuron in the population
    x = -sdata.range:2*1/pop.lsize:sdata.range;
    fx = exp(-(x - v_pref).^2/(2*pop.s(idx)^2));
    % visualize results
    plot(fx, 'LineWidth', 3); hold all;
end
% adjust axes
axis([0, pop.lsize, 0, 1]); box off;
xlabel('neuron index'); ylabel('learned tuning curves');
end