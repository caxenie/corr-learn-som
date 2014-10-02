% function to plot the learned tuning curves and the probability density
% function of the input data
function present_tuning_curves(pop, sdata)
figure; set(gcf, 'color', 'w');
subplot(4,1,1);
% plot the probability distribution of the input data to motivate the
% density of the learned tuning curves, this is the sensory prior p(s)
switch pop.idx
    case 1
        hist(sdata.x, 100); box off;
    case 2
        hist(sdata.y, 100); box off;
end
xlabel(sprintf('input data population %d ', pop.idx)); ylabel('input values distribution');
subplot(4, 1, 2);
neurons_idx = 1:pop.lsize;
plot(neurons_idx, zeros(pop.lsize, 1), 'ok', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k', 'MarkerSize', 5); hold on;
% for each neuron in the current population
for idx = 1:1:pop.lsize
    % extract the preferred values (wight vector) of each neuron
    v_pref = pop.Winput(idx);
    % compute the tuning curve of the current neuron in the population
    x = linspace(sdata.range, -sdata.range, pop.lsize);
    fx = exp(-(x - v_pref).^2/(2*pop.s(idx)^2));
    plot(1:pop.lsize, fx, 'LineWidth', 3); hold all;
    line([idx idx],[0 max(fx)], 'LineWidth', 2, 'LineStyle', '--', 'Color',[.8 .8 .8]);
end
% adjust axes
axis([0, pop.lsize, 0, 1]); box off;
xlabel('neuron index'); ylabel('learned tuning curves');
% the density of the tuning curves (density function) - should increase
% with the increase of the distribution of sensory data (directly proportional with p(s))
% stimuli associated with the peaks of the tuning curves
subplot(4,1,3);
hist(pop.Winput, 100);
xlabel('input values distribution'); ylabel('# of allocated neurons');
% the shape of the tuning curves (shape functions) - should increase with
% values distribution decrease (inverse proportionally with sensory prior p(s))
% measured as the full width at half maximum of the tuning curves
subplot(4,1,4);
plot(pop.s, '.r');
xlabel('input values distribution'); ylabel('width of tuning curves');
end