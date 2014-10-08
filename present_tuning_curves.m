% function to plot the learned tuning curves and the probability density
% function of the input data
function present_tuning_curves(pop, sdata)
figure; set(gcf, 'color', 'w');
subplot(4,1,1);
% plot the probability distribution of the input data to motivate the
% density of the learned tuning curves, this is the sensory prior p(s)
switch pop.idx
    case 1
        hist(sdata.x, 50); box off;
    case 2
        hist(sdata.y, 50); box off;
end
xlabel(sprintf('input data range population %d ', pop.idx)); ylabel('input values distribution');
hndl = subplot(4, 1, 2);
% compute the tuning curve of the current neuron in the population
% the equally spaced mean values 
x = linspace(-sdata.range, sdata.range, pop.lsize);
% for each neuron in the current population compute the receptive field
for idx = 1:pop.lsize
    % extract the preferred values (wight vector) of each neuron
    v_pref = pop.Winput(idx);
    fx = exp(-(x - v_pref).^2/(2*pop.s(idx)^2));
    plot(1:pop.lsize, fx, 'LineWidth', 3); hold all;
end
pop.Winput = sort(pop.Winput); box off;
ax1_pos = get(hndl, 'Position'); set(hndl, 'XTick', []); set(hndl, 'XColor','w');
ax2 = axes('Position',ax1_pos,'XAxisLocation','bottom','Color','none','LineWidth', 3);
set(hndl, 'YTick', []); set(hndl, 'YColor','w');
set(ax2, 'XTick', pop.Winput); set(ax2, 'XTickLabel', []);
set(ax2, 'XLim', [ min(pop.Winput), max(pop.Winput)]);
xlabel('neuron preferred values'); ylabel('learned tuning curves shapes');
% the density of the tuning curves (density function) - should increase
% with the increase of the distribution of sensory data (directly proportional with p(s))
% stimuli associated with the peaks of the tuning curves
subplot(4,1,3);
hist(pop.Winput, 50); box off;
xlabel('input value range'); ylabel('# of allocated neurons');
% the shape of the tuning curves (shape functions) - should increase with
% values distribution decrease (inverse proportionally with sensory prior p(s))
% measured as the full width at half maximum of the tuning curves
subplot(4,1,4);
plot(pop.s, '.r'); box off;
xlabel('neuron index'); ylabel('width of tuning curves');
end