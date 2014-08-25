% function to visualize network data at a given iteration in runtime
function visualize_runtime(sensory_data, populations, d)
if(d~=length(sensory_data.x))
    set(gcf, 'color', 'w');
    % activities for each population (both overall activity and homeostasis)
    subplot(3, 3, 1);
    
    acth3 = plot(population_encoder(sensory_data.x(d), sensory_data.min_val, sensory_data.max_val,  populations(1).lsize), '.r', 'LineWidth', 2); box off;
    xlabel('neuron index'); ylabel('pc coded input in layer 1');
    
    subplot(3, 3, 2);
    
    acth4 = plot(population_encoder(sensory_data.y(d), sensory_data.min_val, sensory_data.max_val,  populations(1).lsize), '.b', 'LineWidth', 2); box off;
    xlabel('neuron index'); ylabel('pc coded input in layer 2');
    
    subplot(3, 3, 3);
    acth5 = plot(sensory_data.x(d), sensory_data.y(d), 'ok', 'MarkerEdgeColor', 'k', 'MarkerSize', 10); hold on; plot(sensory_data.x, sensory_data.y, '.g'); box off;
    xlabel('X'); ylabel('Y'); title('Input data');
    
    subplot(3, 3, 4);
    acth6 = plot(populations(1).a, '-r', 'LineWidth', 2); box off;axis([0,  populations(1).lsize, 0, max(populations(1).a)]);
    xlabel('neuron index'); ylabel('activation in layer 1');
    subplot(3, 3, 5);
    acth7 = plot(populations(2).a,  '-b', 'LineWidth', 2); box off;axis([0,  populations(1).lsize, 0,  max(populations(2).a)]);
    xlabel('neuron index'); ylabel('activation in layer 2');
    
    % hebbian links between populations
    hpc1 = subplot(3, 3, 7);
    ax1=get(hpc1,'position'); % Save the position as ax
    set(hpc1,'position',ax1); % Manually setting this holds the position with colorbar
    acth9 = imagesc(populations(1).Wcross); caxis([0, max(populations(1).Wcross(:))]); colorbar;
    box off; grid off;set(gca,'XAxisLocation','top');
    xlabel('layer 1 - neuron index'); ylabel('layer 2 - neuron index'); title('HL weights 1->2');
    hpc2 = subplot(3, 3, 8);
    ax2 =get(hpc2,'position'); % Save the position as ax
    set(hpc2,'position',ax2); % Manually setting this holds the position with colorbar
    acth10= imagesc(populations(2).Wcross); caxis([0, max(populations(2).Wcross(:))]); colorbar;
    box off; grid off;set(gca,'XAxisLocation','top');
    xlabel('layer 2 - neuron index'); ylabel('layer 1 - neuron index'); title('HL weights 2->1');
    
    % refresh visualization
    set(acth3, 'YData', population_encoder(sensory_data.x(d), sensory_data.min_val, sensory_data.max_val,  populations(1).lsize));
    set(acth4, 'YData', population_encoder(sensory_data.y(d), sensory_data.min_val, sensory_data.max_val,  populations(1).lsize));
    set(acth5, 'XData', sensory_data.x(d));
    set(acth5, 'YData', sensory_data.y(d));
    set(acth6, 'YDataSource', 'populations(1).a');
    set(acth7, 'YDataSource','populations(2).a');
    set(acth9, 'CData', populations(1).Wcross);
    set(acth10,'CData', populations(2).Wcross);
    drawnow;
else % post-runtime visulization
    set(gcf, 'color', 'white');
    for pop_idx = 1:length(populations)
        subplot(1, length(populations), pop_idx);
        imagesc(populations(pop_idx).Wcross); caxis([0,max(populations(pop_idx).Wcross(:))]); colorbar; box off;
        set(gca,'XAxisLocation','top');
        title(sprintf('\t Max value of W is %d | Min value of W is %d', max(populations(pop_idx).Wcross(:)), min(populations(pop_idx).Wcross(:))));
    end
end
end