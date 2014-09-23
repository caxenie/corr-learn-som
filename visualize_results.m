% function to visualize network data at a given iteration in runtime
function visualize_results(sensory_data, populations)
figure;
set(gcf, 'color', 'white');
subplot(4, 1, 1);
plot(sensory_data.x, sensory_data.y, '.g'); xlabel('X'); ylabel('Y'); box off;
title('Encoded relationship');
subplot(4, 1, 2);
hist(sensory_data.x); hold on; box off;
title('Sensory data distribution'); box off;
subplot(4, 1, [3 4]);
imagesc(rot90(populations(2).Wcross')); caxis([0,max(populations(2).Wcross(:))]); colorbar; box off;
xlabel('X'); ylabel('Y');
%set(gca,'XAxisLocation','top');
title(sprintf('\t Max value of W is %d | Min value of W is %d', max(populations(2).Wcross(:)), min(populations(2).Wcross(:))));
end