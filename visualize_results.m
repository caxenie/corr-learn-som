% function to visualize network data at a given iteration in runtime
function id_maxv = visualize_results(sensory_data, populations)
figure;
set(gcf, 'color', 'white');
% sensory data
subplot(4, 1, 1);
plot(sensory_data.x, sensory_data.y, '.g'); xlabel('X'); ylabel('Y'); box off;
title('Encoded relationship');
% sensory data distribution
subplot(4, 1, 2);
hist(sensory_data.x); hold on; box off;
title('Sensory data distribution'); box off;
% learned realtionship encoded in the Hebbian links
subplot(4, 1, [3 4]);
% extract the max weighht on each row (if multiple the first one)
id_maxv = zeros(populations(1).lsize, 1);
for idx = 1:populations(1).lsize
    [~, id_maxv(idx)] = max(populations(1).Wcross(idx, :));
end
imagesc((populations(1).Wcross)'); caxis([0,max(populations(1).Wcross(:))]); colorbar; box off;
hold on; plot(1:populations(1).lsize, (id_maxv) ,'*r');
xlabel('X'); ylabel('Y');
%set(gca,'XAxisLocation','top');
title(sprintf('\t Max value of W is %d | Min value of W is %d', max(populations(1).Wcross(:)), min(populations(1).Wcross(:))));
end