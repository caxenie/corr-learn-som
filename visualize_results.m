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
hist(sensory_data.x, 50); hold on; box off;
title('Sensory data distribution'); box off;
% learned realtionship encoded in the Hebbian links
subplot(4, 1, [3 4]);
% extract the max weight on each row (if multiple the first one)
id_maxv = zeros(populations(1).lsize, 1);
for idx = 1:populations(1).lsize
    [~, id_maxv(idx)] = max(populations(1).Wcross(idx, :));
end
imagesc((populations(1).Wcross)', [0, 1]); box off; colorbar;
hold on; plot(1:populations(1).lsize, (id_maxv) ,'r', 'LineWidth', 2);
hold on; plot(1:populations(1).lsize, (id_maxv) ,'ok', 'MarkerFaceColor','k');
xlabel('X'); ylabel('Y');
end