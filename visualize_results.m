% function to visualize network data at a given iteration in runtime
function id_maxv = visualize_results(sensory_data, populations)
% first pair of variables (x,y)
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
    [~, id_maxv(idx)] = max(populations(1).Wcross1(idx, :));
end
imagesc((populations(1).Wcross1)'); caxis([0,max(populations(1).Wcross1(:))]); colorbar; box off;
hold on; plot(1:populations(1).lsize, (id_maxv) ,'*r');
xlabel('X'); ylabel('Y');
%set(gca,'XAxisLocation','top');
title(sprintf('\t Max value of W is %d | Min value of W is %d', max(populations(1).Wcross1(:)), min(populations(1).Wcross1(:))));

% second pair of variables (x,z)
figure;
set(gcf, 'color', 'white');
% sensory data
subplot(4, 1, 1);
plot(sensory_data.x, sensory_data.z, '.g'); xlabel('X'); ylabel('Z'); box off;
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
    [~, id_maxv(idx)] = max(populations(1).Wcross2(idx, :));
end
imagesc((populations(1).Wcross2)'); caxis([0,max(populations(1).Wcross2(:))]); colorbar; box off;
hold on; plot(1:populations(1).lsize, (id_maxv) ,'*r');
xlabel('X'); ylabel('Z');
%set(gca,'XAxisLocation','top');
title(sprintf('\t Max value of W is %d | Min value of W is %d', max(populations(1).Wcross2(:)), min(populations(1).Wcross2(:))));

% third pair of variables (y,z)
figure;
set(gcf, 'color', 'white');
% sensory data
subplot(4, 1, 1);
plot(sensory_data.y, sensory_data.z, '.g'); xlabel('Y'); ylabel('Z'); box off;
title('Encoded relationship');
% sensory data distribution
subplot(4, 1, 2);
hist(sensory_data.x); hold on; box off;
title('Sensory data distribution'); box off;
% learned realtionship encoded in the Hebbian links
subplot(4, 1, [3 4]);
% extract the max weighht on each row (if multiple the first one)
id_maxv = zeros(populations(2).lsize, 1);
for idx = 1:populations(2).lsize
    [~, id_maxv(idx)] = max(populations(2).Wcross2(idx, :));
end
imagesc((populations(2).Wcross2)'); caxis([0,max(populations(2).Wcross2(:))]); colorbar; box off;
hold on; plot(1:populations(2).lsize, (id_maxv) ,'*r');
xlabel('Y'); ylabel('Z');
%set(gca,'XAxisLocation','top');
title(sprintf('\t Max value of W is %d | Min value of W is %d', max(populations(2).Wcross2(:)), min(populations(2).Wcross2(:))));

end