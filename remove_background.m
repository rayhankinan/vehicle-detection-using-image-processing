% Used only for data cleaning, not for the main application
function Icleaned = remove_background(I, mask, iterations)

% Generate a mask for the background
bw = activecontour(I, mask, iterations, "edge");

% Remove the background
Icleaned = I .* uint8(bw);

end