% Used only for data cleaning, not for the main application
function bw = generate_clean_mask(I, mask, iterations)

% Generate a mask for the background
bw = activecontour(I, mask, iterations, "edge");

end