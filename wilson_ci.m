function ci = wilson_ci(x, n, alpha)
%WILSON_CI Wilson score interval (percent) for x successes out of n.
if nargin < 3, alpha = 0.05; end
z = norminv(1 - alpha / 2);
p = x / n; d = 1 + z^2 / n;
c = p + z^2 / (2 * n);
h = z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2));
ci = 100 * [(c - h) / d, (c + h) / d];
end
