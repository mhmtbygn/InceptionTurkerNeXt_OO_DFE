function [p, b, c] = mcnemar_exact(ytrue, predA, predB)
%MCNEMAR_EXACT Exact two-sided McNemar test on paired predictions.
okA = predA(:) == ytrue(:); okB = predB(:) == ytrue(:);
b = sum(okA & ~okB); c = sum(~okA & okB);
if b + c == 0, p = 1; return; end
p = min(1, 2 * binocdf(min(b, c), b + c, 0.5));
end
