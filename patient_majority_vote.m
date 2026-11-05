function [yP, predP, scoreP, ids] = patient_majority_vote(ytrue, ypred, score, pid, pos)
%PATIENT_MAJORITY_VOTE Aggregate image predictions to patients.
%   The patient prediction is the majority class of its images (ties go to
%   the class with the higher mean score); the patient score is the mean
%   image score.
[ids, ~, gi] = unique(pid(:));
yP = accumarray(gi, ytrue(:), [], @(v) mode(v));
scoreP = accumarray(gi, score(:), [], @mean);
fracPos = accumarray(gi, double(ypred(:) == pos), [], @mean);
neg = setdiff(unique(ytrue), pos);
predP = repmat(neg, numel(ids), 1);
predP(fracPos > 0.5) = pos;
tie = fracPos == 0.5;
predP(tie & scoreP >= 0.5) = pos;
end
