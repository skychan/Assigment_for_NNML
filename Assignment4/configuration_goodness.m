function G = configuration_goodness(rbm_w, visible_state, hidden_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% This returns a scalar: the mean over cases of the goodness (negative energy) of the described configurations.
try
    wake = hidden_state' * rbm_w;
    n = size(wake)(1);
    result = zeros(1,n);
    for i = 1:n
        result(i) = wake(i,:) * visible_state(:,i);
    end
    G = mean(result);
catch
    error('not yet implemented');
end
