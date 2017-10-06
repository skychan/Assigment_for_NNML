function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
try
    visible_data = sample_bernoulli(visible_data);
    % hid_prob = logistic(rbm_w * visible_data);
    hid_prob = visible_state_to_hidden_probabilities(rbm_w, visible_data);
    first_hid_state = sample_bernoulli(hid_prob);
    positive = configuration_goodness_gradient(visible_data,(first_hid_state));

    % re_visible_state = sample_bernoulli(logistic(rbm_w' * first_hid_state));
    % re_hidden_state = sample_bernoulli(logistic(rbm_w * re_visible_state));
    re_visible_state = sample_bernoulli(hidden_state_to_visible_probabilities(rbm_w,first_hid_state));
    re_hidden_state = (visible_state_to_hidden_probabilities(rbm_w, re_visible_state));
    negative = configuration_goodness_gradient((re_visible_state), (re_hidden_state));

    ret = positive - negative;
catch
    error('not yet implemented');
end
