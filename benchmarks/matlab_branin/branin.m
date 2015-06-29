function [value] = branin(varargin)
% Branin test function
%
%    The number of variables n = 2.
%    constraints:
%    -5 <= x <= 10, 0 <= y <= 15
%    three global optima:  (-pi, 12.275), (pi, 2.275), (9.42478, 2.475),
%    where branin = 0.397887"""

% constraints are not tested
tic;
% extract hyperparameters
% everything before --params is an option, everything after is a hyperparameter
[hyperparams, clioptions] = extract_hyperparams(varargin);

x = hyperparams.x;
y = hyperparams.y;

value = (y - (5.1/(4*pi^2)) * x^2 + 5*x/pi - 6)^2;
value = value + (10 * (1 - 1/(8*pi)) * cos(x) + 10);

elapsedTime = toc;
display(sprintf('Result for ParamILS: SAT, %f, 1, %f, %d, x: %f; y: %f; matlab_branin', ...
                abs(elapsedTime), value, -1, x, y));
end

function [hyperparams, options] = extract_hyperparams(cli_string)
    hyperparams_start = find(strcmp(cli_string, '--params'));
    hyperparams = cli_string(hyperparams_start+1:end);
    options = cli_string(1:hyperparams_start-1);
    
    % build hyperparameter struct
    % 1. remove single minus in front of hyperparameters
    for i=1:2:numel(hyperparams)
        display(hyperparams{i}(1));
        assert(hyperparams{i}(1) == '-', 'Expect a minus in front of the parameter key');
        hyperparams{i} = hyperparams{i}(2:end);
    end
    % 2. Make struct
    hyperparams = struct(hyperparams{:});
    % 3. Try to convert everything to numeric
    param_keys = fieldnames(hyperparams);
	for i = 1:numel(param_keys)
        numeric_param = str2double(hyperparams.(param_keys{i}));
        if ~isnan(numeric_param)
            hyperparams.(param_keys{i}) = numeric_param;
        end
	end
    
    % build options struct
    % 2. expect double minus in front of options
    for i=1:2:numel(options)
        display(options{i}(1));
        assert(options{i}(1) == '-' && options{i}(2) == '-', 'Expect a double minus in front of the option key');
        options{i} = options{i}(3:end);
    end
    % 2. Make struct
    options = struct(options{:});
end
