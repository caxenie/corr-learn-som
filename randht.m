function x=randht(n, varargin)
%    Source: http://www.santafe.edu/~aaronc/powerlaws/
type   = '';
xmin   = 1;
alpha  = 2.5;
beta   = 1;
lambda = 1;
mu     = 1;
sigma  = 1;
persistent rand_state;

% parse command-line parameters; trap for bad input
i=1; 
while i<=length(varargin), 
  argok = 1; 
  if ischar(varargin{i}), 
    switch varargin{i},
        case 'xmin',            xmin = varargin{i+1}; i = i + 1;
        case 'powerlaw',        type = 'PL'; alpha  = varargin{i+1}; i = i + 1;
        case 'cutoff',          type = 'PC'; alpha  = varargin{i+1}; lambda = varargin{i+2}; i = i + 2;
        case 'exponential',     type = 'EX'; lambda = varargin{i+1}; i = i + 1;
        case 'lognormal',       type = 'LN'; mu = varargin{i+1}; sigma = varargin{i+2};i = i + 2;
        case 'stretched',       type = 'ST'; lambda = varargin{i+1}; beta = varargin{i+2}; i = i + 2;
        otherwise, argok=0; 
    end
  end
  if ~argok, 
    disp(['(RANDHT) Ignoring invalid argument #' num2str(i+1)]); 
  end
  i = i+1; 
end
if (~isscalar(n) || n<1)
	fprintf('(RANDHT) Error: invalid ''n'' argument; using default.\n');
    n = 10000;
end;
if (~isscalar(xmin) || xmin<1)
	fprintf('(RANDHT) Error: invalid ''xmin'' argument; using default.\n');
    xmin = 1;
end;
if isempty(rand_state)
    rand_state = cputime;
    rand('twister',sum(100*clock));
end;

switch type
    case 'EX', x = xmin - (1/lambda)*log(1-rand(n,1));
    case 'LN',
        y = exp(mu+sigma*randn(10*n,1));
        while true
            y(y<xmin) = [];
            q = length(y)-n;
            if (q==0), break;
            end;
            if (q>0),
                r = randperm(length(y));
                y(r(1:q)) = [];
                break;
            end;
            if (q<0),
                y = [y; exp(mu+sigma*randn(10*n,1))];
            end;
        end;
        x = y;
    case 'ST', x = (xmin^beta - (1/lambda)*log(1-rand(n,1))).^(1/beta);
    case 'PC',
        x = [];
        y = xmin - (1/lambda)*log(1-rand(10*n,1));
        while true
            y(rand(10*n,1)>=(y./xmin).^(-alpha)) = [];
            x = [x; y];
            q = length(x)-n;
            if (q==0), break;
            end;
            if (q>0),
                r = randperm(length(x));
                x(r(1:q)) = [];
                break;
            end;
            if (q<0),
                y = xmin - (1/lambda)*log(1-rand(10*n,1));
            end;
        end;
    otherwise, x = xmin*(1-rand(n,1)).^(-1/(alpha-1));
end;