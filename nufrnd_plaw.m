% function to generate non-uniform distributed values in a given interval
% [vmin, vmax] for a powerlaw function
function y = nufrnd_plaw(x, vmin, vmax, expn)
    if(expn==-1)
            y  = vmin*exp(log(vmax - vmin).*x);
    else
            y  = exp(log(x.*(-vmin^(expn+1) + vmax^(expn+1)) + vmin^(expn+1))/(expn+1));
    end
end
