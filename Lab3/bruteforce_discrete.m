function [max_arg, max_val, calculations] = bruteforce_discrete(arg, f_vals)
    max_arg = arg(1);
    max_val = f_vals(1);
    calculations = 1;
    for i = 2:length(f_vals)
        calculations = calculations + 1;
        if(f_vals(i) > max_val)
            max_val = f_vals(i);
            max_arg = arg(i,:);
        end
    end
end