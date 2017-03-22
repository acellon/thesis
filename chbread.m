function [] = chbread(name, N)
    
    for n = 1:N
        if n < 10
            newname = strcat(name,'0');
        else
            newname = name;
        end
        newname = strcat(newname,num2str(n));
        edfname = strcat(newname, '.edf');
                
        if ~exist(edfname)
            disp(['File ' num2str(n) ' does not exist.'])
            continue
        end
        
        disp(['File ' num2str(n) ' of ' num2str(N)])
        [hdr, rec] = edfread(edfname);
        save(newname, 'hdr', 'rec');
        
    end
    
end