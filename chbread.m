edffiles = dir('*.edf');

for file = edffiles'

    fprintf(1,'Converting file %s\n',file.name);
    [hdr, rec] = edfread(file.name);
    savename = strsplit(file.name,'.');
    savename = savename{1};
    save(savename, 'hdr', 'rec');
end
