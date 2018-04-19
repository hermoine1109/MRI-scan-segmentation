%get the folder with original images and the folder with Ground truth
%images
originalImages = dir( 'original/*.png' );
originalImagesNames = { originalImages.name };
groundTruths = dir( 'groundtruth/*.png' );
groundTruthsNames = { groundTruths.name };
%initialize lists
DSClist=[];
MSDlist=[];
HDlist=[];
recognizedList=[];
%for each image
for i = 1
    %get the file name and segleaf the original
    OIfile = strcat('original/',originalImagesNames{i});
    I = imread(OIfile);
    GTfile = strcat('groundtruth/',groundTruthsNames{i});
    T = imread(GTfile);
    I = segleaf(I);
    %calculate the values of each metric
    iDSC = DSC(I,T);
    iMSD = MSD(I,T);
    iHD = HD(I,T);
    isRecognized = recognized(I,T);
    %display the results
    msg = strcat('the DSC of ',OIfile,' is ');
    disp(msg)
    iDSC
    msg = strcat('the MSD of ',OIfile,' is ');
    disp(msg)
    iMSD
    msg = strcat('the HD of ',OIfile,' is ');
    disp(msg)
    iHD
    msg = strcat(OIfile,' is recognized? ');
    disp(msg)
    isRecognized
    %record the results for future analysis
    DSClist = [DSClist iDSC];
    MSDlist = [MSDlist iMSD];
    HDlist = [HDlist iHD];
    recognizedList = [recognizedList isRecognized];

    
end
%display overall stats
disp('-----------------------------------------------------------')
DSCmean = mean (DSClist)
DSCstd = std(DSClist)

MSDmean = mean (MSDlist)
MSDstd = std(MSDlist)

HDmean = mean (HDlist)
HDstd = std(HDlist)
sum = 0;
for k = 1:10
    sum = sum + recognizedList(k);
    percent = sum / 10;
end
    msg = strcat('the percent of category 1 images recognized is: ');
    disp(msg)
    percent
sum = 0;
for k = 11:20
    sum = sum + recognizedList(k);
    percent = sum / 10;

end
    msg = strcat('the percent of category 2 images recognized is: ');
    disp(msg)
    percent
sum = 0;
for k = 21:30
    sum = sum + recognizedList(k);
    percent = sum / 10;

end
    msg = strcat('the percent of category 3 images recognized is: ');
    disp(msg)
percent