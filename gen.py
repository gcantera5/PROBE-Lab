data = readmatrix('[Insert File Path]');

time     = data(:, 2);
co655 = data(:, 3);
co940 = data(:, 4);
cross655 = data(:, 5);
cross940 = data(:, 6);

%choose 3 intervals of good data 

%% 940 Cross 

isolated = lowpass(cross940, 5, 50); %can edit if the signal looks weird   

%Find peaks the invert signal to get the troughs (DC) 
% adjust MinPeakProminence and MinPeakDistance to not miss any peaks but not get too many 
[Cross940DCfiltered, locsCross940DCfiltered] = findpeaks(-isolated, MinPeakProminence=25, MinPeakDistance=10);

%adjust the indices to get the peaks and troughs from the correct 10 second intervals 
I1CutCross940DCFiltered = Cross940DCfiltered(14:29, :);
I1CutlocsCross940DCfiltered = locsCross940DCfiltered(14:29, :);

I2CutCross940DCFiltered = [Cross940DCfiltered(94:95, :); Cross940DCfiltered(98:106, :)];
I2CutlocsCross940DCfiltered = [locsCross940DCfiltered(94:95, :); locsCross940DCfiltered(98:106, :)];

I3CutCross940DCFiltered = I3Cross940DCfiltered(110:125, :);
I3CutlocsCross940DCfiltered = I3locsCross940DCfiltered(110:125, :);

plot(time, isolated,time(I3CutlocsCross940DCfiltered), -I3CutCross940DCFiltered,"x");

hold on
%repeat the same steps for the peaks 
[Cross940PKfiltered, locsCross940PKfiltered] = findpeaks(isolated, MinPeakProminence=25, MinPeakDistance=10);
[I3Cross940PKfiltered, I3locsCross940PKfiltered] = findpeaks(isolated, MinPeakProminence=45, MinPeakDistance=12);

I1CutCross940PKfiltered = Cross940PKfiltered(14:29, :);
I1CutlocsCross940PKfiltered = locsCross940PKfiltered(14:29, :);

I2CutCross940ACfiltered = [Cross940PKfiltered(95:96, :); Cross940PKfiltered(99:107, :)];
I2CutlocsCross940ACfiltered = [locsCross940PKfiltered(95:96, :); locsCross940PKfiltered(99:107, :)];

I3CutCross940PKfiltered = I3Cross940PKfiltered(110:125, :);
I3CutlocsCross940PKfiltered = I3locsCross940PKfiltered(110:125, :);

plot(time(I3CutlocsCross940PKfiltered), I3CutCross940PKfiltered,"o")

title('Low Pass Filter Applied')
title("940 nm Filtered Data Peaks and Troughs")
xlabel("Time (s)")
ylabel("Filtered Signal")
xlim([12 140])
legend('', 'Troughs', 'Peaks', 'Location', "best")

%% Method 2 - isolate 655 cross

% Repeat the same steps for each channel 

isolated655 = lowpass(cross655, 5, 50);

%figure(4)
[Cross655DCfiltered, locsCross655DCfiltered] = findpeaks(-isolated655, MinPeakProminence=18);
I1CutCross655DCfiltered = Cross655DCfiltered(160:175, :);
I1CutlocsCross655DCfiltered = locsCross655DCfiltered(160:175, :);

I2CutCross655DCfiltered = Cross655DCfiltered(210:227, :);
I2CutlocsCross655DCfiltered = locsCross655DCfiltered(210:227, :);

I3CutCross655DCfiltered = Cross655DCfiltered(261:279, :);
I3CutlocsCross655DCfiltered = locsCross655DCfiltered(261:279, :);

plot(time, isolated655,time(I3CutlocsCross655DCfiltered), -I3CutCross655DCfiltered,"x");

hold on 
[Cross655ACfiltered, locsCross655ACfiltered] = findpeaks(isolated655, MinPeakProminence=18);
I1CutCross655ACfiltered = Cross655ACfiltered(161:176, :);
I1CutlocsCross655ACfiltered = locsCross655ACfiltered(161:176, :);

I2CutCross655ACfiltered = Cross655ACfiltered(211:228, :);
I2CutlocsCross655ACfiltered = locsCross655ACfiltered(211:228, :);

I3CutCross655ACfiltered = Cross655ACfiltered(262:280, :);
I3CutlocsCross655ACfiltered = locsCross655ACfiltered(262:280, :);


plot(time(I3CutlocsCross655ACfiltered), I3CutCross655ACfiltered,"o")
xlim([70 210])
title('Low Pass Filter Applied - 655 nm')
xlabel("Time")
ylabel("Filtered Signal")


%% Co Polarization 

% 940 Co - Method 2  

isolated940Co = lowpass(co940, 5, 20);

[pksAC_940Co,locsAC_940Co] = findpeaks(isolated940Co, MinPeakProminence=40, MinPeakDistance = 8);

I1PK_940Co = pksAC_940Co(139:157); 
I1cutPKlocs940co = locsAC_940Co(139:157);

I2PK_940Co = pksAC_940Co(194:198); % Points 194-200
I2PK_940Co = [I2PK_940Co; pksAC_940Co(200:215)]; % Concatenate points 201-215
I2cutPKlocs940co = [locsAC_940Co(194:198); locsAC_940Co(200:215)];

I3PK_940Co = pksAC_940Co(247:270); 
I3cutPKlocs940co = locsAC_940Co(247:270);

[pksDC_940Co, locsDC_940Co] = findpeaks(-isolated940Co, MinPeakProminence=7, MinPeakDistance=10);

I1DC_940Co = pksDC_940Co(149:167); 
I1cutDClocs940co = locsDC_940Co(149:167);

I2DC_940Co = pksDC_940Co(200:220); 
I2cutDClocs940co = locsDC_940Co(200:220);

I3DC_940Co = pksDC_940Co(249:272); 
I3cutDClocs940co = locsDC_940Co(249:272);

%plots for checking peak finding 

plot(time,isolated940Co,time(I3cutPKlocs940co),I3PK_940Co,"o");
xlabel("Time")
ylabel("Co 940")
title("940 Co Polarized Sample")
axis tight

hold on  
plot(time(I3cutDClocs940co),-I3DC_940Co,"x")
%plot(time, co940)
xlabel("Time (seconds)")
ylabel("Signal")
title("940 Co Polarized Sample")
xlim([70 210])

%% 

%655co - method 2 

isolated655Co = lowpass(co655, 5, 20);

[pksAC_655Co,locsAC_655Co] = findpeaks(isolated655Co, MinPeakProminence=70, MinPeakDistance = 11);

I1PK_655Co = pksAC_655Co(111:127);
I1cutPKlocs655co = locsAC_655Co(111:127);

I2PK_655Co = pksAC_655Co(160:178);
I2cutPKlocs655co = locsAC_655Co(160:178);

I3PK_655Co = pksAC_655Co(208:223);
I3cutPKlocs655co = locsAC_655Co(208:223);

[pksDC_655Co, locsDC_655Co] = findpeaks(-isolated655Co, MinPeakProminence=20, MinPeakDistance=12);

I1DC_655Co = pksDC_655Co(134:150); 
I1cutDClocs655co = locsDC_655Co(134:150);

I2DC_655Co = pksDC_655Co(183:201); 
I2cutDClocs655co = locsDC_655Co(183:201);

I3DC_655Co = pksDC_655Co(233:248); 
I3cutDClocs655co = locsDC_655Co(233:248);

%plots for checking peak finding 

plot(time,isolated655Co,time(I3cutPKlocs655co),I3PK_655Co,"o");
xlabel("Time")
ylabel("Co 655")
title("655 Co Polarized Sample")
axis tight

hold on  
plot(time(I3cutDClocs655co),-I3DC_655Co,"x")
%plot(time, co940)
xlabel("Time (seconds)")
ylabel("Signal")
title("655 Co Polarized Sample")
xlim([70 210])

%% PK to AC
disp('start')
I1AC940Cr = I1CutCross940PKfiltered + I1CutCross940DCFiltered;
I2AC940Cr = I2CutCross940ACfiltered + I2CutCross940DCFiltered;
I3AC940Cr = I3CutCross940PKfiltered + I3CutCross940DCFiltered;

I1AC655Cr = I1CutCross655ACfiltered + I1CutCross655DCfiltered;
I2AC655Cr = I2CutCross655ACfiltered + I2CutCross655DCfiltered;
I3AC655Cr = I3CutCross655ACfiltered + I3CutCross655DCfiltered;

I1AC940Co = I1PK_940Co + I1DC_940Co;
I2AC940Co = I2PK_940Co + I2DC_940Co;
I3AC940Co = I3PK_940Co + I3DC_940Co;

I1AC655Co = I1PK_655Co + I1DC_655Co;
I2AC655Co = I2PK_655Co + I2DC_655Co;
I3AC655Co = I3PK_655Co + I3DC_655Co;

%% Finding Perfusion Index

I1PI940Cr = -(I1AC940Cr./I1CutCross940DCFiltered)*(100);
I2PI940Cr = -(I2AC940Cr./I2CutCross940DCFiltered)*(100);
I3PI940Cr = -(I3AC940Cr./I3CutCross940DCFiltered)*(100);

I1PI655Cr = -(I1AC655Cr./I1CutCross655DCfiltered)*(100);
I2PI655Cr = -(I2AC655Cr./I2CutCross655DCfiltered)*(100);
I3PI655Cr = -(I3AC655Cr./I3CutCross655DCfiltered)*(100);

I1PI940Co = -(I1AC940Co./I1DC_940Co)*(100);
I2PI940Co = -(I2AC940Co./I2DC_940Co)*(100);
I3PI940Co = -(I3AC940Co./I3DC_940Co)*(100);

I1PI655Co = -(I1AC655Co./I1DC_655Co)*(100);
I2PI655Co = -(I2AC655Co./I2DC_655Co)*(100);
I3PI655Co = -(I3AC655Co./I3DC_655Co)*(100);

%3 intervals: 105-115, 143-153, 180-190  (time)
% First subplot for the interval 63-73
subplot(3,1,1)
stairs(time(I1CutlocsCross940DCfiltered), I1PI940Cr, 'r', 'LineWidth', 1.5)
hold on
stairs(time(I1CutlocsCross655DCfiltered), I1PI655Cr, 'b', 'LineWidth', 1.5)
hold on
stairs(time(I1cutDClocs940co), I1PI940Co, 'm', 'LineWidth', 1.5)
hold on
stairs(time(I1cutDClocs655co), I1PI655Co, 'k', 'LineWidth', 1.5)
xlim([105 115])
title("105-115 seconds")
legend("940 Cross PI", "655 Cross PI", "940 Co PI", '655 Co PI', 'Location', "northeastoutside")
xlabel("Time (seconds)")
ylabel("Perfusion Index")
ylim([0 4])
set(gca, 'FontSize', 18)

% Second subplot for the interval 93-103
subplot(3,1,2)
stairs(time(I2CutlocsCross940DCfiltered), I2PI940Cr, 'r', 'LineWidth', 1.5)
hold on
stairs(time(I2CutlocsCross655DCfiltered), I2PI655Cr, 'b', 'LineWidth', 1.5)
hold on
stairs(time(I2cutDClocs940co), I2PI940Co, 'm', 'LineWidth', 1.5)
hold on
stairs(time(I2cutDClocs655co), I2PI655Co, 'k', 'LineWidth', 1.5)
xlim([143 153])
title("143-153 seconds")
legend("940 Cross PI", "655 Cross PI", "940 Co PI", '655 Co PI', 'Location', "northeastoutside")
xlabel("Time (seconds)")
ylabel("Perfusion Index")
ylim([0 4])
set(gca, 'FontSize', 18)

% Third subplot for the interval 151-161
subplot(3,1,3)
stairs(time(I3CutlocsCross940DCfiltered), I3PI940Cr, 'r', 'LineWidth', 1.5)
hold on
stairs(time(I3CutlocsCross655DCfiltered), I3PI655Cr, 'b', 'LineWidth', 1.5)
hold on
stairs(time(I3cutDClocs940co), I3PI940Co, 'm', 'LineWidth', 1.5)
hold on
stairs(time(I3cutDClocs655co), I3PI655Co, 'k', 'LineWidth', 1.5)
xlim([180 190])
title("180-190 seconds")
legend("940 Cross PI", "655 Cross PI", "940 Co PI", '655 Co PI', 'Location', "northeastoutside")
xlabel("Time (seconds)")
ylabel("Perfusion Index")
ylim([0 4])
set(gca, 'FontSize', 18)
sgtitle('ITA = -5.295', 'FontSize', 22)


%% average PI
tmeanPI940Cr = mean([I1PI940Cr; I2PI940Cr; I3PI940Cr]);
tmeanPI655Cr = mean([I1PI655Cr; I2PI655Cr; I3PI655Cr]);
tmeanPI940Co = mean([I1PI940Co; I2PI940Co; I3PI940Co]);
tmeanPI655Co = mean([I1PI655Co; I2PI655Co; I3PI655Co]);

tstdPI940Cr = std([I1PI940Cr; I2PI940Cr; I3PI940Cr]);
tstdPI655Cr = std([I1PI655Cr; I2PI655Cr; I3PI655Cr]);
tstdPI940Co = std([I1PI940Co; I2PI940Co; I3PI940Co]);
tstdPI655Co = std([I1PI655Co; I2PI655Co; I3PI655Co]);