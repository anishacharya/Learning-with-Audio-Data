function [ dataset1] = create()
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
k=1;
for i=1:1:100;
   
     [x,fs]=audioread(strcat(int2str(i),'.au'));
    A=x(:,1);
    N=size(A);
    N1=ceil(0.05*N);
    N2=ceil(0.95*N);
    A1=A(1:N1);
    A2=A(N1+1:N2);
    A3=A(N2+1:N);
    
    %% time domain Analysis:
    %% mean
    x1(k,1)=mean(A1);
    x2(k,1)=mean(A2);
    x3(k,1)=mean(A3);
   
    %% var
    x4(k,1)=var(A1);
    x5(k,1)=var(A2);
    x6(k,1)=var(A3);
    
    %% skewness
    x7(k,1)=skewness(A1);
    x8(k,1)=skewness(A2);
    x9(k,1)=skewness(A3);
    
    %% kurtosis
    x10(k,1)=kurtosis(A1);
    x11(k,1)=kurtosis(A2);
    x12(k,1)=kurtosis(A3);
    
    %% frequency domain Analysis:
    %% psd
     nfft1 = 2^nextpow2(length(A1));
    Pxx1 = abs(fft(A1,nfft1)).^2/length(A1)/fs;
    
     nfft2 = 2^nextpow2(length(A2));
    Pxx2 = abs(fft(A2,nfft2)).^2/length(A2)/fs;
    
     nfft3 = 2^nextpow2(length(A3));
    Pxx3 = abs(fft(A3,nfft3)).^2/length(A3)/fs;
    
    psd1  = dspdata.psd(Pxx1,'fs',fs,'SpectrumType','twosided');
    psd2  = dspdata.psd(Pxx2,'fs',fs,'SpectrumType','twosided');
    psd3  = dspdata.psd(Pxx3,'fs',fs,'SpectrumType','twosided');
    
    x13(k,1)= avgpower(psd1);
    x14(k,1)= avgpower(psd2);
    x15(k,1)= avgpower(psd3);
    
  
    %% Fano Factor
    
        x16(k,1)= x4(k,1)/x1(k,1) ;
        x17(k,1)= x5(k,1)/x2(k,1) ;
        x18(k,1)= x6(k,1)/x3(k,1) ;
        
    %% hyperskewness:
    %5 th order central standardized moment
        x19(k,1)=((1/length(A1))*(sum((A1-mean(A1)).^5)))/(var(A1)^(5/2));
        x20(k,1)=((1/length(A2))*(sum((A2-mean(A2)).^5)))/(var(A2)^(5/2));
        x21(k,1)=((1/length(A3))*(sum((A3-mean(A3)).^5)))/(var(A3)^(5/2));
        
    %% hyperflatness
        x22(k,1)=((1/length(A1))*(sum((A1-mean(A1)).^6)))/(var(A1)^3);
        x23(k,1)=((1/length(A2))*(sum((A2-mean(A2)).^6)))/(var(A2)^3);
        x24(k,1)=((1/length(A3))*(sum((A3-mean(A3)).^6)))/(var(A3)^3);
      
    %% Zero Crossing Rate
        x25(k,1)=feature_zcr(A1);
        x26(k,1)=feature_zcr(A2);
        x27(k,1)=feature_zcr(A3);
        
    %% Spectral Roll-off
        windowFFT=getDFT(A1);
        c=0.15;
        x28(k,1)= feature_spectral_rolloff(windowFFT, c);
        windowFFT=getDFT(A2);
        x29(k,1)= feature_spectral_rolloff(windowFFT, c);
        windowFFT=getDFT(A3);
        x30(k,1)= feature_spectral_rolloff(windowFFT, c);
        
        c=0.5;
        x31(k,1)= feature_spectral_rolloff(windowFFT, c);
        windowFFT=getDFT(A2);
        x32(k,1)= feature_spectral_rolloff(windowFFT, c);
        windowFFT=getDFT(A3);
        x33(k,1)= feature_spectral_rolloff(windowFFT, c);
        
        c=0.85;
        x34(k,1)= feature_spectral_rolloff(windowFFT, c);
        windowFFT=getDFT(A2);
        x35(k,1)= feature_spectral_rolloff(windowFFT, c);
        windowFFT=getDFT(A3);
        x36(k,1)= feature_spectral_rolloff(windowFFT, c);
        
        
     %% entropy
        x37(k,1) = feature_energy_entropy(A1,10);
        x38(k,1) = feature_energy_entropy(A2,10);
        x39(k,1) = feature_energy_entropy(A3,10);
        
%     p1 = hist(A1,length(A1));
%     p2 = hist(A2,length(A2));
%     p3 = hist(A3,length(A3));
%     entropy1 = -sum(p1.*log2(p1));                  %% To Check,@sap,@ip 
%     entropy2 = -sum(p2.*log2(p2));
%     entropy3 = -sum(p3.*log2(p3));
    
    
      %% energy
        x40(k,1)=feature_energy(A1);
        x41(k,1)=feature_energy(A2);
        x42(k,1)=feature_energy(A3);
      
      %% harmonic
        x43(k,1)=feature_harmonic(A1,fs);
        x44(k,1)=feature_harmonic(A2,fs);
        x45(k,1)=feature_harmonic(A3,fs);
        
    %% corrcoef
    
%     x16(k,1)=corrcoef(A1,A2);             %% To Check,@sap,@ip
%     x17(k,1)=corrcoef(A2,A3);
%     x18(k,1)=corrcoef(A3,A1);


    
    
%     plot(p)    
%     figure, plot(x13(k,1))
%     figure, plot(x14(k,1))
%     figure, plot(x15(k,1))
%     scatter(x7,x10);
%     subplot(2,2,1), plot(A)
%     subplot(2,2,2), plot(A1)
%     subplot(2,2,3), plot(A2)
%     subplot(2,2,4), plot(A3)
    
    k=k+1;

 end
dataset1=cat(2,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,x34,x35,x36,x37,x38,x39,x40,x41,x42,x43,x44,x45);
 

end
