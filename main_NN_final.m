

nb_Pin=4;%nb of P dataset in X
nb_Ein=4;%nb of E dataset in X
nb_dSin=3;%nb of S dataset in X
nb_Rin=1;%nb of R dataset in X
nb_INDin=2;%nb of EI dataset in X
nb_in=[nb_Pin nb_Ein nb_dSin nb_Rin];


G=[1 -1 -1 -1]; % P-E-S-R=0 water budget closure matrix
B=[10 0 0 0; 0 10 0 0; 0 0 5 0; 0 0 0 1]; % prescribed covariance matrix for MAP
KPF=eye(4)-B*G'*(G*B*G')^(-1)*G;


%%%% data
%%%%inputs
%%% X=[[P1,P2,P3,P4,E1,E2,E3,E4,S1,S2,S3,R,EI1,EI2],[nbassin*ntimes]]

%%% The target Y include also the stabilization (i.e constraining
%%% intermediate output layer in model)

%%% Yoi=[[Poi,Poi,Poi,Poi,Eoi,Eoi,Eoi,Eoi,Soi,Soi,Soi,Roi,Poi,Eoi,Soi,Roi,0],[nbassin*ntimes]]
%%% Yoi=[[P,P,P,P,E,E,E,E,S,S,S,R,P,E,S,R,0],[nbassin*ntimes]]
%%% with [Poi,Eoi,Soi,Roi]=KPF*[P,E,S,R] see article


%% dividing in two split train and test

Xtr=X(:,rtr);
Xtt=X(:,rtest);

Yoitr=Yoi(:,rtr);
Yoitt=Yoi(:,rtest);

%%%% for estimating normalization parameter
tmpoitr=Yoitr(13:16,:);


%%      net1

nb_neuron_l12=[5 3 3 3 3 4 3 3 3 3 3 3];
nb_neuron_l2=1;
nb_neuron_l3n1=1;
nb_neuron_l3a=8;
nb_neuron_l3a2=[6 6 6 1];
%nb_neuron_l3b=16;
nb_neuron_l4=1;
nb_neuron_l3n2=1;


net1=network;
net1.initFcn = 'initlay';

net1.numInputs=sum(nb_in)+2;
net1.inputs{sum(nb_in)+1}.size = nb_INDin;
net1.inputs{sum(nb_in)+2}.size = 1;
for i=1:sum(nb_in)
net1.inputs{i}.size = 1;
end



net1.numLayers=3*sum(nb_in)+4*3+1;

count=1;
out1=zeros(1,3*sum(nb_in)+4*3);
ED=zeros(4,1);
for i=1:4
   
    if i==1
        st=1;
        ed=3*nb_in(i)+3;
    else
        st=ed+1;
        ed=ed+3*nb_in(i)+3;
    end
 
    id=st:ed;
    
    for j=1:nb_in(i)
out1(id(nb_in(i)+j))=1;
net1.inputConnect(id(j),count)=1;
net1.inputWeights{id(j),count}.learnFcn= 'learngdm';

net1.inputConnect(id(j),net1.numInputs-1)=1;
net1.inputWeights{id(j),net1.numInputs-1}.learnFcn= 'learngdm';


net1.layerConnect(id(j)+nb_in(i),id(j))=1;  
net1.layerWeights{id(j)+nb_in(i),id(j)}.learnFcn = 'learngdm'; 
net1.layers{id(j)}.size=nb_neuron_l12(count);  
count=count+1;
net1.biasConnect(id(j))=1; 
net1.biases{id(j)}.learnFcn='learngdm'; 

net1.layers{id(j)}.transferFcn = 'tansig';  

net1.layers{id(j)}.initFcn = 'initnw'; 


net1.layers{id(j)+nb_in(i)}.size=nb_neuron_l2;  
net1.biasConnect(id(j)+nb_in(i))=1; 
net1.biases{id(j)+nb_in(i)}.learnFcn='learngdm'; 
net1.layers{id(j)+nb_in(i)}.transferFcn = 'purelin';  
net1.layers{id(j)+nb_in(i)}.initFcn = 'initnw'; 
net1.layerConnect(id(j)+2*nb_in(i),id(j)+nb_in(i))=1;   
net1.layerWeights{id(j)+2*nb_in(i),id(j)+nb_in(i)}.learnFcn = 'learngdm';


net1.inputConnect(id(j)+2*nb_in(i),net1.numInputs)=1;%%%% cst for normalization
net1.inputWeights{id(j)+2*nb_in(i),net1.numInputs}.learnFcn= 'learngdm';

net1.layerConnect(ed-2,id(j)+2*nb_in(i))=1;
net1.layerWeights{ed-2,id(j)+2*nb_in(i)}.learnFcn = 'learngdm'; 
net1.layers{id(j)+2*nb_in(i)}.size=nb_neuron_l3n1;  
net1.biasConnect(id(j)+2*nb_in(i))=0;
net1.layers{id(j)+2*nb_in(i)}.transferFcn='purelin';  
net1.layers{id(j)+2*nb_in(i)}.initFcn = 'initnw'; 

end
    
net1.inputConnect(ed-2,net1.numInputs-1)=1;   
net1.inputWeights{ed-2,net1.numInputs-1}.learnFcn= 'learngdm';
net1.layers{ed-2}.size=nb_neuron_l3a2(i);  
net1.biasConnect(ed-2)=1; 
net1.biases{ed-2}.learnFcn='learngdm'; 
if i==4 
   net1.layers{ed-2}.transferFcn = 'purelin';   
else
net1.layers{ed-2}.transferFcn = 'tansig';   
end
net1.layers{ed-2}.initFcn = 'initnw'; 
net1.layerConnect(ed-1,ed-2)=1;   
net1.layerWeights{ed-1,ed-2}.learnFcn = 'learngdm'; 

net1.layers{ed-1}.size=nb_neuron_l4;  
net1.biasConnect(ed-1)=1; 
net1.biases{ed-1}.learnFcn='learngdm'; 
net1.layers{ed-1}.transferFcn = 'purelin';   
net1.layers{ed-1}.initFcn = 'initnw';    

net1.layerConnect(ed,ed-1)=1;   
net1.layerWeights{ed,ed-1}.learnFcn = 'learngdm'; 


net1.inputConnect(ed,net1.numInputs)=1;
net1.inputWeights{ed,net1.numInputs}.learnFcn= 'learngdm';
net1.layers{ed}.size=nb_neuron_l3n2;  
net1.biasConnect(ed)=0;
net1.layers{ed}.transferFcn='purelin';
net1.layers{ed}.initFcn = 'initnw'; 

net1.layerConnect(3*sum(nb_in)+4*3+1,ed)=1;    
net1.layerWeights{3*sum(nb_in)+4*3+1,ed}.learnFcn = 'learngdm'; 
ED(i)=ed;
 
end


net1.layers{3*sum(nb_in)+4*3+1}.size = 5;
net1.layers{3*sum(nb_in)+4*3+1}.transferFcn ='purelin';
net1.layers{3*sum(nb_in)+4*3+1}.initFcn = 'initnw';
net1.outputConnect =[out1 1];

[flag,inputflags,outputflags] = isconfigured(net1);
net1 = init(net1);

%% static weight setting (normalisation cell + KPF)


for i=1:4
   
    if i==1
        st=1;
        ed=3*nb_in(i)+3;
    else
        st=ed+1;
        ed=ed+3*nb_in(i)+3;
    end
 
    id=st:ed;
 
    
    
[tmpn2,ps2]=mapstd(tmpoitr);
    
    for j=1:nb_in(i)

net1.LW{id(j)+2*nb_in(i),id(j)+nb_in(i)}=1/ps2.xstd;
net1.IW{id(j)+2*nb_in(i),net1.numInputs}=-ps2.xmean/ps2.xstd;

net1.layerWeights{id(j)+2*nb_in(i),id(j)+nb_in(i)}.learn=0;
net1.inputWeights{id(j)+2*nb_in(i),net1.numInputs}.learn=0;
    end

net1.LW{ed,ed-1}=ps2.xstd;
net1.IW{ed,net1.numInputs}=ps2.xmean;
net1.inputWeights{ed,net1.numInputs}.learn=0;
net1.layerWeights{ed,ed-1}.learn=0;

end

net1.LW{ED(4)-1,ED(4)-2}=1;
net1.layerWeights{ED(4)-1,ED(4)-2}.learn=0;
net1.b{ED(4)-1}=0;
net1.biases{ED(4)-1}.learn=0;

net1.LW{ED(4)-2,ED(4)-3}=1;
net1.layerWeights{ED(4)-2,ED(4)-3}.learn=0;

net1.IW{ED(4)-2,net1.numInputs-1}=[0 0];
net1.inputWeights{ED(4)-2,net1.numInputs-1}.learn=0;

net1.b{ED(4)-2}=0;
net1.biases{ED(4)-2}.learn=0;


G=[1 -1 -1 -1];
B=[10 0 0 0; 0 10 0 0; 0 0 5 0; 0 0 0 1];

KPF2=[eye(4);[1 -1 -1 -1]];

for i=1:4
net1.LW{3*sum(nb_in)+4*3+1,ED(i)}=KPF2(:,i);
net1.layerWeights{3*sum(nb_in)+4*3+1,ED(i)}.learn=0;
end


net1.biasConnect(3*sum(nb_in)+4*3+1)=0; 




%%

%%  training parameter



net1.performFcn = 'mse';
net1.trainFcn = 'trainlm';
net1.divideFcn='divideind';

net1.divideParam.trainInd=rtr;
net1.divideParam.valInd=rval;
net1.divideParam.testInd=rtest;
net1.plotFcns = {'plotperform'};
net1.trainParam.goal=0;
net1.trainParam.max_fail=6;
net1.trainParam.min_grad=10^(-10);
net1.trainParam.mu=0.001; 
net1.trainParam.mu_dec=0.1;
net1.trainParam.mu_inc=10;
net1.trainParam.mu_max=10^10;
net1.trainParam.epochs=1000;
net1.adaptFcn='adaptwb';
%%

[tttt,psdX]=mapstd(Xtr);
Xtr_n=mapstd('apply',Xtr,psdX);
Xtt_n=mapstd('apply',Xtt,psdX);



EW=[ones(15,1);10;1];   %Output weight for better constraining 0 


net1=train(net1,Xtr_n,Yoitr,{},{},EW);
out_t = sim(net1,Xtt_n);
out_tr = sim(net1,Xtr_n);



