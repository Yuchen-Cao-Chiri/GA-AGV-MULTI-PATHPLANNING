function ga_TSP  

Map=const_map();%�����ܽ������
CityNum=length(Map(:,1)); %�ܽ������
Worklist=const_worklist();%��������
WorkNum=length(Worklist); %�����ڵ�����
if_connect=const_flag(CityNum);%������ͨbool��
[way,dislist]=tsp(Map,if_connect,Worklist);

inn=10; %��ʼ��Ⱥ��С  
gnmax=80;  %������  
pc=0.8; %�������  
pm=0.8; %�������  

%������ʼ��Ⱥ  
s=zeros(inn,WorkNum);  
for i=1:inn  
    s(i,:)=randperm(WorkNum);  
end  
[~,p]=objf(s,dislist);  
  
gn=1;  
ymean=zeros(gn,1);  
ymax=zeros(gn,1);  
xmax=zeros(inn,WorkNum);  
scnew=zeros(inn,WorkNum);  
smnew=zeros(inn,WorkNum);  
while gn<gnmax+1  
   for j=1:2:inn  
      seln=sel(p);  %ѡ�����  
      scro=cro(s,seln,pc);  %�������  
      scnew(j,:)=scro(1,:);  
      scnew(j+1,:)=scro(2,:);  
      smnew(j,:)=mut(scnew(j,:),pm);  %�������  
      smnew(j+1,:)=mut(scnew(j+1,:),pm);  
   end  
   s=smnew;  %�������µ���Ⱥ  
   [f,p]=objf(s,dislist);  %��������Ⱥ����Ӧ��  
   %��¼��ǰ����ú�ƽ������Ӧ��  
   [fmax,nmax]=max(f);  
   ymean(gn)=1000/mean(f);  
   ymax(gn)=1000/fmax;  
   %��¼��ǰ������Ѹ���  
   x=s(nmax,:);  
   xmax(gn,:)=x;  
   drawTSP(Map,Worklist,way,x,ymax(gn),gn,0);   
   gn=gn+1;  
end  
[min_ymax,index]=min(ymax);  
drawTSP(Map,Worklist,way,xmax(index,:),min_ymax,index,1);  
figure(2);  
plot(ymax,'r'); hold on;  
plot(ymean,'b');grid;  
title('��������');  
legend('���Ž�','ƽ����');  
fprintf('�Ŵ��㷨�õ�����̾���:%.2f\n',min_ymax);  
fprintf('�Ŵ��㷨�õ������·��');  
disp(Worklist(xmax(index,:)));  

end 
function [Worklist]=const_worklist()
Worklist=[1,16,41,7,14,30];
end
function [Map]=const_map()
Map=[0 10;10 15;25 15;35 15;50 15;60 15;75 15;85 15;100 15;5 20;30 20;55 20;80 20;105 20;5 50;30 50;55 50;80 50;105 50;
    12.5 62.5;29 54;41.5 54;57 57;67.5 60;75 60;92.5 60;100 60;7.5 75;57 67;80 70;105 70;30 100;45 95;52.5 90;57 82;60 105;75 105;85 105;
    100 105;80 100;105 100];
end
function [flag]=const_flag(n)
flag=zeros(n,n);
flag(1,2)=1;flag(2,1)=1;flag(1,10)=1;flag(10,1)=1;flag(2,10)=1;flag(10,2)=1;flag(2,3)=1;flag(3,2)=1;
flag(3,4)=1;flag(4,3)=1;flag(3,11)=1;flag(11,3)=1;flag(4,11)=1;flag(11,4)=1;flag(4,5)=1;flag(5,4)=1;
flag(5,6)=1;flag(6,5)=1;flag(5,12)=1;flag(12,5)=1;flag(6,12)=1;flag(12,6)=1;flag(6,7)=1;flag(7,6)=1;
flag(7,8)=1;flag(8,7)=1;flag(7,13)=1;flag(13,7)=1;flag(8,13)=1;flag(13,8)=1;flag(8,9)=1;flag(9,8)=1;
flag(9,14)=1;flag(14,9)=1;flag(10,15)=1;flag(15,10)=1;flag(11,16)=1;flag(16,11)=1;flag(12,17)=1;flag(17,12)=1;
flag(13,18)=1;flag(18,13)=1;flag(14,19)=1;flag(19,14)=1;flag(15,20)=1;flag(20,15)=1;flag(15,28)=1;flag(28,15)=1;
flag(16,20)=1;flag(20,16)=1;flag(16,21)=1;flag(21,16)=1;flag(16,22)=1;flag(22,16)=1;flag(17,22)=1;flag(22,17)=1;
flag(17,29)=1;flag(29,17)=1;flag(17,23)=1;flag(23,17)=1;flag(17,24)=1;flag(24,17)=1;flag(18,24)=1;flag(24,18)=1;
flag(18,25)=1;flag(25,18)=1;flag(18,30)=1;flag(30,18)=1;flag(18,26)=1;flag(26,18)=1;flag(19,26)=1;flag(26,19)=1;
flag(19,27)=1;flag(27,19)=1;flag(19,31)=1;flag(31,19)=1;flag(20,28)=1;flag(28,20)=1;flag(20,21)=1;flag(21,20)=1;
flag(21,28)=1;flag(28,21)=1;flag(21,22)=1;flag(22,21)=1;flag(22,29)=1;flag(29,22)=1;flag(23,29)=1;flag(29,23)=1;
flag(22,23)=1;flag(23,22)=1;flag(23,24)=1;flag(24,23)=1;flag(24,29)=1;flag(29,24)=1;flag(24,30)=1;flag(30,24)=1;
flag(24,25)=1;flag(25,24)=1;flag(25,30)=1;flag(30,25)=1;flag(25,26)=1;flag(26,25)=1;flag(26,27)=1;flag(27,26)=1;
flag(26,30)=1;flag(30,26)=1;flag(26,31)=1;flag(31,26)=1;flag(27,31)=1;flag(31,27)=1;flag(28,32)=1;flag(32,28)=1;
flag(29,35)=1;flag(35,29)=1;flag(30,40)=1;flag(40,30)=1;flag(31,41)=1;flag(41,31)=1;flag(32,33)=1;flag(33,32)=1;
flag(32,36)=1;flag(36,32)=1;flag(33,34)=1;flag(34,33)=1;flag(33,35)=1;flag(35,33)=1;flag(33,36)=1;flag(36,33)=1;
flag(34,35)=1;flag(35,34)=1;flag(34,36)=1;flag(36,34)=1;flag(35,36)=1;flag(36,35)=1;flag(36,37)=1;flag(37,36)=1;
flag(37,38)=1;flag(38,37)=1;flag(37,40)=1;flag(40,37)=1;flag(38,39)=1;flag(39,38)=1;flag(38,40)=1;flag(40,38)=1;
flag(39,41)=1;flag(41,39)=1;
end
%------------------------------------------------  
%����������Ⱥ����Ӧ��  
function [f,p]=objf(s,dislist)  
  
inn=size(s,1);  %��ȡ��Ⱥ��С  
f=zeros(inn,1);  
for i=1:inn  
   f(i)=CalDist(dislist,s(i,:));  %���㺯��ֵ������Ӧ��  
end  
f=1000./f'; %ȡ���뵹��  
  
%���ݸ������Ӧ�ȼ����䱻ѡ��ĸ���  
fsum=0;  
for i=1:inn  
   fsum=fsum+f(i)^15;% ����Ӧ��Խ�õĸ��屻ѡ�����Խ��  
end  
ps=zeros(inn,1);  
for i=1:inn  
   ps(i)=f(i)^15/fsum;  
end  
  
%�����ۻ�����  
p=zeros(inn,1);  
p(1)=ps(1);  
for i=2:inn  
   p(i)=p(i-1)+ps(i);  
end  
p=p';  
end  
  
%--------------------------------------------------  
%���ݱ�������ж��Ƿ����  
function pcc=pro(pc)  
test(1:100)=0;  
l=round(100*pc);  
test(1:l)=1;  
n=round(rand*99)+1;  
pcc=test(n);     
end  
  
%--------------------------------------------------  
%��ѡ�񡱲���  
function seln=sel(p)  
  
seln=zeros(2,1);  
%����Ⱥ��ѡ���������壬��ò�Ҫ����ѡ��ͬһ������  
for i=1:2  
   r=rand;  %����һ�������  
   prand=p-r;  
   j=1;  
   while prand(j)<0  
       j=j+1;  
   end  
   seln(i)=j; %ѡ�и�������  
   if i==2&&j==seln(i-1)    %%����ͬ����ѡһ��  
       r=rand;  %����һ�������  
       prand=p-r;  
       j=1;  
       while prand(j)<0  
           j=j+1;  
       end  
   end  
end  
end  
  
%------------------------------------------------  
%�����桱����  
function scro=cro(s,seln,pc)  
  
bn=size(s,2);  
pcc=pro(pc);  %���ݽ�����ʾ����Ƿ���н��������1���ǣ�0���  
scro(1,:)=s(seln(1),:);  
scro(2,:)=s(seln(2),:);  
if pcc==1  
   c1=round(rand*(bn-2))+1;  %��[1,bn-1]��Χ���������һ������λ  
   c2=round(rand*(bn-2))+1;  
   chb1=min(c1,c2);  
   chb2=max(c1,c2);  
   middle=scro(1,chb1+1:chb2);  
   scro(1,chb1+1:chb2)=scro(2,chb1+1:chb2);  
   scro(2,chb1+1:chb2)=middle;  
   for i=1:chb1  
       while find(scro(1,chb1+1:chb2)==scro(1,i))  
           zhi=find(scro(1,chb1+1:chb2)==scro(1,i));  
           y=scro(2,chb1+zhi);  
           scro(1,i)=y;  
       end  
       while find(scro(2,chb1+1:chb2)==scro(2,i))  
           zhi=find(scro(2,chb1+1:chb2)==scro(2,i));  
           y=scro(1,chb1+zhi);  
           scro(2,i)=y;  
       end  
   end  
   for i=chb2+1:bn  
       while find(scro(1,1:chb2)==scro(1,i))  
           zhi=logical(scro(1,1:chb2)==scro(1,i));  
           y=scro(2,zhi);  
           scro(1,i)=y;  
       end  
       while find(scro(2,1:chb2)==scro(2,i))  
           zhi=logical(scro(2,1:chb2)==scro(2,i));  
           y=scro(1,zhi);  
           scro(2,i)=y;  
       end  
   end  
end  
end  
  
%--------------------------------------------------  
%�����족����  
function snnew=mut(snew,pm)  
  
bn=size(snew,2);  
snnew=snew;  
  
pmm=pro(pm);  %���ݱ�����ʾ����Ƿ���б��������1���ǣ�0���  
if pmm==1  
   c1=round(rand*(bn-2))+1;  %��[1,bn-1]��Χ���������һ������λ  
   c2=round(rand*(bn-2))+1;  
   chb1=min(c1,c2);  
   chb2=max(c1,c2);  
   x=snew(chb1+1:chb2);  
   snnew(chb1+1:chb2)=fliplr(x);  
end  
end  
  
%------------------------------------------------  
%��Ӧ�Ⱥ���  
function F=CalDist(dislist,s)  
  
DistanV=0;  
n=size(s,2);  
for i=1:(n-1)  
    DistanV=DistanV+dislist(s(i),s(i+1));  
end  
DistanV=DistanV+dislist(s(n),s(1));  
F=DistanV;  
  
end  
  
%------------------------------------------------  
%��ͼ  
function drawTSP(cities,worklist,way,solu,bsf,p,f)

z1x=[15,30,45,48,33];
z1y=[75,90,80,64,58];
z2x=[10,25,25,10];
z2y=[20,20,50,50];
z3x=[35,50,50,35];
z3y=[20,20,50,50];
z4x=[60,75,75,60];
z4y=[20,20,50,50];
z5x=[85,100,100,85];
z5y=[20,20,50,50];
z6x=[60,75,75,60];
z6y=[70,70,100,100];
z7x=[85,100,100,85];
z7y=[70,70,100,100];


WorkNum=length(worklist);
%�������е�
cityn=length(cities);
for i=1:cityn  
    plot(cities(i,1),cities(i,2),'o','color','b');
    axis([0 110 0 110]);
    text(cities(i,1),cities(i,2),num2str(i));
    fill(z1x,z1y,'k');
    fill(z2x,z2y,'k');
    fill(z3x,z3y,'k');
    fill(z4x,z4y,'k');
    fill(z5x,z5y,'k');
    fill(z6x,z6y,'k');
    fill(z7x,z7y,'k');
    hold on;  
end  
line=[worklist(solu(1))];
for i=2:length(solu)
    sp=solu(i-1);
    ep=solu(i);
    for j=2:length(way(sp,ep,:))
        if way(sp,ep,j)~=0
            line=[line,way(sp,ep,j),];
        end
    end
end
sp=solu(length(solu));
ep=solu(1);
for j=2:length(way(sp,ep,:))
    if way(sp,ep,j)~=0
         line=[line,way(sp,ep,j),];
    end
end
x=cities(line(1),1);
y=cities(line(1),2);
for i=2:length(line)
    x=[x,cities(line(i),1)];
    y=[y,cities(line(i),2)];
end
plot(x,y,'r-','LineWidth',1);
title([num2str(WorkNum),'�����AGVС��·���滮']); 

if f==0
    text(5,5,['�� ',int2str(p),' ��','  ��̾���Ϊ ',num2str(bsf)]);  
else  
    text(5,5,['���������������̾��� ',num2str(bsf),'�� �ڵ� ',num2str(p),' ���ﵽ']);  
end  

hold off;  
pause(0.05);   
end


%------------------------------------------------  
%tsp�滮
function [path,dlist]=tsp(map,flag,worklist)
cityn=length(map(:,1));
workn=length(worklist);
path=zeros(workn,workn,cityn);
mdlist=1./zeros(cityn,cityn);
dlist=1./zeros(workn,workn);
for i=1:cityn
    for j=i:cityn
        if flag(i,j)==1
            mdlist(i,j)=((map(i,1)-map(j,1))^2+(map(i,2)-map(j,2))^2)^0.5;
            mdlist(j,i)=mdlist(i,j);
        end
    end
end
for i=1:workn
    for j=1:workn
        if i~=j
            [dlist(i,j),path(i,j,:)]=floyd(mdlist,worklist(i),worklist(j));
        end
    end
end
end

%------------------------------------------------  
%floyd�����
%W-�ڽӾ���sp-��ʼ�㣬ep-������
%d-���룬path-·��
function [d,path]=floyd(w,sp,ep)
n=size(w,1);
D=w;
path=zeros(n);
for i=1:n
  for j=1:n
      if D(i,j)~=inf
          path(i,j)=j;
      end
  end
end
%����������D path
for k=1:n
  for i=1:n
     for j=1:n
        if D(i,k)+D(k,j)<D(i,j)
           D(i,j)=D(i,k)+D(k,j);
           path(i,j)=path(i,k);
        end
     end
  end
end
p=[sp];
mp=sp;
for k=1:n
    if mp~=ep
        d=path(mp,ep);
        p=[p,d];
        mp=d;
    end
end
d=D(sp,ep);
path=p;
while length(path)<n
    path=[path,0];
end
end

