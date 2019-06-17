function ga_TSP  

Map=const_map();%导入总结点坐标
CityNum=length(Map(:,1)); %总结点总数
Worklist=const_worklist();%工作点编号
WorkNum=length(Worklist); %工作节点总数
if_connect=const_flag(CityNum);%导入连通bool表
[way,dislist]=tsp(Map,if_connect,Worklist);

inn=10; %初始种群大小  
gnmax=80;  %最大代数  
pc=0.8; %交叉概率  
pm=0.8; %变异概率  

%产生初始种群  
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
      seln=sel(p);  %选择操作  
      scro=cro(s,seln,pc);  %交叉操作  
      scnew(j,:)=scro(1,:);  
      scnew(j+1,:)=scro(2,:);  
      smnew(j,:)=mut(scnew(j,:),pm);  %变异操作  
      smnew(j+1,:)=mut(scnew(j+1,:),pm);  
   end  
   s=smnew;  %产生了新的种群  
   [f,p]=objf(s,dislist);  %计算新种群的适应度  
   %记录当前代最好和平均的适应度  
   [fmax,nmax]=max(f);  
   ymean(gn)=1000/mean(f);  
   ymax(gn)=1000/fmax;  
   %记录当前代的最佳个体  
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
title('搜索过程');  
legend('最优解','平均解');  
fprintf('遗传算法得到的最短距离:%.2f\n',min_ymax);  
fprintf('遗传算法得到的最短路线');  
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
%计算所有种群的适应度  
function [f,p]=objf(s,dislist)  
  
inn=size(s,1);  %读取种群大小  
f=zeros(inn,1);  
for i=1:inn  
   f(i)=CalDist(dislist,s(i,:));  %计算函数值，即适应度  
end  
f=1000./f'; %取距离倒数  
  
%根据个体的适应度计算其被选择的概率  
fsum=0;  
for i=1:inn  
   fsum=fsum+f(i)^15;% 让适应度越好的个体被选择概率越高  
end  
ps=zeros(inn,1);  
for i=1:inn  
   ps(i)=f(i)^15/fsum;  
end  
  
%计算累积概率  
p=zeros(inn,1);  
p(1)=ps(1);  
for i=2:inn  
   p(i)=p(i-1)+ps(i);  
end  
p=p';  
end  
  
%--------------------------------------------------  
%根据变异概率判断是否变异  
function pcc=pro(pc)  
test(1:100)=0;  
l=round(100*pc);  
test(1:l)=1;  
n=round(rand*99)+1;  
pcc=test(n);     
end  
  
%--------------------------------------------------  
%“选择”操作  
function seln=sel(p)  
  
seln=zeros(2,1);  
%从种群中选择两个个体，最好不要两次选择同一个个体  
for i=1:2  
   r=rand;  %产生一个随机数  
   prand=p-r;  
   j=1;  
   while prand(j)<0  
       j=j+1;  
   end  
   seln(i)=j; %选中个体的序号  
   if i==2&&j==seln(i-1)    %%若相同就再选一次  
       r=rand;  %产生一个随机数  
       prand=p-r;  
       j=1;  
       while prand(j)<0  
           j=j+1;  
       end  
   end  
end  
end  
  
%------------------------------------------------  
%“交叉”操作  
function scro=cro(s,seln,pc)  
  
bn=size(s,2);  
pcc=pro(pc);  %根据交叉概率决定是否进行交叉操作，1则是，0则否  
scro(1,:)=s(seln(1),:);  
scro(2,:)=s(seln(2),:);  
if pcc==1  
   c1=round(rand*(bn-2))+1;  %在[1,bn-1]范围内随机产生一个交叉位  
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
%“变异”操作  
function snnew=mut(snew,pm)  
  
bn=size(snew,2);  
snnew=snew;  
  
pmm=pro(pm);  %根据变异概率决定是否进行变异操作，1则是，0则否  
if pmm==1  
   c1=round(rand*(bn-2))+1;  %在[1,bn-1]范围内随机产生一个变异位  
   c2=round(rand*(bn-2))+1;  
   chb1=min(c1,c2);  
   chb2=max(c1,c2);  
   x=snew(chb1+1:chb2);  
   snnew(chb1+1:chb2)=fliplr(x);  
end  
end  
  
%------------------------------------------------  
%适应度函数  
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
%画图  
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
%画出所有点
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
title([num2str(WorkNum),'任务点AGV小车路径规划']); 

if f==0
    text(5,5,['第 ',int2str(p),' 代','  最短距离为 ',num2str(bsf)]);  
else  
    text(5,5,['最终搜索结果：最短距离 ',num2str(bsf),'， 在第 ',num2str(p),' 代达到']);  
end  

hold off;  
pause(0.05);   
end


%------------------------------------------------  
%tsp规划
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
%floyd求最短
%W-邻接矩阵，sp-起始点，ep-结束点
%d-距离，path-路径
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
%迭代，更新D path
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

