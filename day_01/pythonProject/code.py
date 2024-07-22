#!/usr/bin/envpython
#coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pmdarima as pm
from sklearn.linear_model import LinearRegression
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
table=pd.read_excel(r"./data/附件 1.xlsx")
for i in range(2017,2017+5):
#移除table最后一条数据（重复了）
#print(table.iloc[len(table)-1])
table.drop((len(table)-1),inplace=True)
i=str(i)
temp=pd.read_excel(r"./data/附件 1.xlsx",sheet_name=i)
table=pd.concat([table,temp])
table=table.reset_index(drop=True)
table
#补齐时间
table['年'].fillna(method='ffill',inplace=True)
table['月'].fillna(method='ffill',inplace=True)
table['日'].fillna(method='ffill',inplace=True)
table
#数据预处理
time_list=[]
for i in range(len(table)):m,d,h=str(int(table.iloc[i,1])),str(int(table.iloc[i,2])),str(table.iloc[i,3])
if(int(table.iloc[i,1])<10):m="0"+str(int(table.iloc[i,1]))
if(int(table.iloc[i,2])<10):d="0"+str(int(table.iloc[i,2]))
#print(m,d)
time=str(int(table.iloc[i,0]))+"-"+m+"-"+d+"-"+h
#print(time)
time_list.append(time)
temp=pd.DataFrame(time_list,columns=["时刻"])
temp["时刻"]=pd.to_datetime(temp["时刻"])
#temp.to_csv('example3.csv',index=False)
#temp
table1=pd.concat([table,temp],axis=1)
#table
df=table1.iloc[:,[7,4,5,6]]
df.to_csv('example2.csv',index=False)
#将索引转换为日期时间
#df.set_index("时刻",inplace=True)
df
df["时刻"]=pd.to_datetime(df["时刻"])
#将时间序列转换为数值型特征
df1=df.copy()
df1['时刻']=df1['时刻'].apply(lambda x:x.timestamp())
df1
#提取时间、水位、水流量和含沙量的数据
data=df1[pd.notna(df["含沙量(kg/m3)"])]
X=data[['时刻','水位(m)','流量(m3/s)']]
y=data['含沙量(kg/m3)']
y
#建立线性回归模型
model=LinearRegression()
model.fit(X,y)
new_df=df1[pd.isna(df.loc[:,"含沙量(kg/m3)"])]
new_X=new_df.loc[:,['时刻','水位(m)','流量(m3/s)']]
new_df.loc[:,"含沙量(kg/m3)"]=model.predict(new_X)
new_df
#使用fillna方法填充空白部分
table['含沙量(kg/m3)'].fillna(new_df['含沙量(kg/m3)'],inplace=True)
df['含沙量(kg/m3)'].fillna(new_df['含沙量(kg/m3)'],inplace=True)
#table.to_csv('example.csv',index=False)
table
#In[242]:
#计算每年的总水流量和总排沙量
yearly_data=table.groupby(table["年"]).agg({'流量(m3/s)':'sum','含沙量(kg/m3)':'sum'})
#输出近 6 年的年总水流量和年总排沙量
print('近 6 年的年总水流量为：',yearly_data['流量(m3/s)'].sum(),'m³')
print('近 6 年的年总排沙量为：',yearly_data['含沙量(kg/m3)'].sum(),'t')
#In[243]:
#计算水沙通量
df["水沙通量"]=df['含沙量(kg/m3)']*df['流量(m3/s)']
df
#In[14]:
#读取数据
data=pd.read_csv('example2.csv')
#设置日期时间列为索引
data.set_index('时刻',inplace=True)
#创建子图
fig,axes=plt.subplots(nrows=3,ncols=1,figsize=(10,10))
#绘制水位数据
axes[0].plot(data.index,data['水位(m)'],label='WaterLevel',color='blue')
axes[0].set_ylabel('WaterLevel(m)')
axes[0].set_title('WaterLevelOverTime')
#绘制水流量数据
axes[1].plot(data.index,data['流量(m3/s)'],label='FlowRate',color='green')
axes[1].set_ylabel('FlowRate(m^3/s)')
axes[1].set_title('FlowRateOverTime')
#绘制含沙量数据
axes[2].plot(data.index,data['含沙量(kg/m3)'],label='SedimentContent',color='red')
axes[2].set_xlabel('Time')
axes[2].set_ylabel('SedimentContent')
axes[2].set_title('SedimentContentOverTime')
#添加图例
foraxinaxes:ax.legend()
#调整子图布局
plt.tight_layout()
#显示图形
plt.show()
##分析近 6 年水沙通量的突变性、季节性和周期性等特性
###突变性分析
df
#滑动窗口分析
#定义滑动窗口的大小，这里设置为 10
window_size=10
#创建一个空的DataFrame用于存储突变点
change_points=pd.DataFrame(columns=['时刻','水位(m)','流量(m3/s)','含沙量(kg/m3)','水沙通量'])
#进行滑动窗口分析
foriinrange(len(df)-window_size+1):window=df.iloc[i:i+window_size]
#计算窗口内数据的均值和标准差
mean_values=window.iloc[:,[4]].mean()
std_values=window.iloc[:,[4]].std()
#设置阈值，可以根据实际情况调整
threshold=2.8#假设阈值为 2
#检测是否有数据超过阈值，如果有，则认为有突变点
if(window.iloc[:,[4]]-mean_values).abs().max().max()>threshold*std_values.max():cp=pd.DataFrame(window.iloc[-1,:]).T
change_points=pd.concat([change_points,cp])#将突变点添加到结果DataFrame中
#打印突变点
print("突变点:")
print(change_points)
change_points
#创建一个新的Figure
plt.figure(figsize=(12,6))
plt.subplot(411)
plt.boxplot(df['水位(m)'],labels=['waterlevel'],vert=False)
plt.title('waterlevelBoxPlot')
plt.subplot(412)
plt.boxplot(df['流量(m3/s)'],labels=['FlowRate'],vert=False)
plt.title('FlowRateBoxPlot')
plt.subplot(413)
plt.boxplot(df['含沙量(kg/m3)'],labels=['SedimentContent'],vert=False)
plt.title('SedimentContentBoxPlot')
plt.subplot(414)
plt.boxplot(df['水沙通量'],labels=['WaterAndSedimentFlux'],vert=False)
plt.title('WaterAndSedimentFluxBoxPlot')
#显示图形
plt.show()
#创建一个新的Figure
plt.figure(figsize=(12,6))
#可视化水位数据
plt.subplot(311)
plt.plot(df['时刻'],df['水位(m)'],label='waterlevel',color='blue')
plt.xlabel('Time')
plt.ylabel('WaterLevel')
plt.title('WaterLevelOverTime')
#可视化水流量数据
plt.subplot(312)
plt.plot(df['时刻'],df['流量(m3/s)'],label='FlowRate',color='green')
plt.xlabel('Time')
plt.ylabel('FlowRate')
plt.title('FlowRateOverTime')
#可视化含沙量数据
plt.subplot(313)
plt.plot(df['时刻'],df['含沙量(kg/m3)'],label='SedimentContent',color='red')
plt.xlabel('Time')
plt.ylabel('SedimentContent')
plt.title('SedimentContentOverTime')
#在图上标记突变点
forindex,rowinchange_points.iterrows():plt.subplot(311)
plt.axvline(row['时刻'],color='gray',linestyle='--',linewidth=1)
plt.annotate('change',xy=(row['时刻'],df['水位(m)'].max()),xytext=(-20,30), textcoords='offsetpoints',arrowprops=dict(arrowstyle="->",color='gray'))
plt.subplot(312)
plt.axvline(row['时刻'],color='gray',linestyle='--',linewidth=1)
plt.annotate('change',xy=(row['时刻'],df['流量(m3/s)'].max()),xytext=(-20,30), textcoords='offsetpoints',arrowprops=dict(arrowstyle="->",color='gray'))
plt.subplot(313)
plt.axvline(row['时刻'],color='gray',linestyle='--',linewidth=1)
plt.annotate('change',xy=(row['时刻'],df['含沙量(kg/m3)'].max()),xytext=(-20,30), textcoords='offsetpoints',arrowprops=dict(arrowstyle="->",color='gray'))
#调整子图的布局
plt.tight_layout()
#显示图形
plt.show()
#In[250]:
#将索引转换为日期时间
df.set_index("时刻",inplace=True)
df
#In[255]:
#计算每日季节性成分
seasonal_window=12#每年季节性
seasonal=df.rolling(window=seasonal_window,min_periods=1).mean()
#计算趋势
trend=df-seasonal
#可视化分解结果
plt.figure(figsize=(12,8))
plt.subplot(311)
plt.plot(df['水沙通量'],label='Original')
plt.legend(loc='best')
plt.subplot(312)
plt.plot(trend['水沙通量'],label='Trend')
plt.legend(loc='best')
plt.subplot(313)
plt.plot(seasonal['水沙通量'],label='Seasonal')
plt.legend(loc='best')
plt.tight_layout()
#显示图形
plt.show()
#In[251]:
#数据重采样
monthly_df=df.resample('M').mean()
monthly_df
#In[256]:
fromstatsmodels.tsa.arima.modelimportARIMA
#使用auto_arima选择ARIMA模型的参数
model=pm.auto_arima(monthly_df['水沙通量'],seasonal=True,m=12)#m表示季节性周期
#拟合ARIMA模型
model.fit(monthly_df['水沙通量'])
#预测未来时间点的值
forecast_horizon=24#预测未来 2 年的数据
forecast,conf_int=model.predict(n_periods=forecast_horizon,return_conf_int=True)
#构造预测时间索引
forecast_index=pd.date_range(start=monthly_df['水沙通量'].index[-1], periods=forecast_horizon,freq='M')
#可视化原始数据、趋势、季节性和预测结果
plt.figure(figsize=(12,8))
plt.plot(monthly_df['水沙通量'],label='OriginalData',color='blue')
plt.plot(trend['水沙通量'],label='Trend',color='green')
plt.plot(seasonal['水沙通量'],label='Seasonal',color='red')
plt.plot(forecast_index,forecast,label='Forecast',color='purple')
plt.fill_between(forecast_index,conf_int[:,0],conf_int[:,1],color='purple',alpha=0.3)
plt.legend(loc='best')
plt.title('OriginalData,Trend,Seasonal,andForecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
##使用遗传算法来指令未来两年最优的采样监测方案
#In[259]:
from deap import base,creator,tools,algorithms
import random
#定义问题
creator.create("FitnessMin",base.Fitness,weights=(-1.0,-1.0))
creator.create("Individual",list,fitness=creator.FitnessMin)
#定义参数
n_samples=100#采样监测次数
n_days=730#监测天数
#定义适应度函数
defevaluate(individual):cost=sum(individual)
#计算监测成本资源
#计算监测效果
#TODO:根据具体的数学模型进行计算
fitness=0
returnfitness,cost
#In[260]:
#定义遗传算法参数
toolbox=base.Toolbox()
toolbox.register("attr_bool",random.randint,0,1)
toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.attr_bool,n_days)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)
toolbox.register("evaluate",evaluate)
toolbox.register("mate",tools.cxTwoPoint)
toolbox.register("mutate",tools.mutFlipBit,indpb=0.05)
toolbox.register("select",tools.selTournament,tournsize=3)
#运行遗传算法
pop=toolbox.population(n=100)
hof=tools.HallOfFame(1)
stats=tools.Statistics(lambdaind:ind.fitness.values)
stats.register("avg",np.mean)
stats.register("min",np.min)
pop,log=algorithms.eaSimple(pop,toolbox,cxpb=0.5,mutpb=0.2,ngen=50,stats=stats,halloffame=hof)
#输出最优解
best_ind=hof[0]
best_fitness,best_cost=evaluate(best_ind)
print("最优解：",best_ind)
print("最优适应度：",best_fitness)
print("最小成本：",best_cost)
#In[14]:
#根据 8-12&1-5 月数据对 6，7 月水沙通量进行预测
monthly_df
#In[262]:
#将df进行划分成训练集和测试集
start_date=pd.to_datetime('2016-01-01')
end_date=pd.to_datetime('2016-06-01')
sub_df=monthly_df[(monthly_df.index>=start_date)&(monthly_df.index<end_date)]
sub_train_dfs=[]
whilestart_date<monthly_df.index.max():sub_train_dfs.append(sub_df)
start_date=end_date+pd.DateOffset(months=2)
end_date=start_date+pd.DateOffset(months=10)
sub_df=monthly_df[(monthly_df.index>=start_date)&(monthly_df.index<end_date)]
start_date=pd.to_datetime('2016-06-01')
end_date=pd.to_datetime('2016-08-01')
sub_test_dfs=[]
whilestart_date<monthly_df.index.max():end_date=start_date+pd.DateOffset(months=2)
sub_df=monthly_df[(monthly_df.index>=start_date)&(monthly_df.index<end_date)]
sub_test_dfs.append(sub_df)
start_date=end_date+pd.DateOffset(months=10)
sub_test_dfs
#In[263]:
#用于存储预测结果的列表
forecast_results=[]
#遍历train_dfs和test_dfs中的对应DataFrame
fortrain_df,test_dfinzip(sub_train_dfs,sub_test_dfs):model=ARIMA(train_df["水沙通量"],order=(1,1,1))#替换p、d和q为适当的阶数
#从train_df中提取训练数据
model_fit=model.fit()
#从test_df中提取测试数据
forecast_6,forecast_7=model_fit.forecast(steps=2)
#将预测结果存储到列表中
forecast_results.append([forecast_6,forecast_7])
forecast_results=pd.DataFrame(forecast_results,columns=["6 月预测","7 月预测"])
forecast_results.index+=2016
forecast_results
#In[264]:
#将测试集dfs内数据转成一个df
test_results=[]
foriinrange(len(sub_test_dfs)):test_results.append(sub_test_dfs[i]["水沙通量"].values)
test_results=pd.DataFrame(test_results,columns=["6 月预测","7 月预测"])
test_results.index+=2016
test_results
#In[265]:
#计算差异指标
DID=forecast_results-test_results
DID["差异指标"]=(DID["6 月预测"]+DID["7 月预测"])/2
DID["差异指标"]
#In[54]:
#绘制每年的样本内拟合和样本外预测
plt.figure(figsize=(15,10))
#可视化含沙量数据
plt.plot(monthly_df.index,monthly_df['水沙通量'],label='WaterAndSedimentFlux',color='red')
plt.xlabel('Time')
plt.ylabel('WaterAndSedimentFlux')
plt.title('WaterAndSedimentFluxOverTime')
plt.plot([pd.to_datetime('2016-06-01'),pd.to_datetime('2016-07-01')],forecast_results.iloc[0,:],label=' WaterAndSedimentFlux',color='blue')
plt.plot([pd.to_datetime('2017-06-01'),pd.to_datetime('2017-07-01')],forecast_results.iloc[1,:],label=' WaterAndSedimentFlux',color='blue')
plt.plot([pd.to_datetime('2018-06-01'),pd.to_datetime('2018-07-01')],forecast_results.iloc[2,:],label=' WaterAndSedimentFlux',color='blue')
plt.plot([pd.to_datetime('2019-06-01'),pd.to_datetime('2019-07-01')],forecast_results.iloc[3,:],label=' WaterAndSedimentFlux',color='blue')
plt.plot([pd.to_datetime('2020-06-01'),pd.to_datetime('2020-07-01')],forecast_results.iloc[4,:],label=' WaterAndSedimentFlux',color='blue')
plt.plot([pd.to_datetime('2021-06-01'),pd.to_datetime('2021-07-01')],forecast_results.iloc[5,:],label=' WaterAndSedimentFlux',color='blue')
#In[64]:
table2=pd.read_excel(r"./data/附件 2.xlsx")
table2
#In[174]:
#计算每日的高程均值
Elevation=[]
foriinrange(6):Elevation.append([table2.columns[2*i],table2.iloc[1:,2*i+1].mean()])
Elevation=pd.DataFrame(Elevation)
Elevation
#In[56]:
table3=pd.read_excel(r"./data/附件 3.xlsx")
table3
#In[59]:
#填充表 3
table3["水位(m)"].fillna(method='ffill',inplace=True)
table3["水深(m)"].fillna(method='ffill',inplace=True)
table3["日期"].fillna(method='ffill',inplace=True)
table3
#In[163]:
#使用groupby计算同一日的平均河底高程
table3['河底高程']=table3['水位(m)']-table3['水深(m)']
daily_mean_elevation=table3.groupby('日期')['河底高程'].mean()
daily_mean_elevation=pd.DataFrame(daily_mean_elevation,index=pd.to_datetime(daily_mean_elev
ation.index))
daily_mean_elevation=daily_mean_elevation.reset_index()
daily_mean_elevation
#In[269]:
Elevation[0]=pd.to_datetime(Elevation[0])
new_column_names={1:'河底高程', 0:"日期"}
Elevation=Elevation.rename(columns=new_column_names)
Elevation=pd.concat([Elevation,daily_mean_elevation])
Elevation["日期"]=Elevation["日期"].dt.strftime('%Y/%m/%d')
#Elevation=Elevation.drop(pd.to_datetime("1970/01/0100:00"))
Elevation
#In[270]:
Elevation['日期']=pd.to_datetime(Elevation['日期'])
#从日期列中提取年份
Elevation['年份']=Elevation['日期'].dt.year
#使用groupby计算同一年下的均值
yearly_mean=Elevation.groupby('年份')['河底高程'].mean()
yearly_mean
#In[271]:
#使用auto_arima选择ARIMA模型的参数
model=pm.auto_arima(yearly_mean,seasonal=False)
#拟合ARIMA模型
model.fit(yearly_mean)
#预测未来时间点的值
forecast_horizon=10#预测未来 2 年的数据
forecast_y,conf_int=model.predict(n_periods=forecast_horizon,return_conf_int=True)
#构造预测时间索引
forecast_index=pd.date_range(start=f'{yearly_mean.index[-1]}-01-01', periods=forecast_horizon,freq='Y')
yearly_mean=pd.DataFrame(yearly_mean).reset_index()
yearly_mean['年份']=pd.to_datetime(yearly_mean['年份'],format='%Y')
#可视化原始数据、趋势、季节性和预测结果
plt.figure(figsize=(12,8))
plt.ylim(40,60)
plt.plot(yearly_mean["年份"],yearly_mean["河底高程"],label='OriginalData',color='blue')
plt.plot(forecast_index,forecast_y,label='Forecast',color='purple')
plt.show()
#In[285]:
#change_points.to_csv('./results/突变点统计.csv',index=False)
#monthly_df.to_csv('./results/月平均.csv',index=False)
#forecast.to_csv('./results/未来两年水沙通量预测.csv',index=False)
#df.to_csv('./results/水沙通量预测.csv',index=False)
#DID['差异指标'].to_csv('./results/调水调沙差异指标.csv',index=False)
#Elevation.to_csv('./results/平均河底高程.csv',index=False)
#forecast_y.to_csv('./results/未来十年河底高程.csv',index=False)