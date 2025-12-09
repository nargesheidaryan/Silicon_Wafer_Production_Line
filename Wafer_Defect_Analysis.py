import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from pandas.plotting import scatter_matrix
import seaborn as sns

# Download latest version
path = "C:\\NARGES\\Data_Science\\My-Projects\\Semiconductor\\Data"
file = "semiconductor_quality_control.csv"
df = pd.read_csv(os.path.join(path, file))
df['join_helper'] = df['Join_Status'].apply(lambda x: 1 if x == 'Joining' else 0)
print(df.head())
print(df.info())
print(df.describe())
#----------------------------------------histogram for all features-----------------------------------------
hist_col = ['Tool_Type', 'Chamber_Temperature', 'Gas_Flow_Rate', 'RF_Power', 'Etch_Depth',
       'Rotation_Speed', 'Vacuum_Pressure', 'Stage_Alignment_Error',
       'Vibration_Level', 'UV_Exposure_Intensity', 'Particle_Count', 'Defect',
       'Join_Status']
fig , ax = plt.subplots(3 , 5, figsize=(10,5))
ax = ax.flatten()
for i, column in enumerate(hist_col):
    df[column].hist(ax= ax[i], edgecolor = 'black')
    ax[i].set_title(column)
    ax[i].set_ylabel('Count')
for j in range(len(hist_col), len(ax)):
    fig.delaxes(ax[j])    
plt.tight_layout()
plt.show()
#----------------------------------------split data to test and train-----------------------------------------
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
print(len(train_set), len(test_set))
df_train = train_set.copy()
#----------------------------------------count of joining & non-joining based tool type ----------------------
df_train_NonJoin = df_train[df_train['Join_Status'] == 'Non-Joining']
df_train_Join = df_train[df_train['Join_Status'] == 'Joining']
#print(f'Number of non-joined samples: ', len(df_train_NonJoin))
df_train_NonJoin_group = df_train_NonJoin.groupby('Tool_Type')['Join_Status'].count().reset_index()
#print(df_train_NonJoin_group.head())
df_train_Join_group = df_train_Join.groupby('Tool_Type')['Join_Status'].count().reset_index()
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.bar(df_train_NonJoin_group['Tool_Type'], df_train_NonJoin_group['Join_Status'],
         color = 'purple', edgecolor='gray')
ax1.set_xlabel('Tool Type')
ax1.set_ylabel('Count of non-joining samples')
ax2.bar(df_train_Join_group['Tool_Type'], df_train_Join_group['Join_Status'],
         color = 'purple', edgecolor='gray')
ax2.set_xlabel('Tool Type')
ax2.set_ylabel('Count of joining samples')
plt.show()
#-------------------------------------------scatter for join and defect-----------------------------
plt.scatter(df_train['Defect'], df_train['join_helper'])
plt.xlabel('Defect Status')
plt.ylabel('Joining Status')
plt.xticks([0,1], ['No Defect', 'Defect'])
plt.yticks([0,1], ['Non-Joined', 'Joined'])
plt.title('Correlation between Defect and Joining Status')
plt.show()
#---------------------------------------correlation between features and target ------------------
features = [
    'Chamber_Temperature', 'Gas_Flow_Rate', 'RF_Power', 'Etch_Depth',
    'Rotation_Speed', 'Vacuum_Pressure', 'Stage_Alignment_Error',
    'Vibration_Level', 'UV_Exposure_Intensity', 'join_helper', 'Particle_Count'
]
corr = []
target = 'Defect'
for i in features:
              pearson_coef , p_value = stats.pearsonr (df[target], df[i])
              corr.append([target, i , pearson_coef, p_value])
df_corr = pd.DataFrame(corr, columns=['Target', 'Feature', 'Pearson_coef', 'p_value'])              
print(df_corr)
#-----------------------------------------Scatter Matrix---------------------
attributes = ['Defect', 'Chamber_Temperature', 'RF_Power', 'Rotation_Speed', 'Vibration_Level']
scatter_matrix(df_train[attributes], figsize=(12,8))
plt.show()
#-------------------------------------------Multiple plot for correlation-------------------
fig, axs = plt.subplots(2, 2)
ax1, ax2, ax3, ax4 = axs.flatten()
ax1.scatter(df_train['Defect'], df_train['Chamber_Temperature'], color = 'skyblue')
ax1.set_xlabel('Defect Status')
ax1.set_xticks([0,1])
ax1.set_xticklabels(['No Defect', 'Defect'])
ax1.set_ylabel('Chamber Temperature(°C)')
ax2.scatter(df_train['Defect'], df_train['RF_Power'], color = 'steelblue')
ax2.set_xlabel('Defect Status')
ax2.set_xticks([0,1])
ax2.set_xticklabels(['No Defect', 'Defect'])
ax2.set_ylabel('RF power (w)')
ax3.scatter(df_train['Defect'], df_train['Rotation_Speed'], color = 'powderblue')
ax3.set_xlabel('Defect Status')
ax3.set_xticks([0,1])
ax3.set_xticklabels(['No Defect', 'Defect'])
ax3.set_ylabel('Rotation Speed (rpm)')
ax4.scatter(df_train['Defect'], df_train['Vibration_Level'], color = 'deepskyblue')
ax4.set_xlabel('Defect Status')
ax4.set_xticks([0,1])
ax4.set_xticklabels(['No Defect', 'Defect'])
ax4.set_ylabel('Vibration Level')
plt.show()
#---------------------------------scatter plot for tool type and particle count-------------------
df_train_NonJoin = df_train[df_train['Join_Status'] == 'Non-Joining']
sns.scatterplot(data=df_train_NonJoin, x=df_train_NonJoin['Tool_Type'], y=df_train_NonJoin['Particle_Count'], 
                hue='Defect', palette='Set1')
plt.legend(title='Defect', loc='upper center', bbox_to_anchor=(0.7, 0.2))
plt.show()




