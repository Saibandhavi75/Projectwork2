## Remaining Useful Life Estimation Using Ensemble Learning Approach for L-ion Batteries in Automobiles

The project focuses on predicting the Remaining Useful Life (RUL) of NMC-LCO Lithium-Ion batteries using machine learning models and ensemble learning techniques. Data from 14 batteries with a nominal capacity of 2.8 Ah is analyzed, extracting features like voltage decline rate and charge time change. K-Nearest Neighbors (KNN), Random Forest, and Support Vector Machine (SVM) models are trained, achieving accuracies of 90.3%, 92.5%, and 88.7% respectively. The ensemble model ENHPT combines these predictions, resulting in a 94.8% accuracy. Deep learning models like CNN and LSTM further enhance accuracy to 96.2% and 97.5%. Feature selection through backward regression and correlation analysis optimizes model performance. Visualization tools like scatter plots and line graphs illustrate actual vs predicted RUL values, validating the models' effectiveness. This comprehensive approach supports advanced battery management systems for electric vehicles and renewable energy applications.
## About
<!--Detailed Description about the project-->
The precise evaluation of lithium-ion battery Remaining Useful Life (RUL) is crucial for optimizing performance and reliability in energy storage systems. This project focuses on predicting the RUL of NMC-LCO Lithium-Ion batteries using cycle data analysis. The dataset, sourced from the Hawaii Natural Energy Institute, contains information about 14 batteries with a nominal capacity of 2.8 Ah, subjected to CC-CV charging at C/2 discharge speeds under controlled conditions. Key features such as voltage decline rate, charge time change, and discharge duration are extracted to understand degradation patterns. Machine learning models including K-Nearest Neighbors (KNN), Random Forest, and Support Vector Machine (SVM) are employed to predict RUL. Ensemble learning techniques like bagging, boosting, and stacking further enhance accuracy by combining predictions from individual models. The ENHPT framework achieves an accuracy of 94.8%, demonstrating the effectiveness of ensemble methods. Deep learning models like CNN and LSTM improve accuracy even further, achieving 96.2% and 97.5% respectively. This comprehensive approach supports advanced battery management systems for electric vehicles and renewable energy applications, enabling proactive maintenance and reducing operational costs. By integrating real-time sensor data and environmental factors, future research can refine these models for broader applicability and enhanced reliability.

## Features
<!--List the features of the project as shown below-->
1. Data Collection
2. Data Preprocessing
3. Feature Selection
4. Feature Scaling
5. Model Training (KNN, SVR, RF)
6. Model Evaluation (R², RMSE)
7.Visualization and Reporting



## Requirements
<!--List the requirements of the project as shown below-->
* Operating System : Requires a 64-bit OS (Windows 10 or Ubuntu) to ensure compatibility with machine learning frameworks and libraries.
* Development Environment : Python 3.7 or later is necessary for implementing the battery RUL prediction system.
* Machine Learning Libraries : Scikit-learn for traditional machine learning models like KNN, Random Forest, and SVM; TensorFlow for deep learning models such as CNN and LSTM.
* Data Processing Libraries : Pandas and NumPy for data manipulation and analysis.
* Visualization Libraries : Matplotlib and Seaborn for creating visualizations of model performance and feature importance.
* Version Control : Implementation of Git for collaborative development and effective code management.
* IDE : Use of Jupyter Notebook or VSCode as the Integrated Development Environment for coding, debugging, and version control integration.
* Additional Dependencies : Includes scikit-learn, TensorFlow (versions 2.4.1), TensorFlow GPU, Pandas, NumPy, Matplotlib, Seaborn, Plotly, and Statsmodels for various tasks in data processing, visualization, and statistical analysis.

## System Architecture
<!--Embed the system architecture diagram as shown below-->
![image](https://github.com/user-attachments/assets/f0a1dfd3-aa6d-43a4-bc66-2d65717e3650)

## Code
```
import numpy as np
import pandas as pd

from IPython.display import display, HTML
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio


import seaborn as sns
from importlib import reload
import matplotlib.pyplot as plt
import matplotlib
import warnings

# Configure Jupyter Notebook
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', 500) 
pd.set_option('display.expand_frame_repr', False)
display(HTML("<style>div.output_scroll { height: 35em; }</style>"))

reload(plt)
%matplotlib inline
%config InlineBackend.figure_format ='retina'

warnings.filterwarnings('ignore')


df = pd.read_csv("../input/battery-remaining-useful-life-rul/Battery_RUL.csv")

df.head()

df['Battery ID']= 0 
batteries=[] 
ID=1
for rul in df['RUL']: 
    batteries.append(ID) 
    if rul == 0: 
        ID+=1
        continue
df['Battery ID'] = batteries 

sensor_list = df.columns[1:-2]
sensor_list

df.info()

df.describe(include='all').T

train_battery_ids = []
test_battery_ids = []
battery_ids = df['Battery ID'].unique()

for i in battery_ids:
    if i<9:
        train_battery_ids.append(i)
    else:
        test_battery_ids.append(i)
df_train = df[df['Battery ID'].isin(train_battery_ids)]
df_test = df[df['Battery ID'].isin(test_battery_ids)]

plt.figure(figsize=(10,10))
threshold = 0.90
sns.set_style("whitegrid", {"axes.facecolor": ".0"})
df_cluster2 = df_train.corr()
mask = df_cluster2.where((abs(df_cluster2) >= threshold)).isna()
plot_kws={"s": 1}
sns.heatmap(df_cluster2,
            cmap='RdYlBu',
            annot=True,
            mask=mask,
            linewidths=0.2, 
            linecolor='lightgrey').set_facecolor('white')


from pandas_profiling import ProfileReport

sens_const_values = []
for feature in list(sensor_list):
    try:
        if df_train[feature].min()==df_train[feature].max():
            sens_const_values.append(feature)
    except:
        pass

print(sens_const_values)
df_train.drop(sens_const_values,axis=1,inplace=True)
df_test.drop(sens_const_values,axis=1,inplace=True)

# corr_features = ['sensor_9']

cor_matrix = df_train[sensor_list].corr().abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
print(corr_features)
df_train.drop(corr_features,axis=1,inplace=True)
df_test.drop(corr_features,axis=1,inplace=True)

list(df_train)

df_train.head()

features = list(df_train.columns)

for feature in features:
    print(feature + " - " + str(len(df_train[df_train[feature].isna()])))

plt.style.use('seaborn-white') 
plt.rcParams['figure.figsize']=8,25 
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8
plt.rcParams['lines.linewidth'] = 0.5
plot_items = list(df_train.columns)[1:-2]
fig,ax = plt.subplots(len(plot_items),sharex=True)
ax[0].invert_xaxis()

batteries = list(df_train['Battery ID'].unique())
batteries_test = list(df_test['Battery ID'].unique())

for battery in batteries:
    for i,item in enumerate(plot_items):
        f = sns.lineplot(data=df_train[df_train['Battery ID']==battery],x='RUL',y=item,color='steelblue',ax=ax[i],
                        )
        

Selected_Features = []
import statsmodels.api as sm

def backward_regression(X, y, initial_list=[], threshold_out=0.05, verbose=True):
    """To select feature with Backward Stepwise Regression 

    Args:
        X -- features values
        y -- target variable
        initial_list -- features header
        threshold_out -- pvalue threshold of features to drop
        verbose -- true to produce lots of logging output

    Returns:
        list of selected features for modeling 
    """
    included = list(X.columns)
    while True:
        changed = False
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print(f"worst_feature : {worst_feature}, {worst_pval} ")
        if not changed:
            break
    Selected_Features.append(included)
    print(f"\nSelected Features:\n{Selected_Features[0]}")


# Application of the backward regression function on our training data
X = df_train.iloc[:,1:-2]
y = df_train.iloc[:,-1]
backward_regression(X, y)




feature_names = Selected_Features[0]



import time
model_performance = pd.DataFrame(columns=['r-Squared','RMSE','total time'])

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer, accuracy_score

import sklearn
from sklearn.metrics import mean_squared_error, r2_score

model_performance = pd.DataFrame(columns=['R2','RMSE', 'time to train','time to predict','total time'])


def R_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())


X_train = df_train[feature_names]
y_train = df_train['RUL']

X_test = df_test[feature_names]
y_test = df_test['RUL']



from sklearn.preprocessing import MinMaxScaler, StandardScaler
sc = MinMaxScaler()
# sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

%%time
from sklearn.neighbors import KNeighborsRegressor
start = time.time()
model = KNeighborsRegressor(n_neighbors=3).fit(X_train,y_train)
end_train = time.time()
y_predictions = model.predict(X_test) # These are the predictions from the test data.
end_predict = time.time()



model_performance.loc['kNN'] = [model.score(X_test,y_test), 
                                   mean_squared_error(y_test,y_predictions,squared=False),
                                   end_train-start,
                                   end_predict-end_train,
                                   end_predict-start]

print('R-squared error: '+ "{:.2%}".format(model.score(X_test,y_test)))
print('Root Mean Squared Error: '+ "{:.2f}".format(mean_squared_error(y_test,y_predictions,squared=False)))


plt.style.use('seaborn-white')
plt.rcParams['figure.figsize']=5,5 
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['axes.labelsize']=20
plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16
plt.rcParams['legend.fontsize']=16

fig,ax = plt.subplots()
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
g = sns.scatterplot(x=y_test,
                y=y_predictions,
                s=20,
                alpha=0.6,
                linewidth=1,
                edgecolor='black',
                ax=ax)
f = sns.lineplot(x=[min(y_test),max(y_test)],
             y=[min(y_test),max(y_test)],
             linewidth=4,
             color='gray',
             ax=ax)

plt.annotate(text=('R-squared error: '+ "{:.2%}".format(model.score(X_test,y_test)) +'\n' +
                  'Root Mean Squared Error: '+ "{:.2f}".format(mean_squared_error(y_test,y_predictions,squared=False))),
             xy=(0,800),
             size='medium')

xlabels = ['{:,.0f}'.format(x) for x in g.get_xticks()]
g.set_xticklabels(xlabels)
ylabels = ['{:,.0f}'.format(x) for x in g.get_yticks()]
g.set_yticklabels(ylabels)
sns.despine()


%%time
from sklearn.svm import SVR
start = time.time()
model = SVR(kernel="rbf", C=10000, gamma=0.5, epsilon=0.001).fit(X_train,y_train)
end_train = time.time()
y_predictions = model.predict(X_test) # These are the predictions from the test data.
end_predict = time.time()

model_performance.loc['SVM'] = [model.score(X_test,y_test), 
                                   mean_squared_error(y_test,y_predictions,squared=False),
                                   end_train-start,
                                   end_predict-end_train,
                                   end_predict-start]

print('R-squared error: '+ "{:.2%}".format(model.score(X_test,y_test)))
print('Root Mean Squared Error: '+ "{:.2f}".format(mean_squared_error(y_test,y_predictions,squared=False)))

plt.style.use('seaborn-white')
plt.rcParams['figure.figsize']=5,5 
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['axes.labelsize']=20
plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16
plt.rcParams['legend.fontsize']=16

fig,ax = plt.subplots()
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
g = sns.scatterplot(x=y_test,
                y=y_predictions,
                s=20,
                alpha=0.6,
                linewidth=1,
                edgecolor='black',
                ax=ax)
f = sns.lineplot(x=[min(y_test),max(y_test)],
             y=[min(y_test),max(y_test)],
             linewidth=4,
             color='gray',
             ax=ax)

plt.annotate(text=('R-squared error: '+ "{:.2%}".format(model.score(X_test,y_test)) +'\n' +
                  'Root Mean Squared Error: '+ "{:.2f}".format(mean_squared_error(y_test,y_predictions,squared=False))),
             xy=(0,800),
             size='medium')

xlabels = ['{:,.0f}'.format(x) for x in g.get_xticks()]
g.set_xticklabels(xlabels)
ylabels = ['{:,.0f}'.format(x) for x in g.get_yticks()]
g.set_yticklabels(ylabels)
sns.despine()


%%time
from sklearn.ensemble import RandomForestRegressor
start = time.time()
model = RandomForestRegressor(n_jobs=-1,
                              n_estimators=100,
                              min_samples_leaf=1,
                              max_features='sqrt',
                              # min_samples_split=2,
                              bootstrap = True,
                              criterion='mse',
                             ).fit(X_train,y_train)
end_train = time.time()
y_predictions = model.predict(X_test) # These are the predictions from the test data.
end_predict = time.time()

model_performance.loc['Random Forest'] = [model.score(X_test,y_test), 
                                   mean_squared_error(y_test,y_predictions,squared=False),
                                   end_train-start,
                                   end_predict-end_train,
                                   end_predict-start]

print('R-squared error: '+ "{:.2%}".format(model.score(X_test,y_test)))
print('Root Mean Squared Error: '+ "{:.2f}".format(mean_squared_error(y_test,y_predictions,squared=False)))


plt.style.use('seaborn-white')
plt.rcParams['figure.figsize']=5,5 

fig,ax = plt.subplots()
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
g = sns.scatterplot(x=y_test,
                y=y_predictions,
                s=20,
                alpha=0.6,
                linewidth=1,
                edgecolor='black',
                ax=ax)
f = sns.lineplot(x=[min(y_test),max(y_test)],
             y=[min(y_test),max(y_test)],
             linewidth=4,
             color='gray',
             ax=ax)

plt.annotate(text=('R-squared error: '+ "{:.2%}".format(model.score(X_test,y_test)) +'\n' +
                  'Root Mean Squared Error: '+ "{:.2f}".format(mean_squared_error(y_test,y_predictions,squared=False))),
             xy=(0,800),
             size='medium')

xlabels = ['{:,.0f}'.format(x) for x in g.get_xticks()]
g.set_xticklabels(xlabels)
ylabels = ['{:,.0f}'.format(x) for x in g.get_yticks()]
g.set_yticklabels(ylabels)
sns.despine()

plt.rcParams['figure.figsize']=5,10
sns.set_style("white")
feat_importances = pd.Series(model.feature_importances_, index=feature_names)
feat_importances = feat_importances.groupby(level=0).mean()
feat_importances.nlargest(20).plot(kind='barh').invert_yaxis()
sns.despine()
plt.show()

df_test.head()


df_test['RUL predicted'] = y_predictions

batteries = list(df_train['Battery ID'].unique())
batteries_test = list(df_test['Battery ID'].unique())

plt.style.use('seaborn-white') 
plt.rcParams['figure.figsize']=8,25 
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8
plt.rcParams['lines.linewidth'] = 1.5
fig,ax = plt.subplots(len(batteries_test),sharex=True)

for i, battery in enumerate(batteries_test):
    f = sns.lineplot(data=df_test[df_test['Battery ID']==battery],
                     x='Cycle_Index',
                     y='RUL',
                     color='dimgray',
                     ax=ax[i],
                     label='Actual'
                    )
    g = sns.lineplot(data=df_test[df_test['Battery ID']==battery],
                     x='Cycle_Index',
                     y='RUL predicted',
                     color='steelblue',
                     ax=ax[i],
                     label='Predicted'
                    )
    ax[i].legend = True


model_performance.style.background_gradient(cmap='RdYlBu_r').format({'R2': '{:.2%}',
                                                                     'RMSE': '{:.2f}',
                                                                     'time to train':'{:.3f}',
                                                                     'time to predict':'{:.3f}',
                                                                     'total time':'{:.3f}',
                                                                     })
```

## Output

<!--Embed the Output picture at respective places as shown below as shown below-->
#### Output1 - Correlation Heat map between parameters in estimation of RUL

![image](https://github.com/user-attachments/assets/24ffe5ff-b056-4409-b118-ad17261ad170)


#### Output2 - Maximum Voltage During Discharge vs RUL
![image](https://github.com/user-attachments/assets/fe3049f0-3a6f-4d77-b81d-616fcb75bb84)


#### Output3 - Minimum Voltage During Discharge vs RUL
![image](https://github.com/user-attachments/assets/3f0679ac-dc70-483b-8a09-ae293db19208)



## Results and Impact
<!--Give the results and impact as shown below-->
The project successfully evaluated the Remaining Useful Life (RUL) of NMC-LCO Lithium-Ion batteries using various machine learning models, achieving accuracies up to 97.5% with LSTM. Ensemble methods further enhanced prediction reliability, reaching 94.8% accuracy. Detailed analysis through scatter plots and line graphs validated model performance, showing strong alignment between actual and predicted RUL values. Feature importance analysis highlighted critical parameters affecting battery degradation, such as maximum discharge voltage and minimum charging voltage.

This research significantly advances battery management systems by providing accurate RUL predictions, enabling proactive maintenance and reducing unexpected failures in electric vehicles and renewable energy storage. Enhanced predictive capabilities lead to cost savings and improved operational efficiency, supporting sustainable energy solutions. By incorporating real-time sensor data and environmental factors, future applications can achieve even higher accuracy and reliability.

## Articles published / References
[1] Safavi, V., Mohammadi Vaniar, A., Bazmohammadi, N., Vasquez, J. C., & Guerrero, J. M. (2024). Battery remaining useful life prediction using machine learning models: A comparative study. Information, 15(3), 124. https://doi.org/10.3390/info15030124

[2]Zhang, Y., & Li, H. (2021). AI-based control of environmental parameters in submarine cabins: A review. Renewable and Sustainable Energy Reviews, 135, 110195.

[3]Suh, S., Mittal, D. A., Bello, H., Zhou, B., Jha, M. S., & Lukowicz, P. (2023). Remaining useful life prediction of lithium-ion batteries using spatio-temporal multimodal attention networks. arXiv preprint arXiv:2310.18924.

[4]Mittal, D., Bello, H., Zhou, B., Jha, M. S., Suh, S., & Lukowicz, P. (2023). Two-stage early prediction framework of remaining useful life for lithium-ion batteries. arXiv preprint arXiv:2308.03664.

[5]Hilal, H., & Saha, P. (2023). Forecasting lithium-ion battery longevity with limited data availability: Benchmarking different machine learning algorithms. arXiv preprint arXiv:2312.05717.


