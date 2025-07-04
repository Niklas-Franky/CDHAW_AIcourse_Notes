import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics

file_path = r"F:\python-training\ai_course_data\telco-churn.csv"
df = pd.read_csv(file_path)

df.drop(columns = 'customerID', inplace = True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(how='any', inplace=True)
enc = OrdinalEncoder()
drop_columns = df.select_dtypes(exclude=['object']).columns
df_dropped = df[drop_columns]
df.drop(columns = drop_columns, inplace = True)
encoded = enc.fit_transform(df)
df_encoded = pd.DataFrame(encoded, columns = df.columns, index = df.index)
df_new = pd.concat([df_dropped, df_encoded], axis=1)
df_shuffled = shuffle(df_new, random_state=42)
X = df_shuffled.iloc[:,0:-1]
y = df_shuffled.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)
xgbc = xgb.XGBClassifier(n_estimators=1000,learning_rate=0.01,use_label_encoder = False,gamma=10,max_depth=4,random_state=42)
xgbc.fit(X_train, y_train)
ypred = xgbc.predict(X_test)
print('XGBoost Classifier report\n', metrics.classification_report(y_test,ypred))
xgbc.save_model('telco-churn_xgb.json')