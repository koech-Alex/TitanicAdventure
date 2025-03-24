import joblib  
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train_numeric[['Age', 'Fare', 'FamilySize']])  
joblib.dump(scaler, 'scaler.pkl')
