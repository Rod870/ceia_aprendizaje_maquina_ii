"""
train.py



DESCRIPCIÓN:
This is a script to train a linear regression model with preprocessed data 
obtained from a feature pipeline, provided through a specific path. This 
location of the preprocessed data, along with the location where the trained 
model should be saved, are the parameters passed to the file.

AUTOR:
    Grupo AdMII: 
        - Federico Otero - fede.e.otero@gmail.com, 
        - Rodrigo Carranza - rodrigocarranza81@gmail.com, 
        - Agustin Menara - menaragustin@gmail.com
        
FECHA: 07/10/2023
"""

# Imports
       
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn import metrics
from sklearn.linear_model import LinearRegression

class ModelTrainingPipeline(object):

    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path

    def read_data(self) -> pd.DataFrame:
        """        
        :return pandas_df: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        
        """
            
        data = pd.read_csv(input_path)

        return data
        

    
    def model_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
     :description: 
      This method aims to train a linear regression model using preprocessed data.
     :parameters: 
          df: Pandas DataFrame : Preprocessed training data
     :return:
          Pandas Dataframe : Coefficients of the trained model
     :rtype:
          pd.DataFrame
        """
        


seed = 28
model = LinearRegression()

# División de dataset de entrenaimento y validación
X = df.drop(columns='Item_Outlet_Sales') #[['Item_Weight', 'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type']] # .drop(columns='Item_Outlet_Sales')
x_train, x_val, y_train, y_val = train_test_split(X, df['Item_Outlet_Sales'], test_size = 0.3, random_state=seed)

# Entrenamiento del modelo
model.fit(x_train,y_train)

# Predicción del modelo ajustado para el conjunto de validación
pred = model.predict(x_val)

# Cálculo de los errores cuadráticos medios y Coeficiente de Determinación (R^2)
mse_train = metrics.mean_squared_error(y_train, model.predict(x_train))
R2_train = model.score(x_train, y_train)
print('Métricas del Modelo:')
print('ENTRENAMIENTO: RMSE: {:.2f} - R2: {:.4f}'.format(mse_train**0.5, R2_train))

mse_val = metrics.mean_squared_error(y_val, pred)
R2_val = model.score(x_val, y_val)
print('VALIDACIÓN: RMSE: {:.2f} - R2: {:.4f}'.format(mse_val**0.5, R2_val))

print('\nCoeficientes del Modelo:')
# Constante del modelo
print('Intersección: {:.2f}'.format(model.intercept_))

# Coeficientes del modelo
coef = pd.DataFrame(x_train.columns, columns=['features'])
coef['Coeficiente Estimados'] = model.coef_
print(coef, '\n')
coef.sort_values(by='Coeficiente Estimados').set_index('features').plot(kind='bar', title='Importancia de las variables', figsize=(12, 6))

plt.show()
        
        return coef

    def model_dump(self, model_trained) -> None:
        """
     :description: 
      This script stores the trained model.
     :parameters: 
          model_trained : 
     :return:
          None
     :rtype:
          None
        
        """

    with open('pickle_model.pkl', 'wb') as model_file:
    pickle.dump(model_trained, model_file)
        
        return None

    def run(self):
    
        df = self.read_data()
        model_trained = self.model_training(df)
        self.model_dump(model_trained)

if __name__ == "__main__":

    ModelTrainingPipeline(input_path = '../data/pre_processed/Preprocessed_BigMart.csv',
                          model_path = '../model/pickle_model.pkl').run()
