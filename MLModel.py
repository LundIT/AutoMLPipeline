import pickle
from io import BytesIO

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import generic_app.models as models
from generic_app.submodels.AutoMLPipeline.ModelSetting import ModelSetting
import pandas as pd
import numpy as np

class MLModel(models.CalculatedModelMixin):

    id = models.AutoField(primary_key=True)
    model_setting = models.ForeignKey(to=ModelSetting, on_delete=models.CASCADE)
    trained_model = models.FileField(max_length=300)
    mean_test_score = models.FloatField()
    model_information = models.TextField(max_length=300)
    mse = models.FloatField()
    rmse = models.FloatField()

    defining_fields = ['model_setting']
    def get_selected_key_list(self, key):
        if key=='model_setting':
            return ModelSetting.objects.all()

    def validate_pipeline(self, X, y, CV):
        """:type X: pd.Dataframe with shape (number_of_validation_instances, number_of_features)"""

        print('Performance')
        print('mean_test_score', list(CV.cv_results_['params'][0].keys())[1:])
        self.mean_test_score = CV.cv_results_['mean_test_score'].mean()
        print_array = [(mean, list(x.values())[1:]) for x, mean in zip(CV.cv_results_['params'], CV.cv_results_['mean_test_score'])]
        self.model_information = '\n'.join([str(x) for x in print_array])
        y_hat = CV.predict(X)
        self.mse = mean_squared_error(y, y_hat)
        print('mse', self.mse)
        self.rmse = mean_squared_error(y, y_hat, squared=False)
        print('rmse', self.rmse)


    def calculate(self):
        ds_train_x = pd.read_csv(self.model_setting.train_file_x)
        ds_train_y = pd.read_csv(self.model_setting.train_file_y)
        ds_train_x = self.update_X(ds_train_x)
        train_x, test_x, train_y, test_y = train_test_split(ds_train_x, ds_train_y, train_size=0.7)
        CV_model = self.train_pipeline(train_x, train_y)
        self.validate_pipeline(test_x, test_y, CV_model)
        #predict_pipeline(ds_test_x)
        self.trained_model.save(f'grid.pickle', content=BytesIO(pickle.dumps(CV_model)))

    def train_pipeline(self, X, y):
        """
        This function trains a RandomForestClassifier with the X to fit the y values
        It saves the prediction of the validation_x data into the file y_hat.csv
        :type X: pd.Dataframe with shape (number_of_instances, number_of_features)
        :type y: pd.DataFrame with shape (number_of_instances, 1)

        :rtype: None
        """

        categorical_features = ['id', 'Sector', 'is_earning_5', 'is_earning_22']
        numeric_features = [x for x in X.columns if x not in categorical_features]
        from sklearn.compose import ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)])
        rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestRegressor())])

        param_grid = [
            {
                'classifier__n_estimators': [1000, 1200, 1400],
                'classifier__max_features': ['auto'],
                'classifier__max_depth': [20, 24, 28, 32],
                'classifier__min_samples_leaf': [40, 50, 60],
                'classifier__min_samples_split': [30, 40],
                'classifier': [RandomForestRegressor(bootstrap=True, oob_score=True)],
                'classifier__n_jobs': [-1],
            }]
        param_grid = [
            {
                'classifier__n_estimators': [1],
                'classifier__max_features': ['auto'],
                'classifier__max_depth': [4],
                'classifier__min_samples_leaf': [3],
                'classifier__min_samples_split': [3],
                'classifier': [RandomForestRegressor(bootstrap=True, oob_score=True)],
                'classifier__n_jobs': [-1],
            }]
        from sklearn.model_selection import GridSearchCV
        print(len(X))
        print((np.arange(0, len(X) * 0.8), np.arange(len(X) * 0.8, len(X) - 1)))
        CV = GridSearchCV(rf, param_grid, n_jobs=20, verbose=10, cv=[(np.arange(0, int(len(X) * 0.8)), np.arange(int(len(X) * 0.8), len(X) - 1)), ])
        CV.fit(X, y.squeeze())
        print(CV.best_params_)
        print(CV.best_score_)
        print(CV.best_estimator_)
        return CV



    def update_X(self, X):
        X["New_Variable_1"] = X['close_vol_adjusted_252'] * X['volume_article__rollingmean__min_periods_None__window_44']
        X["New_Variable_2"] = X['webhose_valence3_single_negative_entity_ent_conf_weighted__rollingmean__min_periods_None__window_22'] * X['close_vol_252']
        X["New_Variable_3"] = X['close_vol_adjusted_22'] * X['nb_sources_esg__misc_neg_listv3__rollingmean__min_periods_None__window_22']
        X["New_Variable_4"] = X['close_vol_adjusted_252'] * X['bruit_esg__env_neg_listv3__rollingmean__min_periods_None__window_44']

        X['lower_close__bollinger__min_periods_None__nb_periods_5'] = X['lower_close__bollinger__min_periods_None__nb_periods_5'] * X['lower_close__bollinger__min_periods_None__nb_periods_44'] * X[
            'lower_close__bollinger__min_periods_None__nb_periods_22'] * X['close__rollingmean__min_periods_None__window_44'] * X['close__rollingmean__min_periods_None__window_5'] * X[
                                                                                   'upper_close__bollinger__min_periods_None__nb_periods_22'] * X['upper_close__bollinger__min_periods_None__nb_periods_44'] * X[
                                                                                   'upper_close__bollinger__min_periods_None__nb_periods_5']
        X['close'] = X['close'] * X['market_cap']
        X['close_adjusted'] = X['close_adjusted'] * X['close_adjusted__rollingmean__min_periods_None__window_22'] * X['lower_close_adjusted__bollinger__min_periods_None__nb_periods_44'] * X[
            'lower_close_adjusted__bollinger__min_periods_None__nb_periods_5']
        X['webhose_valence3_single_negative_entity_mention_weighted__rollingmean__min_periods_None__window_44'] = X['webhose_valence3_single_negative_entity_mention_weighted__rollingmean__min_periods_None__window_44'] * \
                                                                                                                           X[
                                                                                                                               'webhose_valence3_single_negative_entity_ent_conf_weighted__rollingmean__min_periods_None__window_22']
        X['nb_sources_esg__env_neg_listv3__rollingmean__min_periods_None__window_44'] = X['nb_sources_esg__env_neg_listv3__rollingmean__min_periods_None__window_44'] * X[
            'bruit_esg__env_neg_listv3__rollingmean__min_periods_None__window_44']
        X['nb_sources_esg__misc_neg_listv3__rollingmean__min_periods_None__window_22'] = X['nb_sources_esg__misc_neg_listv3__rollingmean__min_periods_None__window_22'] * X[
            'nb_sources_esg__gouv_neg_listv3__rollingmean__min_periods_None__window_22'] * X['nb_sources_esg__env_neg_listv3__rollingmean__min_periods_None__window_44']

        X.drop(
            ['lower_close__bollinger__min_periods_None__nb_periods_44', 'lower_close__bollinger__min_periods_None__nb_periods_22', 'close__rollingmean__min_periods_None__window_44', 'close__rollingmean__min_periods_None__window_5',
             'webhose_valence3_single_negative_entity_ent_conf_weighted__rollingmean__min_periods_None__window_22', 'bruit_esg__env_neg_listv3__rollingmean__min_periods_None__window_44',
             'nb_sources_esg__gouv_neg_listv3__rollingmean__min_periods_None__window_22', 'nb_sources_esg__env_neg_listv3__rollingmean__min_periods_None__window_44',
             'lower_close_adjusted__bollinger__min_periods_None__nb_periods_5', 'lower_close_adjusted__bollinger__min_periods_None__nb_periods_44', 'upper_close__bollinger__min_periods_None__nb_periods_22',
             'upper_close__bollinger__min_periods_None__nb_periods_44', 'upper_close__bollinger__min_periods_None__nb_periods_5', 'market_cap', 'close_adjusted__rollingmean__min_periods_None__window_22'], axis=1, inplace=True)
        return X