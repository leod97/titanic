from titanic.get_data import get_train,get_test
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

num_feature=['Age','Fare']
cat_feature=['Pclass','Sex','SibSp','Parch','Embarked']


class preprocessing:

    def __init__(self,df):
        self.df=df
    
    def who(self):
        if df['Age']<16.0:
            return 'child'
        elif df['Sex']=='male':
            return 'male'
        elif df['Sex']=='female':
            return 'female'
    

    def feature_engineering(self):
        df['who']=df.apply(who,axis=1)
        return df

    def pre_preprocessing(self):
        df.drop(['PassengerId','Cabin','Name','Ticket','Sex'],axis=1,inplace=True)
        df.dropna(axis=0, subset=['Embarked'],inplace=True)

    def num_preprocessing(self):
        num_transformer = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', RobustScaler())])

        num_preprocessor = ColumnTransformer([('num_transformer',
                                                        num_transformer,
                                                        num_feature)])
    
    def cat_preprocessing(self):
        cat_ohe_transformer=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                                    ('OHE',OneHotEncoder(sparse=True,handle_unknown='ignore'))
                                        ])

        cat_preprocessor = ColumnTransformer([
                                    ('cat_to_ohe',cat_ohe_transformer,cat_feature)
                                    ],remainder='passthrough')
                            
    def preprocessing(self):
        preprocessor = ColumnTransformer([

                                        ('num_pipeline',num_preprocessor,num_feature),
                                        ('cat_pipeline',cat_preprocessor,cat_feature)

                                        ],remainder='passthrough'
                                        )
    
    def run(self):
        
