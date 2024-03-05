import sys
import pandas as pd
from exception import CustomException
from utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,feature):
        try:
            model_path = '/Users/pankajyadav/Desktop/important data/Assignment/mlproject/venv/src/artifact/model.pkl'
            preprocessor_path = '/Users/pankajyadav/Desktop/important data/Assignment/mlproject/venv/src/artifact/preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor =  load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(feature)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e,sys)
            
        

class CustomData:
    def __init__(self,gender:str,race_ethnicity:str,parental_level_education,lunch:str,reading_score:int,writing_score:int,test_preparation:str):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_education = parental_level_education
        self.lunch = lunch
        self.reading_score  = reading_score
        self.writing_score = writing_score
        self.test_preparation = test_preparation
    
    def get_data_as_frame(self):
         try:
            custom_data_input_dict = {
                "gender" : [self.gender],
                "race/ethnicity" :[self.race_ethnicity],
                "parental level of education" :[self.parental_level_education],
                "lunch":[self.lunch],
                "reading score":[self.reading_score],
                "writing score":[self.writing_score],
                "test preparation course":[self.test_preparation]
            }

            return pd.DataFrame(custom_data_input_dict)
         except Exception as e:
            raise CustomException(e,sys)




