import pathlib

from django.test import TestCase
from unittest import TestCase

from generic_app.submodels.AutoMLPipeline.MLModel import MLModel
from generic_app.submodels.AutoMLPipeline.ModelSetting import ModelSetting


class ml_model_test(TestCase):

    @classmethod
    def setUp(clscls)->None:

        path = pathlib.Path(__file__).parent.absolute().__str__()

        path = path + "/data2/"
        #path = "AutoMLPipeline/Tests/Tests/data2/"
        model_setting = ModelSetting(train_file_x=path + "ref_train_x_final.csv",
                                     train_file_y=path + "ref_train_y_final.csv")
        model_setting.save()

    def testSomething(self):
        print("Testing")