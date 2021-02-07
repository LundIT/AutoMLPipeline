import generic_app.models as models
from ProcessAdminRestApi.models.process_admin_model import DependencyAnalysisMixin


class ModelSetting(DependencyAnalysisMixin):

    id = models.AutoField(primary_key=True)
    train_file_x = models.FileField(max_length=300, upload_to="upload_data")
    train_file_y = models.FileField(max_length=300, upload_to="upload_data")

    def directly_dependent_entries(self):
        from generic_app.submodels.AutoMLPipeline.MLModel import MLModel
        ml_models = MLModel.objects.filter(model_setting=self)
        if ml_models.count() == 0:
            MLModel.create()
        return ml_models

