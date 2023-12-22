
import bentoml
import mlflow
import time
from datetime import datetime


def deploy_model(experiment_name,run_id):
    
    """Deploys an MLflow model to the Bentoml api server.

    Args:
        experiment_name: The experiment name that the run id belongs to.
        run_id: the run id which the model benlongs to.

    Returns:
        None.
    """

    try:
        # Register the model with MLflow.
        model_name = mlflow.search_runs(experiment_names=[experiment_name], filter_string=f"run_id='{run_id}'").iloc[0]["tags.Model"]        
        model_uri=f"runs:/{run_id}/{model_name}"
        registered_model_name = f"{experiment_name}_{model_name}"
        registered_model = mlflow.register_model(model_uri, registered_model_name)

        print(f"Registered Model Name : {registered_model_name}\n")

        # Get the latest version of the model.
        client = mlflow.tracking.MlflowClient()
        time.sleep(10)
        model_version = client.get_latest_versions(registered_model_name)[0].version

        print(f"Registered model version : {model_version}\n")

        # Transition the latest version of the model to the production stage.
        client.transition_model_version_stage(
            name=registered_model_name,
            version=model_version,
            stage="production")

        print(f"{registered_model_name} version {model_version} staged to production.\n")
        
        # Delete the previous Bentoml model if it already exists.
        model_list = bentoml.models.list()
        present_experiment_name=mlflow.get_experiment(mlflow.get_run(run_id).info.experiment_id).name

        matching_label_found = False

        if bentoml.models.list():
            for x in bentoml.models.list():
                # Check if the model has the 'experiment_name' label
                if 'experiment_name' in x.info.labels:
                    # If the label matches the present_experiment_name
                    if x.info.labels['experiment_name'] == present_experiment_name:
                    
                        matching_label_found = True
                        if str(x.tag).split(":")[0] == run_id:
                            print(f"Found an existing model with the same run id {run_id} in the API server.\n")
                        else:
                            print(f"Found an existing model {str(x.tag).split(':')[0]} in the API server for the same experiment name {present_experiment_name}. This will be replaced by the new model {run_id}.\n")
                            bentoml.models.delete(str(x.tag).split(":")[0])
                            print("Existing model deleted.\n")

                            bentoml_model_name = run_id
                            bentoml.mlflow.import_model(
                                bentoml_model_name,
                                f"models:/{registered_model_name}/{model_version}",
                                labels={"experiment_name": present_experiment_name}
                            )

                            print(f"Model with run id {run_id} integrated successfully into the API server.\n")

            # If no matching label was found, import the model
            if not matching_label_found:
                print(f"No model with experiment name '{present_experiment_name}' found. Importing the model as a new one.\n")
                bentoml_model_name = run_id
                bentoml.mlflow.import_model(
                    bentoml_model_name,
                    f"models:/{registered_model_name}/{model_version}",
                    labels={"experiment_name": present_experiment_name}
                )
                print("done")       
        else:
            print("bentoml api server was empty . registering frst model")
            bentoml_model_name=f"{run_id}"
            bentoml.mlflow.import_model(
            bentoml_model_name,
            f"models:/{registered_model_name}/{model_version}",
            labels={"experiment_name":f"{experiment_name}"})

            print(f"Model with run id {run_id} integrated successfuly into the api server .\n")        
        
        my_dict = {
        "File Name": [experiment_name],
        "run_id": [run_id],
        "model_name":[model_name],
        "timestamp":[str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))]
    }

        return my_dict

    except Exception as e:
        print("BentoML Exception:",e)

