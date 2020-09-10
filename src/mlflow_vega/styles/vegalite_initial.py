"""

"""
# Standard Libraries
import importlib
import os
import copy
import json

# External Libraries
import mlflow.pyfunc
from typing import List, Set, Dict, Tuple, Optional, Mapping, IO
from google.protobuf.json_format import MessageToJson, ParseDict
import pandas as pd  # to support builtin model deployment apis

# Define the model class

"""

TODO: Consilidate the fit, predict, and validation methods
into a custom display method that passes a data location to a model 
and emits an object for display.

For this class I want the following
- specs:
    + base
    + transformed
- methods
    + transform:  accepts transformers and creates a transformed spec
    + fit (populates data field)
    + validate: uses fit and produces a viz using the transformed or base specs, with a validation data set
    + predict: passes a data location string to the transformed spec stored in the transformed spec attribute
- data:
    + source_data
    + reference_data: for validation
    
The idea is that someone can start off with a base spec or submit one
they can start with a transformed spec, use the base spec as the transformed spec
or submit a lit of funcs (or maybe a dict of funcs etc) and run a pipeline
that transforms the spec.  They can apply reference data to see if it worked out.
They can add source_data (need a better name) to look at their data of interest visualized.
The class instance stores this state, that gets serialized and stored (hopefully including the
transformers used, but thaty may take some wrangling).  The fit method is used to add
the data to the spec object.
"""


class VGL_VIZ(mlflow.pyfunc.PythonModel):
    REFERENCE_DATA = None
    SCHEMA = "https://vega.github.io/schema/vega-lite/v4.json"
    WIDTH = 600
    HEIGHT = 400

    def __init__(
            self,
            base_spec: Optional[Mapping[str, Dict[str, List]]] = None,
            width: int = WIDTH,
            height: int = HEIGHT,
            schema: str = SCHEMA,
            reference_data: str = REFERENCE_DATA,
            source_data: Optional[str] = None,
            transformed_spec: Optional[Mapping[str, Dict[str, List]]] = None,
            transformers: Optional[List] = None, # TODO implement transformers, likely in another module
    ):

        self.base_spec = base_spec
        self.width = width
        self.height = height
        self.schema = schema
        self.reference_data = reference_data
        self.transformed_spec = transformed_spec
        self.transformers = transformers

        if not base_spec:
            self.base_spec = {
                "config": {
                    "view": {"continuousWidth": width, "continuousHeight": height}
                },
                "$schema": "",
                "data": {"url": ""},
                "transform": [],
                "facet": {},
                "spec": {"layer": []},
                "resolve": {},
            }

    @classmethod
    def create_artifact_folders(
            cls, target_folders: Optional[List[str]] = None
    ):
        """
        Creates a folder to house artifacts (specs in this case) if one does not
        already exist.

        Returns:
            IO[str]: [description]
        """

        cls.all_folders = {"spec_folder": "spec"}

        if target_folders:
            folders_to_create = {
                key: val
                for (key, val) in cls.all_folders.items()
                if key in target_folders
            }

        else:
            folders_to_create = cls.all_folders

        for folder_type, folder_name in folders_to_create.items():
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

    @classmethod
    def write_spec_file(
            cls,
            spec_to_write: Mapping[str, Dict[str, List]],
            write_path: str,
            indent: int = 4,
    ) -> IO[bytes]:
        """
        Accepts a vega specification and writes it to a file.

        Args:
            spec_to_write (Mapping[str, Dict[str, List]]): [description]
            write_path (str): [description]
            indent (int, optional): [description]. Defaults to 4.

        Returns:
            IO[bytes]: [description]
        """

        with open(write_path, "w") as write_file:
            json.dump(spec_to_write, write_file, indent=indent)

    def transform(
            self, base_spec: Mapping[str, Dict[str, List]], *args, **kwargs
    ) -> Mapping[str, Dict[str, List]]:
        """
        This is the initial implementation of a transform pipeline for Vega Specs,
        it accepts a base spec (a spec to serve as the foundation for the transformed
        spec) and returns a transformed spec.
        
        Currently 
        Args:
            base_spec (Mapping[str, Dict[str, List]]): A vegalite spec to serve
            as the basis of the transform.  Currently this function just returns
            the "transformed_spec" attribute, such that you can transform the base spec
            manually, or just start with a "transformed spec" and store it in a class
            instance.
            
            This method also creates artifact folders and writes the spec to a file.

        Returns:
            transformed_spec (Mapping[str, Dict[str, List]]): [description]
        """

        # TODO: unpack args and kwargs and then insert into base spec where kwargs
        # TODO: stop looking at me sideways matplotlib

        initial_spec = copy.deepcopy(base_spec)

        # we may adding a transformed spec from external rather than creating one
        transformed_spec = self.transformed_spec

        VGL_VIZ.create_artifact_folders()
        spec_folder_name = VGL_VIZ.all_folders.get("spec_folder")
        VGL_VIZ.write_spec_file(
            transformed_spec, f"{spec_folder_name}/transformed_spec.json"
        )

        return transformed_spec

    def fit(
            self, transformed_spec: Mapping[str, Dict[str, List]], data_to_fit_spec: str
    ):
        """
        A fit function that joins a transformed spec with a data location
        via a string object.

        Args:
            transformed_spec (Mapping[str, Dict[str, List]]): A vegalite specification
            data_to_fit_spec (str): A string representing a data location.

        Returns:
            The return is a Tuple of two objects:
            fitted_spec (Mapping[str, Dict[str, List]]): A vegalite specification
            "fit" with a data location.
            A string representation of the vegalite spec json object.
        """

        # TODO: implement population of data field

        fitted_spec = copy.deepcopy(transformed_spec)
        fitted_spec.update(data={"url": data_to_fit_spec})

        return fitted_spec, json.dumps(fitted_spec)

    def validate(self) -> Mapping[str, Dict[str, List]]:
        """
        This is similiar to fit and needs to be evaluated to see if it
        makes sense.  The point of this function is to fit it with 
        a spec that gives a known visualization output as a "reality check"
        validation, prior to pointing at other data sources.
        
        Returns:
            validation_spec (Mapping[str, Dict[str, List]]): A vegalite spec
            "fit" with a validation/reality check data location.
        """

        validation_spec = copy.deepcopy(self.transformed_spec)
        validation_spec, _ = self.fit(validation_spec, self.reference_data)

        VGL_VIZ.create_artifact_folders()
        spec_folder_name = VGL_VIZ.all_folders.get("spec_folder")
        VGL_VIZ.write_spec_file(
            validation_spec, f"{spec_folder_name}/validation_spec.json"
        )

        return validation_spec

    def predict(
            self, context: Optional[List] = None, model_input= None
    ) -> Mapping[str, Dict[str, List]]:
        """
        This is an implementation of the predict function associated with MLflow's pyfunc
        flavor.  The type check is an adaptation to an older version (current at the time
        it was initially implemented), which only accepted arrays/dfs to facilitate passing a string (data location)
        rather than a dataframe.  This is basically just a way to pass an arbritray
        data location to transformed spec.  

        Args:
            model_input (Optional[pd.Dataframe] | str): A path to data in a sting object or pandas dataframe
            Note that a Pandas DataFrame is the expected input type when this function is used
            with the builtin model serve API invocation endpoint.  For other use cases, where
            the function is loaded and used directly, a string can be passed.
            context (Optional[List], optional): [description]. Defaults to None.

        Returns:
            predict_spec (Mapping[str, Dict[str, List]]): Transformed spec attribute populated with
            a data location.
        """
             
        predict_spec = copy.deepcopy(self.transformed_spec)

        # accomodate the builtin model deployment api, which requires pandas
        if isinstance(model_input, pd.DataFrame):
            try:
                source_data = model_input.data_paths[0]
            except Exception as err:
                print(err)
        else:
            source_data = model_input

        # standard workflow
        predict_spec, predict_spec_string = self.fit(predict_spec, source_data)

        return predict_spec

    @property
    def artifacts(self):
        
        """
        Contents of the spec folder as an attribute.
        """

        artifacts = {"spec_files": VGL_VIZ.all_folders.get("spec_folder")}

        return artifacts

    def __str__(self):
       
        return self.__repr__()

    def __repr__(self):
               
        return json.dumps(self.__dict__)
