import os
import sys


class PyqPath:
    @staticmethod
    def get_pyq_package_dir():
        paths_script_dir, _ = os.path.split(__file__)
        return paths_script_dir

    @staticmethod
    def get_project_absolute_dir():
        pyq_project_dir, _ = os.path.split(PyqPath.get_pyq_package_dir())
        return pyq_project_dir

    @staticmethod
    def get_project_internal_dir():
        return os.path.basename(PyqPath.get_project_absolute_dir())

    @staticmethod
    def get_relative_internal_file_path():
        project_absolute_dir = PyqPath.get_project_absolute_dir()
        return os.getcwd().replace(project_absolute_dir + "/", "")

    @staticmethod
    def get_class_path(class_type: type):
        return os.path.abspath(sys.modules[class_type.__module__].__file__)
