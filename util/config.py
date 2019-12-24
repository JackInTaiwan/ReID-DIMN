import os
import abc
import shutil
import logging

from ruamel.yaml import YAML

logger = logging.getLogger(__name__)



class TrainerConfig:
    def __init__(self, config_path=None, args={}):
        self.config_path = config_path
        self.args = args


    def load_config(self):
        yaml = YAML(typ="safe")
        with open(self.config_path) as f:
            args = yaml.load(f)
        self.args.update(args)
    

    def __getitem__(self, key):
        key_list = key.split(".")
        value = self.args

        for k in key_list:
            value = value[k]
        value = value if type(value) != dict else TrainerConfig(args=value)

        return value


    def copy_config(self):
        dir_path = self["checkpoints.save_dir"]
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
            logger.info("| Created new directory \'{}\' .".format(dir_path))

        try:
            shutil.copy(self.config_path, dir_path)
            logger.info("| Copy config file to \'{}\' .".format(os.path.join(dir_path, os.path.split(self.config_path)[1])))
        except shutil.SameFileError:
            pass
