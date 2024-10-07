import yaml
import threading


def singleton(cls):
    instances = dict()

    lock = threading.Lock()

    def wrap(*args, **kwargs):
        nonlocal instances

        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        print("Singleton: ", instances[cls])
        return instances[cls]

    return wrap


@singleton
class LoadConfig:
    def __init__(self, yaml_filename: str = None):
        self.__filename = yaml_filename
        self.__section = None
        self.__config = None
        if yaml_filename is not None:
            self.set_yamlfile(yaml_filename)

    def __str__(self):
        return f"LoadConfig:: filename: {self.__filename} section: {self.__section}"

    def set_yamlfile(self, yaml_filename: str):
        with open(yaml_filename, 'r') as file:
            self.__config = yaml.safe_load(file)

    def set_section(self, section: str):
        if section not in self.__config:
            raise KeyError(f"section: {section} not found")
        self.__section = section

    def parameter(self, param_name: str):
        # warnings.warn("Busca parametros hasta un solo nivel de anidamiento", UserWarning)
        # TODO search parameters at all level
        if self.__section is None:
            raise Exception("CONFIG:: section not loaded")

        if param_name in self.__config[self.__section]:
            return self.__config[self.__section][param_name]

        for _val in self.__config[self.__section].values():
            if isinstance(_val, dict):
                if param_name in _val:
                    return _val[param_name]

        raise KeyError(f"CONFIG:: parameter: {param_name} not found in {self.__section}")

    def get_section(self, section_name):
        if section_name not in self.__config:
            raise Exception(f"section: {section_name} not found")
        return self.__config[section_name]

    def get_config(self):
        return self.__config


if __name__ == "__main__":
    config_loader = LoadConfig("config.yml")
    config_loader.set_section('openai')
    cnf2 = LoadConfig()
    print(config_loader.parameter('tokens'))
    print(config_loader.parameter('max_gpt_35'))
    print(config_loader.parameter('max_batches_tables'))
    print(config_loader is cnf2)
