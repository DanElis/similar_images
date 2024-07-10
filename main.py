import argparse
import yaml

from controller import Controller
from factories import FactoryReaders, FactorySearchEngines, FactorySaver


def parse_args():
    parser = argparse.ArgumentParser(description="Measure the search time for similar images")
    parser.add_argument(
        "--config"
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    factory_readers = FactoryReaders()
    factory_saver = FactorySaver()
    factory_search_engines = FactorySearchEngines()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    controller = Controller(config=config, factory_readers=factory_readers, factory_search_engines=factory_search_engines, factory_savers=factory_saver)
    controller.run()