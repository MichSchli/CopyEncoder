import argparse

from Mindblocks.interface import BasicInterface

interface = BasicInterface()

parser = argparse.ArgumentParser(description='Train (and test) a model with a given block.')
#parser.add_argument('--block')
args = parser.parse_args()

block_filepath = "blocks/autoencoder_without_gates.xml"
block_name = ".".join(block_filepath.split('/')[-1].split(".")[:-1])

data_filepath = "data/toy"

interface.load_file(block_filepath)
interface.set_variable("data_folder", data_filepath)
interface.initialize()

for i in range(100):
    interface.train(1)
    for line in interface.predict():
        print(line)