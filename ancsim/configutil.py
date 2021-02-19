import ancsim.utilities as util
from pathlib import Path
import yaml

def getBaseConfig():
    with open(Path(__file__).parent.joinpath("config.yaml")) as f:
        config = yaml.safe_load(f)
    return config

def processConfig(conf):
    #if isinstance(conf["BLOCKSIZE"], int):
    #    conf["BLOCKSIZE"] = [conf["BLOCKSIZE"] for _ in range(numFilt)]

    checkConfig(conf)
    return conf

def checkConfig(conf):
    if isinstance(conf["BLOCKSIZE"], list):
        for bs in conf["BLOCKSIZE"]:
            assert util.isPowerOf2(bs)

   # assert numFilt == len(conf["BLOCKSIZE"])