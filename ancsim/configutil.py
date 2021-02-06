import ancsim.utilities as util
import yaml

def getBaseConfig():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    return config

def processConfig(conf, numFilt):
    if isinstance(conf["BLOCKSIZE"], int):
        conf["BLOCKSIZE"] = [conf["BLOCKSIZE"] for _ in range(numFilt)]

    checkConfig(conf, numFilt)
    return conf

def checkConfig(conf, numFilt):
    if isinstance(conf["BLOCKSIZE"], list):
        for bs in conf["BLOCKSIZE"]:
            assert util.isPowerOf2(bs)

    assert len(conf["SOURCEAMP"]) == conf["NUMSOURCE"]
    assert numFilt == len(conf["BLOCKSIZE"])