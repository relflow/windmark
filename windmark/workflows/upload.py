import flytekit as fl
import os


@fl.task(cache=True, cache_version="2.0")
def create() -> fl.FlyteDirectory:
    return fl.FlyteDirectory.from_source("s3://union-oc-production-demo/windmark/trxns/")


@fl.task
def show(data: fl.FlyteDirectory):
    data.download()

    filenames = [filename for filename in os.listdir(data.path)]

    for filename in filenames:
        print(filename)


@fl.workflow
def upload():
    show(create())
