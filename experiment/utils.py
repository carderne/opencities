from urllib.parse import urlparse

import boto3
from pystac import STAC_IO


def my_read_method(uri):
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path[1:]
        s3 = boto3.resource("s3")
        obj = s3.Object(bucket, key)
        return obj.get()["Body"].read().decode("utf-8")
    else:
        return STAC_IO.default_read_text_method(uri)


def my_write_method(uri, txt):
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path[1:]
        s3 = boto3.resource("s3")
        s3.Object(bucket, key).put(Body=txt)
    else:
        STAC_IO.default_write_text_method(uri, txt)


def str_to_bool(x):
    if type(x) == str:
        if x.lower() == "true":
            return True
        elif x.lower() == "false":
            return False
        else:
            raise ValueError(f"{x} is expected to be true or false")
    return x


def read_list(file):
    result = []
    with open(file) as f:
        result = f.readlines()
    result = [l.strip() for l in result]
    return result
