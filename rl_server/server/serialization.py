import zlib
import marshal

# marshal is the fastest for serialization, sending and receiving data
# also tested
# - json
# - pickle


def serialize(object):
    return zlib.compress(marshal.dumps(object, 2))


def deserialize(bytes):
    return marshal.loads(zlib.decompress(bytes))
