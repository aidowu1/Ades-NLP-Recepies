import pickle as pl

class GenericDataSerializer(object):
    """
    Generic Serilizer of data to python's pickle format
    """

    @staticmethod
    def serialize(data, cache_file_path):
        """
        Serilaizes Data to pickle format
        """
        try:
            print("Starting to Serialize data to disk path: {}".format(cache_file_path))
            with open(cache_file_path, 'wb') as file:
                pl.dump(data, file)
                print("Serialized data to disk path: {}".format(cache_file_path))
        except Exception as ex:
            msg = "Error raised during serialization of data:\n{}\n".format(str(ex))
            print(msg)
            raise ex

    @staticmethod
    def deSerializeCache(cache_file_path):
        """
        De-serilaizes Data from pickle format to native Python object
        """
        try:
            with open(cache_file_path, 'rb') as file:
                data = pl.load(file)
                print("De-Serializing data located in the path: {}".format(cache_file_path))
            return data
        except Exception as ex:
            msg = "Error raised during de-serialization of data:\n{}\n".format(str(ex))
            print(msg)
            raise ex