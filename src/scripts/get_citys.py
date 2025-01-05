import os
import zipfile

from gluoncv.utils import makedirs
from mxnet.gluon.utils import check_sha1

_TARGET_DIR = os.path.expanduser("data/citys")


def download_city(path, overwrite=False):
    _CITY_DOWNLOAD_URLS = [
        ("gtFine_trainvaltest.zip", "99f532cb1af174f5fcc4c5bc8feea8c66246ddbc"),
        ("leftImg8bit_trainvaltest.zip", "2c0b77ce9933cc635adda307fbba5566f5d9d404"),
    ]
    download_dir = os.path.join(path, "downloads")
    makedirs(download_dir)
    for filename, checksum in _CITY_DOWNLOAD_URLS:
        if not check_sha1(filename, checksum):
            raise UserWarning(
                "File {} is downloaded but the content hash does not match. "
                "The repo may be outdated or download may be incomplete. "
                'If the "repo_url" is overridden, consider switching to '
                "the default repo.".format(filename)
            )
        # extract
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(path=path)
        print("Extracted", filename)


if __name__ == "__main__":
    download_city(_TARGET_DIR, overwrite=False)
