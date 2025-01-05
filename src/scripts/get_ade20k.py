import os
import zipfile

from gluoncv.utils import download, makedirs

_TARGET_DIR = os.path.expanduser("data/ade")


def download_ade(path, overwrite=False):
    _AUG_DOWNLOAD_URLS = [
        (
            "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
            "219e1696abb36c8ba3a3afe7fb2f4b4606a897c7",
        ),
        (
            "http://data.csail.mit.edu/places/ADEchallenge/release_test.zip",
            "e05747892219d10e9243933371a497e905a4860c",
        ),
    ]

    download_dir = os.path.join(path, "downloads")
    makedirs(download_dir)

    for url, checksum in _AUG_DOWNLOAD_URLS:
        filename = download(
            url, path=download_dir, overwrite=overwrite, sha1_hash=checksum
        )
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(path=path)


if __name__ == "__main__":
    download_ade(_TARGET_DIR, overwrite=False)
