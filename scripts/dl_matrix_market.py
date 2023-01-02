import requests
import sys
import os
from alive_progress import alive_bar
from bs4 import BeautifulSoup
import wget
import tarfile

def download(matrix_name: str, output_path: str) -> None:
    output_path_complete = f"{output_path}/{matrix_name}.mm"
    try:
        # Get the URL of the matrix
        url = _get_matrix_url(matrix_name)
        print("Downloading matrix from:", url, "    to:", output_path_complete)
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        # Download the file
        _dl_from_url(url, output_path_complete)
    except Exception as e:
        print(e)


def _get_matrices_url() -> dict:
    browse_page_url = "https://math.nist.gov/MatrixMarket/matrices.html"
    response = requests.get(browse_page_url)
    if response.status_code != 200:
        raise Exception("Failed to get matrix URL:", response.status_code)
    soup = BeautifulSoup(response.text, 'html.parser')
    block = soup.find('html').find('body').find_all('center')[1].find('table')
    data = {}
    for link in block.find_all('a', recursive=True):
        data[link.text.lower()] = f"https://math.nist.gov{link['href']}"
    return data


def _get_matrix_url(matrix_name: str) -> str:
    matrix_name = matrix_name.lower()
    matrices = _get_matrices_url()
    url = matrices.get(matrix_name)
    if url is None:
        raise Exception(f"Matrix {matrix_name} not found")
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to get matrix URL:", response.status_code)
    soup = BeautifulSoup(response.text, 'html.parser')
    dl_block = soup.find('html').find('body').find('ul').find('dt').find('li')
    for link in dl_block.find_all('a'):
        if link['href'].endswith(".gz"):
            return f"https://math.nist.gov{link['href']}"
    raise Exception("Failed to find matrix URL")


def _dl_from_url(url: str, output_destination: str) -> None:
    tmp_destination = "/tmp/temporary.mtx.gz"
    with alive_bar(bar='blocks', spinner='classic') as bar:
        wget.download(url, out=tmp_destination, bar=lambda current, total, _: bar(int(current / total * 100)))
    try:
        file = tarfile.open(tmp_destination)
        file.extractall(path=output_destination)
        file.close()
    finally:
        os.remove(tmp_destination)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <matrix_names ...>\n"
              "You can find the matrices here: https://math.nist.gov/MatrixMarket/matrices.html"
              "\n")
        sys.exit(1)
    for matrix_name in sys.argv[1:]:
        download(matrix_name, "matrices")


if __name__ == '__main__':
    main()
