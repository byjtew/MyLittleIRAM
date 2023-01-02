import requests
import sys
import os
from alive_progress import alive_bar
from bs4 import BeautifulSoup
import wget


def download(matrix_name: str, output_path: str) -> None:
    print(f"Downloading {matrix_name} to {output_path}")
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    # Get the URL of the matrix
    url = _get_matrix_url(matrix_name)
    print("Downloading matrix from:", url)
    # Download the file
    _dl_from_url(url, f"{output_path}/{matrix_name}.txt")


def _get_matrices_url() -> dict:
    browse_page_url = "https://math.nist.gov/MatrixMarket/matrices.html"
    response = requests.get(browse_page_url)
    if response.status_code != 200:
        print("Failed to get matrix URL:", response.status_code)
        sys.exit(1)
    soup = BeautifulSoup(response.text, 'html.parser')
    block = soup.find('html').find('body').find_all('center')[1].find('table')
    data = {}
    for link in block.find_all('a', recursive=True):
        data[link.text] = f"https://math.nist.gov{link['href']}"
    return data


def _get_matrix_url(matrix_name: str) -> str:
    matrices = _get_matrices_url()
    url = matrices.get(matrix_name)
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to get matrix URL:", response.status_code)
        sys.exit(1)
    soup = BeautifulSoup(response.text, 'html.parser')
    dl_block = soup.find('html').find('body').find('ul').find('dt').find('li')
    for link in dl_block.find_all('a'):
        if link['href'].endswith(".gz"):
            return f"https://math.nist.gov{link['href']}"
    print("Failed to find matrix URL")
    sys.exit(1)


def _dl_from_url(url: str, output_destination: str) -> None:
    with alive_bar(bar='blocks', spinner='classic') as bar:
        wget.download(url, out=output_destination, bar=lambda current, total, _: bar(int(current / total * 100)))
