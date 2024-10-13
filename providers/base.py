# Licensed under the Apache License, Version 2.0 (the "License");
# Copyright 2024 Øivind Loe
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0 for details.
# ~

import os
from enum import Enum

import pandas as pd
from PyPDF2 import PdfReader

from olib.py.utils.csv import readCSV


class LoadType(Enum):
    fullLoad = 0
    incremental = 1
    onlyCache = 2


class Provider:
    _providers: dict[str, type['Provider']] = {}

    def __init__(self, finance):
        self.fin = finance

    def iterDirectoryPdfs(self, path):
        for root, _, files in os.walk(f'{self.fin.records_root}/{path}'):
            for fileName in files:
                if fileName.endswith('.pdf'):
                    path = os.path.join(root, fileName)
                    with open(path, 'rb') as f:
                        content = '\n'.join(page.extract_text() for page in PdfReader(f).pages)

                        yield content, path

    def iterDirectoryCSVs(self, path):
        for root, _, files in os.walk(f'{self.fin.records_root}/{path}'):
            for fileName in files:
                if fileName.endswith('.csv'):
                    path = os.path.join(root, fileName)

                    # Provide provider/account as reference
                    pathSplit = path.replace('/statements', '').split('/')
                    account = f'{pathSplit[-3]}/{pathSplit[-2]}'

                    with readCSV(path) as csv:  # pylint: disable=contextmanager-generator-missing-cleanup
                        yield csv, account

    def readDirectoryCSVs(self, path, skipRows=None):
        frames = []
        for root, _, files in os.walk(f'{self.fin.records_root}/{path}'):
            for fileName in files:
                if fileName.lower().endswith('.csv'):
                    path = os.path.join(root, fileName)

                    # Provide provider/account as reference
                    pathSplit = path.replace('/statements', '').split('/')
                    account = f'{pathSplit[-3]}/{pathSplit[-2]}'

                    data = pd.read_csv(path, skiprows=skipRows)
                    if data.shape[0]:
                        data['_account'] = account
                        frames.append(data)

        return pd.concat(frames, ignore_index=True)

    def __init_subclass__(cls, **kwargs):
        Provider._providers[cls.__name__] = cls

    @staticmethod
    def getProviders():
        return Provider._providers


class FinProvider(Provider):
    pass
