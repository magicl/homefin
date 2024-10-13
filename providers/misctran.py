# Licensed under the Apache License, Version 2.0 (the "License");
# Copyright 2024 Øivind Loe
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0 for details.
# ~

import pandas as pd

from olib.py.utils.clients.google.sheets import gsReadDataFrame

from .base import FinProvider


class MiscTranProvider(FinProvider):
    """Pulls transactions from MiscTran for adhoc transactions"""

    statementPath = '*misc transactions*'

    def readTransactions(self):
        df = gsReadDataFrame('MiscTran', gs=self.fin.gs)
        df = df[~df['Date'].isna()]

        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        df['_cents'] = df['Amount'] * 100
        df['_memo'] = df['Memo'].fillna('')

        return df[['Account', 'Date', '_cents', 'Description', '_memo']]
