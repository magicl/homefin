# Licensed under the Apache License, Version 2.0 (the "License");
# Copyright 2024 Øivind Loe
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0 for details.
# ~

import pandas as pd

from .base import FinProvider


class ChaseProvider(FinProvider):
    statementPath = 'chase'

    def readTransactions(self):
        df = self.readDirectoryCSVs(self.statementPath)

        df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], format='%m/%d/%Y')
        df['_cents'] = df['Amount'].astype(float) * 100
        df['Category'] = df['Category'].fillna('')

        return df[['_account', 'Transaction Date', '_cents', 'Description', 'Category']]
