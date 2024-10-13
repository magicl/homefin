# Licensed under the Apache License, Version 2.0 (the "License");
# Copyright 2024 Øivind Loe
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0 for details.
# ~

import pandas as pd

from .base import FinProvider


class DiscoverProvider(FinProvider):
    """
    Discover notes:
    """

    statementPath = 'discover'

    def readTransactions(self):
        df = self.readDirectoryCSVs(self.statementPath)

        df['Post Date'] = pd.to_datetime(df['Post Date'], format='%m/%d/%Y')
        df['_cents'] = -df['Amount'].astype(float) * 100

        return df[['_account', 'Post Date', '_cents', 'Description', 'Category']]
