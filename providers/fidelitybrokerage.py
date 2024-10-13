# Licensed under the Apache License, Version 2.0 (the "License");
# Copyright 2024 Øivind Loe
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0 for details.
# ~

import pandas as pd

from .base import FinProvider


class FidelityBrokerageProvider(FinProvider):
    """
    Used for HSA and other brokerage accounts we want to track
    """

    statementPath = 'fidelity/brokerage'

    def readTransactions(self):
        df = self.readDirectoryCSVs(self.statementPath, skipRows=2)

        # Clear text rows at bottom of doc
        df = df[~df['Action'].isna()]

        desc = df['Security Description'] if 'Security Description' in df.columns else df['Description']
        desc = desc.fillna('').str.replace('No Description', '').str.strip()

        df['Run Date'] = pd.to_datetime(df['Run Date'].str.strip(), format='%m/%d/%Y')
        df['_cents'] = df['Amount ($)'].astype(float) * 100
        df['_memo'] = (df['Symbol'].fillna('') + ' - ' + desc).str.strip()
        df['_account'] = 'fidelity/' + df['_account']
        df['Action'] = df['Action'].str.strip()

        return df[['_account', 'Run Date', '_cents', 'Action', '_memo']]
