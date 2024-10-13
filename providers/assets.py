# Licensed under the Apache License, Version 2.0 (the "License");
# Copyright 2024 Øivind Loe
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0 for details.
# ~

import numpy as np
import pandas as pd

from olib.py.utils.clients.google.sheets import gsReadDataFrame

from .base import FinProvider


class AssetsProvider(FinProvider):
    """
    Currently uses the 'Assets' tab in the spreadsheet where account balances are recorded
    """

    statementPath = '*assets*'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accounts = []

    def readTransactions(self):
        data = gsReadDataFrame('Assets', gs=self.fin.gs)
        data = data[[c for c in data.columns if not data[c].isna().all()]]  # Remove columns that are all NA
        data = data.rename(columns={'Date': 'date'})
        data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')
        data = data.sort_values('date')

        frames = []

        # Add transactions for all account values
        for c in data.columns:
            if c == 'date':
                continue

            subset = data[['date', c]][~data[c].isna()].rename(columns={c: 'value'})

            # Transaction delta.. This will then later be converted into balances
            values = subset['value'].values
            subset['cents'] = np.array([values[0], *(values[1:] - values[:-1])]) * 100
            subset['account'] = c
            subset['name'] = 'homefin - balance adjustment'
            subset['memo'] = ''

            frames.append(subset)
            self.accounts.append(c)

        df = pd.concat(frames)
        return df[['account', 'date', 'cents', 'name', 'memo']]

    def updateTransactions(self):
        frames = []
        for c in self.accounts:
            account = self.fin.accounts.loc[c]

            # Copy in transactions
            matches = self.fin.trans[
                (self.fin.trans['category'] == account['inboundCat'])
                | (self.fin.trans['category'] == account['outboundCat'])
            ]
            if matches.shape[0]:
                copy = matches.copy()
                copy['category'] = 'Exclude/InvestTrans'
                copy['account'] = c
                copy['cents'] = -copy['cents']
                copy['desc'] = 'contra: ' + copy['desc']

                frames.append(copy)

        self.fin.trans = pd.concat([self.fin.trans, *frames])

        # Adjust initial balance adjustments. Note, trans are sorted again after all update calls, so sorting is ok
        self.fin.trans = self.fin.trans.sort_values(by=['account', 'date']).reset_index(drop=True)
        index = self.fin.trans[self.fin.trans['account'].isin(self.accounts)].index

        trans = self.fin.trans.loc[index]
        trans['group'] = (trans['category'] == 'Exclude/BalanceAdjust').loc[::-1].cumsum()[::-1]  # Reverse cumsum

        def func(group):
            if group['category'].iloc[-1] == 'Exclude/BalanceAdjust':
                # Value of preceeding items + value of last should be equal to current value of last
                preVal = group['cents'].iloc[:-1].sum()
                balVal = group['cents'].iloc[-1]

                group.loc[group.index[-1], 'desc'] += f' (original: {balVal/100})'
                group.loc[group.index[-1], 'cents'] = balVal - preVal

            return group

        trans = trans.groupby('group').apply(func).reset_index(level=0, drop=True).drop('group', axis=1)
        self.fin.trans.loc[index] = trans
