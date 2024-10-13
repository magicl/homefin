# Licensed under the Apache License, Version 2.0 (the "License");
# Copyright 2024 Øivind Loe
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0 for details.
# ~

import os

import pandas as pd

from .base import FinProvider


class AmazonProvider(FinProvider):
    """
    Amazon notes:
    - Must use data request to get data. Data comes in folder with orders, returns etc. spread across subfolders
    - Only process the last export, as each export contains all-time data
    - Must look at payment type to determine payer

    Files in use:
    - Retail.OrderHistory.1        - List of purchased orders. Not tied to payments to amazon account
    - Retail.CustomerReturns.1     - ?? return authorizations
    - Retail.CustomerReturns.1.1   - ?? return transactions

    """

    statementPath = 'amazon'

    def readTransactions(self):
        # Find the last export directory
        basePath = os.path.expanduser(f'{self.fin.records_root}/{self.statementPath}')
        basePath = os.path.join(basePath, max(d for d in os.listdir(basePath) if d.endswith('Your Orders')))

        # Read orders
        orders = pd.read_csv(os.path.join(basePath, 'Retail.OrderHistory.1/Retail.OrderHistory.1.csv'))
        # Order dates do not consistently have a microsecond component. Ignore microsecond and 'Z' suffix
        orders['Order Date'] = pd.to_datetime(orders['Order Date'], format='%Y-%m-%dT%H:%M:%S', exact=False)

        # Ignore
        payment_instr_exclude = self.fin.env.list('AMAZON_PAYMENT_TYPE_EXCLUDE')
        orders = orders[~orders['Payment Instrument Type'].isin(payment_instr_exclude)]

        orders['_account'] = 'amazon'
        orders['_cents'] = -orders['Total Owed'].str.replace(',', '').astype(float) * 100
        orders['_desc'] = orders['Quantity'].astype(str) + 'x ' + orders['Product Name']

        # Add in payment transactions to amazon. Do this on a per-order basis, not per-line
        payments = (
            orders.set_index(['Order ID', 'Order Date'])
            .groupby(['Order ID', 'Order Date'])['_cents']
            .sum()
            .reset_index()
        )
        payments['_desc'] = 'Assumed Payment'
        payments['ASIN'] = ''
        payments['_cents'] = -payments['_cents']
        payments['_account'] = 'amazon'

        output = pd.concat([orders, payments])

        return output[['_account', 'Order Date', '_cents', '_desc', 'ASIN']]
