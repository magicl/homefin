# Licensed under the Apache License, Version 2.0 (the "License");
# Copyright 2024 Øivind Loe
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0 for details.
# ~

import datetime
import math
import operator
import os
import re
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from olib.py.utils.clients.google.sheets import (
    addGroupsOffset,
    gsApplyColGroups,
    gsApplyRowGroups,
    gsOpen,
    gsReadDataFrame,
    gsUpdateFormatted,
    gsWriteDataFrame,
    sheetCoord,
)
from providers.base import Provider

from .budget import Budget
from .formatting import D as fmtDollars
from .formatting import P as fmtPercent
from .formatting import (
    dfMixColumns,
    fmtBgGoodBad,
    fmtDfToGridHierarchical,
    fmtGridApplyFmtFuncs,
    fmtGridsAssemble,
)
from .utils import create_aggregates

# from gspread_formatting import color  # pylint: disable=no-name-in-module
# from gspread_formatting import cellFormat  # pylint: disable=no-name-in-module


class DataSet:
    def __init__(self, freq: str, fin: 'Finance'):
        self.freq = freq
        self.fin = fin

        self.pnl = self.dataPNL()
        self.bs = self.dataBS()

        self.pnlBlank = self.pnl.copy()
        self.pnlBlank.loc[:, :] = 0

        self.bsBlank = self.bs.copy()
        self.bsBlank.loc[:, :] = 0

        self.pnlBudget = pd.DataFrame()
        self.pnlBudgetDelta = pd.DataFrame()

        self.bsContrib = pd.DataFrame()
        self.bsOrganic = pd.DataFrame()
        self.bsRateReturn = pd.DataFrame()

        self.miscTax = pd.DataFrame()
        self.miscTaxBlank = pd.DataFrame()

        # REMOVE NEED FOR BLANKs
        # ADD Profit cumsum within months -> make chart
        # ADD Invest cumsum within months -> make chart

    def finalize(self, budget):
        # Finalize is a separate function, because sheets must be updated based on pnl/bs before budget can be created
        self.pnlBudgetDelta, self.pnlBudget = budget.dataBudget(self.freq, self.pnl, self.bs)
        self.bsContrib, self.bsOrganic, self.bsRateReturn = self.dataBSExtraCols()

        self.miscTax = self.dataTax()
        self.miscTaxBlank = self.miscTax.copy()
        self.miscTaxBlank.loc[:, :] = 0

    def dataPNL(self):
        """
        Profits & Losses
        Creates a table with taxonomy categories as rows, time as columns, and sum(cents) as values
        """
        table = (
            self.fin.trans[~self.fin.trans['pnlExclude']]
            .groupby([pd.Grouper(key='date', freq=self.freq, label='right'), 'category'])['cents']
            .sum()
            .unstack(level=0)
            .reindex(self.fin.taxonomy[~self.fin.taxonomy['pnlExclude']].index)  # Column order defined by taxonomy
            .fillna(0)
            / 100
        )
        table = table[[c for c in table.columns if c >= self.fin.cutoff]]
        table.index = table.index.map(self.fin.taxonomy['longName'])

        table = pd.concat([table, *create_aggregates(table)])

        return table

    def dataBS(self):
        """
        Balance Sheet
        Creates a table with accounts as rows, time as columns, and sum(cents) as values
        """
        table = (
            self.fin.trans.groupby([pd.Grouper(key='date', freq=self.freq, label='right'), 'account'])['balance']
            .last()
            .unstack(level=0)
            .reindex(self.fin.accounts.index)
            .reindex(self.pnl.columns, axis=1)
            .ffill(axis=1)  # Fill forward
            .fillna(0)  # Additional fill for cases where there is no activity
            / 100
        )
        table = table[[c for c in table.columns if c >= self.fin.cutoff]]
        table.index = table.index.map(self.fin.accounts['longName'])

        table = pd.concat([table, *create_aggregates(table, pnl=False)])

        return table

    def dataBSExtra(self, categories: list[str]):
        """
        Provides additional data for balance sheet
        """
        table = (
            self.fin.trans[self.fin.trans['category'].isin(categories)]
            .groupby([pd.Grouper(key='date', freq=self.freq, label='right'), 'account'])['cents']
            .sum()
            .unstack(level=0)
            .reindex(self.fin.accounts.index)
            .reindex(self.pnl.columns, axis=1)
            .fillna(0)  # Additional fill for cases where there is no activity
            / 100
        )
        table = table[[c for c in table.columns if c >= self.fin.cutoff]]
        table.index = table.index.map(self.fin.accounts['longName'])

        table = pd.concat([table, *create_aggregates(table, pnl=False)])

        return table

    def _calcRateReturn(self, moOrganic, moBegin, colNames, yearly):
        # If we see infinite growth, it is likely due to funding the account the same months as we saw growth. In this case, zero out
        # growth, as we don't know exactly what happened
        rr = (moOrganic / moBegin.abs()).replace(np.inf, 0).replace(-np.inf, 0).fillna(0)

        # Aggregate
        if yearly:
            rr = rr + 1

            rrAgg = pd.DataFrame(index=rr.index)
            for yearI, year in enumerate(rr.columns.year.unique()):
                cols = rr.columns[rr.columns.year == year]
                rrAgg[colNames[yearI]] = rr[cols].prod(axis=1)

            rr = rrAgg - 1

        return rr

    def dataBSExtraCols(self):
        dataBSContrib = self.dataBSExtra(['Exclude/InvestTrans', 'Tran/External'])
        dataBSOrganic = self.dataBSExtra(['Exclude/BalanceAdjust', 'Inc/Interest'])

        # For Time-Weighted Rate of Return (TWRR), we need month frequency even if overall frequency is year
        if self.freq == 'YE':
            if self.fin.dsM is None:
                raise Exception('dsM must be created')

            moDataBS = self.fin.dsM.bs
            moOrganic = self.fin.dsM.dataBSExtra(['Exclude/BalanceAdjust', 'Inc/Interest'])
        else:
            moDataBS = self.bs
            moOrganic = dataBSOrganic

        # Calculate month over month rate of return. Organic growth of time period divided by beginning balance of period
        moDataBSBegin = (moDataBS - moDataBS.diff(axis=1)).fillna(0)
        rr = self._calcRateReturn(moOrganic, moDataBSBegin, self.pnl.columns, self.freq == 'YE')

        return dataBSContrib, dataBSOrganic, rr

    def dataTax(self):
        if self.freq != 'YE':
            return pd.DataFrame()

        salary = self.pnl.loc['Income/Salary', :]
        tax_year = self.pnl.loc['Tax/Payment', :]
        tax_refund = self.pnl.loc['Tax/Refund', :].shift(-1).fillna(0)
        tax_total = tax_year + tax_refund
        tax_rate = -tax_total / salary

        return pd.DataFrame(
            {
                'Salary': salary,
                'Tax/*': tax_total,
                'Tax/In Year': tax_year,
                'Tax/Refund': tax_refund,
                'Tax Rate': tax_rate,
            }
        ).T


class Finance:
    OPERATORS = {
        '==': operator.eq,
        '!=': operator.ne,
        '<=': operator.le,
        '>=': operator.ge,
        '<': operator.lt,
        '>': operator.gt,
    }

    def __init__(self, records_root, sheetKey, sheetCreds, env):
        self.records_root = os.path.abspath(os.path.expanduser(records_root))
        self._sheetKey = sheetKey
        self.sheetCreds = sheetCreds
        self.env = env
        self._gs = None
        self.trans = pd.DataFrame()
        self.taxonomy = pd.DataFrame()
        self.accounts = pd.DataFrame()
        self.budget = None
        self.dsY: DataSet | None = None
        self.dsM: DataSet | None = None

        self.cutoff = pd.Timestamp(year=2022, month=1, day=1)
        self.providers = [
            provider(self)
            for _, provider in sorted(Provider.getProviders().items(), key=lambda v: getattr(v[1], 'statementPath', ''))
        ]

    @property
    def gs(self):
        if self._gs is None:
            self._gs = gsOpen(key=self._sheetKey, creds=self.sheetCreds)
        return self._gs

    @staticmethod
    def fmtRule(r):
        return f'{r.rule} - {r.filter}'

    @staticmethod
    def fmtTrans(t):
        return f'{t.date} - ${t.cents/100} - {t.account} - {t.desc} - {t.ref}'

    def load(self):
        """Loads and connects transactions"""
        tStart = time.time()

        ############################################
        # Taxonomy
        ############################################

        taxonomy = gsReadDataFrame('Taxonomy', gs=self.gs)
        self.taxonomy = taxonomy[['Name', 'Long Name', 'Description', 'PNL Exclude']]
        self.taxonomy = self.taxonomy.rename(
            columns={
                'Name': 'name',
                'Long Name': 'longName',
                'Description': 'description',
                'PNL Exclude': 'pnlExclude',
            }
        )
        self.taxonomy = self.taxonomy[~self.taxonomy['name'].isna()]
        self.taxonomy = self.taxonomy.astype({'name': str, 'longName': str, 'description': str})

        self.taxonomy['longName'] = self.taxonomy['longName'].where(
            self.taxonomy['longName'] != 'nan', self.taxonomy['name']
        )
        self.taxonomy['description'] = self.taxonomy['description'].where(self.taxonomy['description'] != 'nan', '')
        self.taxonomy['pnlExclude'] = self.taxonomy['pnlExclude'] == 'Y'

        self.taxonomy.set_index('name', inplace=True)

        ############################################
        # Transactions
        ############################################

        columnNames = ['account', 'date', 'cents', 'desc', 'ref']
        frames = []
        importErrors = False
        for provider in self.providers:
            if hasattr(provider, 'readTransactions'):
                frame = provider.readTransactions()
                originalColNames = list(frame.columns)

                # Providers may not provide a balance value. In that case, it is added below
                if isinstance(frame, list):
                    frame = pd.DataFrame(frame, columns=columnNames[: len(frame[0])])
                else:
                    frame = frame.copy()
                    frame.columns = columnNames[: frame.shape[1]]

                # if 'balance' not in frame.columns:
                #     frame['balance'] = None

                # # frame.loc[frame['balance'] == '', 'balance'] = None
                # frame['balance'] = frame['balance'].astype(
                #     'float64'
                # )  # Explicit float to avoid issues during concat with NaNs

                # NaN checking
                for colName, colNameOriginal in zip(columnNames, originalColNames):
                    if frame[colName].isna().sum():
                        records = frame[frame[colName].isna()]
                        print(
                            f'  ERROR: {records.shape[0]} records from "{provider.statementPath}" have NaN in "{colName}" ("{colNameOriginal}")'
                        )
                        print(records.iloc[:5])
                        importErrors = True
                        continue

                frames.append(frame)
                minDate = frame['date'].min().strftime('%Y-%m-%d')
                maxDate = frame['date'].max().strftime('%Y-%m-%d')
                print(f'  {provider.statementPath:<30} {frame.shape[0]:>5} ε [{minDate}, {maxDate}]')

        if importErrors:
            print('Please fix import errors to continue')
            sys.exit(1)

        self.trans = pd.concat(frames)
        self.trans['date'] = pd.to_datetime(self.trans['date'])
        self.trans['cents'] = self.trans['cents'].round().astype(int)
        self.trans['desc'] = self.trans['desc'].astype('string[pyarrow]')
        self.trans['account'] = self.trans['account'].astype('string[pyarrow]')

        # Important to drop duplicates before a cumulative operation like calculating balances
        self.trans = self.trans.reset_index(drop=True)
        self.trans.drop_duplicates(inplace=True)  # , subset=['account', 'date', 'cents', 'desc'])

        ############################################
        # Accounts
        ############################################

        accounts = gsReadDataFrame('Accounts', gs=self.gs)
        self.accounts = accounts[
            ['Name', 'Long Name', 'Description', 'Balance Ref', 'Inbound Category', 'Outbound Category']
        ]
        self.accounts = self.accounts.rename(
            columns={
                'Name': 'name',
                'Long Name': 'longName',
                'Description': 'description',
                'Balance Ref': 'balanceRef',
                'Inbound Category': 'inboundCat',
                'Outbound Category': 'outboundCat',
            }
        )
        self.accounts = self.accounts[~self.accounts['name'].isna()]
        self.accounts = self.accounts.astype({'name': str, 'longName': str, 'description': str, 'balanceRef': str})

        # Add missing accounts based on transactions
        missingAccounts = set(self.trans['account'].unique()) - set(self.accounts['name'])
        if missingAccounts:
            self.accounts = pd.concat(
                [
                    self.accounts,
                    pd.DataFrame(
                        data=[
                            {'name': n, 'longName': None, 'description': None, 'balanceRef': None}
                            for n in missingAccounts
                        ]
                    ),
                ]
            )

        # Wrap up accounts data
        self.accounts['longName'] = self.accounts['longName'].where(
            self.accounts['longName'] != 'nan', self.accounts['name']
        )
        self.accounts['description'] = self.accounts['description'].where(self.accounts['description'] != 'nan', '')
        self.accounts['inboundCat'] = self.accounts['inboundCat'].where(self.accounts['inboundCat'] != 'nan', '')
        self.accounts['outboundCat'] = self.accounts['outboundCat'].where(self.accounts['outboundCat'] != 'nan', '')

        self.accounts.set_index('name', inplace=True)

        print(f'  Time: {int(time.time() - tStart)} s')

    @staticmethod
    def connect_chunk(trans, rules, taxonomy):
        ruleMatches = pd.Series(0, index=rules.index, dtype='int32')
        ruleIds = pd.Series(0, index=rules.index, dtype='int32')
        ruleErrors = pd.Series('', index=rules.index, dtype='str')
        ruleMoves = pd.Series(0, index=rules.index, dtype='int32')
        ruleMoveExp = pd.Series(0, index=rules.index, dtype='int32')

        for ruleI, rule in enumerate(rules.itertuples()):
            if not isinstance(rule.rule, str):
                continue

            inTaxonomy = rule.category in taxonomy.index
            if not inTaxonomy:
                ruleErrors[ruleI] = 'Invalid category'

            ruleMatch = trans['desc'].str.contains(rule.rule, regex=True, case=False)
            hasDateRule = False
            moveRule = None
            moveCount = 1
            # pnlExclude = False
            if isinstance(rule.filter, str) and rule.filter:
                for filterItem in rule.filter.split(','):
                    filterItemSplit = filterItem.split(':')
                    if len(filterItemSplit) == 2:
                        key, val = filterItemSplit
                    elif len(filterItemSplit) == 1:
                        key, val = filterItemSplit[0], ''
                    else:
                        raise Exception('Invalid options')

                    op: Callable = operator.eq
                    if val:
                        for opK, opV in Finance.OPERATORS.items():
                            if val.startswith(opK):
                                val = val[len(opK) :]
                                op = opV
                                break

                    if key == 'amount':
                        if not val:
                            raise Exception('missing value')

                        ruleMatch &= op(trans['cents'], int(float(val) * 100))

                    elif key == 'date':
                        if not val:
                            raise Exception('missing value')

                        try:
                            ruleMatch &= op(trans['date'], pd.Timestamp(datetime.datetime.strptime(val, '%Y-%m-%d')))
                        except ValueError as e:
                            e.add_note(f'Invalid date: {val} for rule: "{rule.filter}"')
                            raise

                        hasDateRule = True

                    elif key == 'move':
                        if not val:
                            raise Exception('missing value')

                        # Command to move transaction. Only allowd if only one transaction matches rule AND the rule has a 'date' restriction
                        if '#' in val:
                            val, moveCountStr = val.split('#')
                            moveCount = int(moveCountStr)

                        if not hasDateRule:
                            raise Exception('move rule has to be proceeded by a date rule')

                        moveRule = pd.Timestamp(datetime.datetime.strptime(val, '%Y-%m-%d'))

            if isinstance(rule.srcAccount, str):
                ruleMatch &= trans['account'].str.contains(rule.srcAccount)

            # Only apply category / owner to transactions the first time they are seen
            firstMatch = ruleMatch & (trans['rules'] == '')
            trans.loc[firstMatch, 'category'] = rule.category
            trans.loc[firstMatch, 'owner'] = rule.owner
            trans.loc[firstMatch, 'pnlExclude'] = taxonomy.loc[rule.category, 'pnlExclude'] if inTaxonomy else False

            if moveRule is not None:
                # We check number of moves later
                ruleMoves[ruleI] += firstMatch.sum()
                ruleMoveExp[ruleI] = moveCount

                trans.loc[firstMatch, 'date'] = moveRule

            # Don't pollute by adding 'unmapped' to unmapped rules
            if rule.type != 'UNMAP':
                # Set rule match on all matching transactions for debug
                # Specific rules are denoted with a !. Should have at most 1 specific rule per trans
                specificRule = '' if rule.type == 'SHARED' else '!'
                trans.loc[ruleMatch, 'rules'] += f'{ruleI}{specificRule},'
                if rule.type != 'SHARED':
                    trans.loc[ruleMatch, 'specific'] += 1
                if isinstance(rule.notes, str):
                    trans.loc[ruleMatch, 'notes'] += rule.notes

            # Update rule data
            ruleMatches[ruleI] = ruleMatch.sum()
            ruleIds[ruleI] = ruleI

        return trans, ruleMatches, ruleIds, ruleErrors, ruleMoves, ruleMoveExp

    def connect(self, serial=False):
        """Connects transactions based on user-defined rules"""
        tStart = time.time()

        # Add columns to transactions to manage and report on matches
        self.trans['category'] = ''
        self.trans['owner'] = ''
        self.trans['rules'] = ''
        self.trans['specific'] = 0
        self.trans['notes'] = ''
        self.trans['dateOriginal'] = self.trans['date']
        self.trans['pnlExclude'] = False

        # Process rules
        rules = gsReadDataFrame('MatchRules', gs=self.gs)
        rules = rules.rename(
            columns={
                'Src Account': 'srcAccount',
                'Dst Account': 'dstAccount',
                'Rule': 'rule',
                'Filter': 'filter',
                'Owner': 'owner',
                'Category': 'category',
                'Notes': 'notes',
                'Type': 'type',  # Y if shared. UNMAP if used for unmapped items
                'RuleId': 'id',
                'Matches': 'matches',
            }
        )
        rules['matches'] = 0
        rules['moves'] = 0
        rules['moveExp'] = 0
        rules['errors'] = ''

        if serial:
            self.trans, ruleMatches, ruleIds, ruleErrors, ruleMoves, ruleMoveExp = self.connect_chunk(
                self.trans, rules, self.taxonomy
            )
            rules['matches'] = ruleMatches
            rules['errors'] = ruleErrors
            rules['moves'] = ruleMoves
            rules['id'] = ruleIds
            rules['moveExp'] = ruleMoveExp
        else:
            # Chunk, process, recombine. Use one less chunk than CPUs, since we are batching, and other stuff is going on too
            chunk_size = math.ceil(len(self.trans) / ((os.cpu_count() or 1) - 1))
            chunks = [self.trans.iloc[i : i + chunk_size] for i in range(0, len(self.trans), chunk_size)]

            futures = []
            trans_res = []

            with ProcessPoolExecutor() as executor:
                for chunk in chunks:
                    futures.append(executor.submit(Finance.connect_chunk, chunk, rules, self.taxonomy))

                for idx, future in enumerate(futures):
                    result, ruleMatches, ruleIds, ruleErrors, ruleMoves, ruleMoveExp = future.result()

                    trans_res.append(result)
                    rules['matches'] += ruleMatches
                    rules['errors'] += ruleErrors
                    rules['moves'] += ruleMoves
                    if not idx:
                        rules['id'] = ruleIds
                        rules['moveExp'] = ruleMoveExp

            self.trans = pd.concat(trans_res)

        # Verify number of moves per rule matching specification
        for rule in rules.itertuples():
            if rule.moves != rule.moveExp:
                if rule.moveExp == 1:
                    raise Exception(
                        'move rule can by default only be applied when only 1 transaction matches. Add #N prefix to apply to more'
                    )
                print(
                    f'ERROR: move rule specified a count of {rule.moveExp}, but {rule.moves} matches were found:\n'
                    + f'  RULE:    {Finance.fmtRule(rule)}\n'
                )

        # Write match info back to rules sheet
        gsWriteDataFrame(
            rules[['id', 'matches', 'errors']].rename(
                columns={'id': 'Rule Id', 'matches': 'Matches', 'errors': 'Errors'}
            ),
            'MatchRules',
            col=10,
            clear=False,
            gs=self.gs,
        )

        print(f'  Time: {int(time.time() - tStart)} s')

    def update(self):
        """Must run after connect(), as connect() can move transactions around, and update() calculates balances"""
        tStart = time.time()

        ############################################
        # Transaction Post
        ############################################

        for provider in self.providers:
            if hasattr(provider, 'updateTransactions'):
                provider.updateTransactions()

        ############################################
        # Balances
        ############################################

        # Calculate default balance, however only apply to accounts that do not already have a balance given by the provider
        self.trans.sort_values(by=['account', 'date'], inplace=True)
        self.trans['balance'] = self.trans.groupby('account')['cents'].cumsum()

        # Correct balances based on balance reference specified for each account
        for account in self.accounts.itertuples():
            if account.balanceRef not in ['', 'nan']:
                if account.balanceRef is None:
                    print(f'  missing account entry for "{account.Index}"')
                    continue

                refMatch = re.match(r'\$(-?[\d\.]+) @ (\d{4}-\d{2}-\d{2})', account.balanceRef)
                if refMatch is None:
                    print(f'  invalid balance ref: {account.balanceRef}')
                    continue

                refAmtStr, refDateStr = refMatch.groups()
                refDate = datetime.datetime.strptime(refDateStr, '%Y-%m-%d').date()
                beforeTrans = self.trans[
                    (self.trans['account'] == account.Index) & (self.trans['date'].dt.date <= refDate)
                ]
                beforeCents = (
                    beforeTrans['balance'].iloc[-1] if not beforeTrans.empty else 0
                )  # Assume current balance is 0 if no prior transactions

                self.trans.loc[self.trans['account'] == account.Index, 'balance'] += (
                    int(round(float(refAmtStr) * 100)) - beforeCents
                )

        ############################################
        # Misc
        ############################################

        ############################################
        # Reporting
        ############################################

        unmappedCount = (
            self.trans[self.trans['date'] >= self.cutoff]['category'].isin(['Exp/Misc/Unmapped', 'Inc/Unmapped']).sum()
        )
        if unmappedCount:
            print(f'  Unmapped: {unmappedCount:>4}')

        ############################################
        # DataSet init
        ############################################

        self.dsY = DataSet('YE', self)
        self.dsM = DataSet('ME', self)

        ############################################
        # Intermediate sheets
        ############################################

        self.renderPNLClean('cleanY', self.dsY)  # Simple name for use in formulas. Needed by budget

        ############################################
        # Other Systems
        ############################################

        self.budget = Budget(self)
        self.budget.load()

        self.dsY.finalize(self.budget)
        self.dsM.finalize(self.budget)

        print(f'  Time: {int(time.time() - tStart)} s')

    def render(self):
        """Output data"""
        tStart = time.time()

        self.renderTransactions('Trans[UM]', lambda d: d[d['category'].isin(['Exp/Misc/Unmapped', 'Inc/Unmapped'])])
        self.renderTransactions('Trans[All]')
        self.renderPNL('PnL[M]', self.dsM, '%Y-%m')
        self.renderPNL('PnL[Y]', self.dsY, '%Y')

        print(f'  Time: {int(time.time() - tStart)} s')

    def renderPNL(self, sheetName, ds, dateFmt):
        assert self.budget is not None  # nosec

        print(f'  {sheetName}')

        pnlLast = ds.pnl.columns[-1]
        mixPNL, mixGroups = dfMixColumns(
            ds.pnl,
            [ds.pnlBudgetDelta, ds.pnlBudget, ds.pnlBlank],
            [lambda c: 'Delta', lambda c: 'Budget*' if c == pnlLast and ds.freq == 'YE' else 'Budget', lambda c: ''],
            groupColOffset=1,
        )
        mixBS, _ = dfMixColumns(
            ds.bs,
            [ds.bsContrib, ds.bsOrganic, ds.bsRateReturn],
            [lambda c: 'Contrib', lambda c: 'Organic', lambda c: 'RR'],
            groupColOffset=1,
        )

        print('combine all MISC things in this table')
        if ds.freq == 'YE':
            mixTax, _ = dfMixColumns(
                ds.miscTax,
                [ds.miscTaxBlank, ds.miscTaxBlank, ds.miscTaxBlank],
                [lambda c: '', lambda c: '', lambda c: ''],
                groupColOffset=1,
            )
        else:
            mixTax = None

        fmtPNL, fmtPNLGroups = fmtDfToGridHierarchical(mixPNL, dateFmt)
        fmtBS, fmtBSGroups = fmtDfToGridHierarchical(mixBS, dateFmt)  # , self.bsAggregator(aggContext))
        if mixTax is not None:
            fmtTax, fmtTaxGroups = fmtDfToGridHierarchical(mixTax, dateFmt)  # , self.bsAggregator(aggContext))
        else:
            fmtTaxGroups = None
            fmtTax = None

        # Additional formatting by functions
        fmtGridApplyFmtFuncs(
            [fmtPNL],
            [
                # Apply good/bad coloring to 'delta' column
                lambda d, i, j: (
                    (
                        fmtBgGoodBad(
                            b
                            / abs(b)  # Sign of budget. I.e. positive budget: +1, negative budget: -1
                            * (
                                d[i][j][0] / b
                            )  # Percentage compared to budget. Negative if below budget, positive if above
                            * 5  # Multiply up. x5 results in full red on 20% above budget
                        ),
                    )
                    if i != 0
                    and d[0][j][0].startswith('Delta')
                    and (b := d[i][j + 1][0]) != 0
                    and not isinstance(b, str)
                    else None
                ),
            ],
        )

        fmtGridApplyFmtFuncs(
            [fmtPNL, fmtBS, *([fmtTax] if mixTax is not None else [])],
            [
                # Apply % formatting to RR column
                lambda d, i, j: (
                    (
                        (fmtPercent,)
                        if (d is fmtBS and d[0][j][0] == 'RR')
                        or (d is fmtTax and d[i][0][0] == 'Tax Rate' and d[0][j][0] != '')
                        else (fmtDollars,)
                    )
                    if i != 0 and not isinstance(d[i][j][0], str)
                    else None
                )
            ],
        )

        # Compose
        def fmtSpace(title):
            return [
                [
                    ['', *([''] * (len(fmtPNL[0]) - 1))],
                    [title, *([''] * (len(fmtPNL[0]) - 1))],
                ],
            ]

        out = fmtGridsAssemble(
            [
                # ['P&L']
                [fmtPNL],
                fmtSpace('Balances'),
                [fmtBS],
                *(
                    [
                        fmtSpace('Tax'),
                        [fmtTax],
                    ]
                    if mixTax is not None
                    else []
                ),
            ]
        )

        # Write
        sheet = self.gs.worksheet(sheetName)
        gsUpdateFormatted(out, sheet, gs=self.gs)
        sheet.freeze(1, 1)
        sheet.resize(len(out), len(out[0]))

        gsApplyRowGroups(
            sheet,
            [
                *fmtPNLGroups,
                *addGroupsOffset(fmtBSGroups, len(fmtPNL) + 2),
                *(addGroupsOffset(fmtTaxGroups, len(fmtPNL) + len(fmtBS) + 4) if mixTax is not None else []),
            ],
        )
        gsApplyColGroups(sheet, mixGroups)

    def renderPNLClean(self, sheetName, ds):
        output = pd.concat([ds.pnl, ds.bs]).reset_index()

        sheet = self.gs.worksheet(sheetName)
        gsWriteDataFrame(output, sheet, gs=self.gs, header='Do not manually update')
        sheet.freeze(2, 1)

    def renderTransactions(self, sheetName, filter=None):
        print(f'  {sheetName}')

        output = self.trans.copy()
        output = output[output['date'] >= self.cutoff]

        if filter is not None:
            output = filter(output)

        output.drop(columns='balance', inplace=True)

        output = output.rename(
            columns={
                'account': 'Account',
                'date': 'Date',
                'cents': 'Amount',
                'desc': 'Desc',
                'ref': 'Ref',
                'category': 'Category',
                'owner': 'Owner',
                'rules': 'Rules',
                'specific': 'Specific Count',
                'notes': 'Notes',
                'dateOriginal': 'Original Date',
            }
        )
        output['Amount'] = output['Amount'] / 100.0
        output['Year'] = output['Date'].dt.year
        output['Month'] = "'" + output['Date'].dt.strftime('%Y-%m')

        sheet = self.gs.worksheet(sheetName)
        gsWriteDataFrame(output, sheet, gs=self.gs, header='Do not manually update')
        sheet.freeze(2, 0)
        sheet.set_basic_filter(f'{sheetCoord(1, 2)}:{sheetCoord(sheet.col_count, sheet.row_count)}')

        # - TBD Output per-account timelines
        # - TBD: output stats to taxonomy sheet in same way as MatchRuels
