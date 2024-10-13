# HomeFin

This is a personal finance app which helps you stay on top of spend and budgeting. It reads in CSV exports from banks and other entities, and composes a view of how much money is spent month by month and year by year across different categories. It also allows you to make a budget across these categories, and helps you track spend according to that.

Currently, the core database is a Google Sheet. The user fills in tabs to explain account setup and matching rules, and the app reads this along with CSVs and outputs data into several other tabs in the sheet.


## Instructions

### Prerequisites

1. Follow the prerequisites for [olib](https://github.com/magicl/olib)
2. Clone this repo
3. Run `olib/scripts/init.sh` to init the python env and install packages

### Google Sheets access

1. Create a Google Sheet to use with homefin
2. Create a Google Cloud account and a service account able to use the Google Sheets API
3. Export the authentication json file

### Environment setup

1. Create a file .env.local based on .env.local.example
2. Create a directory to house your records. Map this to records_root in .env.local

### Records directory

1. As an example, if you have a Truist account: Create a `truist` folder inside the `records_root`
2. Inside this, add a folder for each account you have in Truist. The folder name could be the last 4 digits of the account number
3. Inside this, add CSV expors from Truist

### Google sheets setup

Create the following tabs:
- PnL[Y]
- PnL[M]
- Trans[All]
- Trans[UM]
- MatchRules
- Taxonomy
- Accounts
- Assets
- MiscTran

Full explanation of the sheets is a later exercise.

### Run the app

To run the app, which will read in all CSVs, as well as rules from the google sheet, run this command:

```
./app.py run
```

The general flow is, e.g. on a monthly basis:
1. Download data from banks etc, and put it in the folders. The app will automatically de-duplicate
2. Run the app
3. Look at the Trans[UM] tab. This will show what transactions are unmapped
4. Add rules to MatchRules to map the unmapped transactions
5. Repeat from 2 until all transactions are mapped
6. Review financials

## Providers

In `./providers` you will find code built to deal with exports from different types of banks and other companies. For now, data export is manual, but this may change in the future. I am playing with using playwright to automate this.


### Downloading records from providers

Note: Some of these might be slightly outdated.


#### AMAZON

For each account

1. Log in
2. Click "Account & Lists" in top menu bar
3. Click "Request Your Information" under privacy
4. Select "Your Orders"
5. Click submit
6. Click confirmation link in email
7. Wait.. Can take 24 hours or more to get your data
8. Unzip, then copy folder to `{records_root}/amazon`

#### TRUIST

For each account

1. Log in
2. Go to Account details for given account
3. In the activity area, click the "download transactions" icon
4. Copy to `{records_root}/truist/{acct}`

#### FIDELITY

For each account

1. Log in
2. Go to Accounts & Trade -> Cash Management
3. Click {account} under Cedit Cards
4. Click Download Transactions
5. Copy to `{records_root}/fidelity/{acct}`

#### Capital One

For each account

1. Log in
2. Click into {account}
3. Click Download Transactions
4. Copy to `{records_root}/capitalone/{acct}`

#### Discover

1. "All activity & statements"
2. Search transactions
3. From 1/1/20 to current date
4. CSV

#### Chase

For each account

1. Log in
2. Click "See All Transactions"
3. Click download (Activity: "All transactions")
4. Copy to `{records_root}/chase/{acct}`

#### Schwab

For each account

1. Log in
2. Click "Investor Checking"
3. Click Export
4. Copy to `{records_root}/schwab/{acct}`

#### Paypal

For each account

1. Login
2. Go to "Activity"
3. Click the download icon
4. "See More reports"
5. Select "balance affecting" and date-range. Click "create report"

#### Venmo

For each account

1. Log in
2. Click Statements
3. Download manually for each month
4. Copy to `{records_root}/venmo/{acct}`



## Internals

### Architecture

There are two types of accounts in the system
1) Accounts that have full transaction histories and that are accurately represented by these. These are typically liquid accounts such as bank accounts, credit cards, etc. and are used for spending
2) Investment accounts which don't have full transaction histories, and where putting money in does not necessarily retain the money

Accounts of type (1) all have their own transaction logs, and are expected to have matching transactions.. If money is taken out, there is expected to be a transaction inside the account, and a matching transaction
in another account where the money went to/from. Accounts of type (2) do not have internal transactions, just outside. Their balances are governed by the assets sheet.
