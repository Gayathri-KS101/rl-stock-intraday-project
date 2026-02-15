import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build


class GoogleSheetsLogger:
    def __init__(self, run_name, sheet_name="RL_Training_Log"):
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]

        creds = Credentials.from_service_account_file(
            "credentials.json",
            scopes=scope
        )

        client = gspread.authorize(creds)
        self.spreadsheet = client.open(sheet_name)
        self.spreadsheet_id = self.spreadsheet.id
        self.service = build('sheets', 'v4', credentials=creds)

        # Create new worksheet per run
        try:
            self.sheet = self.spreadsheet.add_worksheet(
                title=run_name,
                rows="1000",
                cols="20"
            )
        except:
            self.sheet = self.spreadsheet.worksheet(run_name)

        # Add headers
        self.sheet.append_row([
            "Episode",
            "Total Reward",
            "Final Net Worth",
            "Percent Return (%)",
            "Number of Trades"
        ])
    def create_chart(self, num_rows):
        sheet_id = self.sheet.id

        requests = [{
            "addChart": {
                "chart": {
                    "spec": {
                        "title": "Episode vs Total Reward",
                        "basicChart": {
                            "chartType": "LINE",
                            "legendPosition": "BOTTOM_LEGEND",
                            "axis": [
                                {"position": "BOTTOM_AXIS", "title": "Episode"},
                                {"position": "LEFT_AXIS", "title": "Total Reward"}
                            ],
                            "domains": [{
                                "domain": {
                                    "sourceRange": {
                                        "sources": [{
                                            "sheetId": sheet_id,
                                            "startRowIndex": 1,
                                            "endRowIndex": num_rows + 1,
                                            "startColumnIndex": 0,
                                            "endColumnIndex": 1
                                        }]
                                    }
                                }
                            }],
                            "series": [{
                                "series": {
                                    "sourceRange": {
                                        "sources": [{
                                            "sheetId": sheet_id,
                                            "startRowIndex": 1,
                                            "endRowIndex": num_rows + 1,
                                            "startColumnIndex": 1,
                                            "endColumnIndex": 2
                                        }]
                                    }
                                }
                            }],
                            "headerCount": 1
                        }
                    },
                    "position": {
                        "overlayPosition": {
                            "anchorCell": {
                                "sheetId": sheet_id,
                                "rowIndex": 1,
                                "columnIndex": 6
                            },
                            "offsetXPixels": 20,
                            "offsetYPixels": 20
                        }
                    }
                }
            }
        }]

        body = {"requests": requests}
        self.service.spreadsheets().batchUpdate(
            spreadsheetId=self.spreadsheet_id,
            body=body
        ).execute()


    def log_episode(self, episode, reward, net_worth, percent_return, trades):
        self.sheet.append_row([
            episode,
            reward,
            net_worth,
            percent_return,
            trades
        ])
