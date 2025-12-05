from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
from typing import List, Optional


from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
from typing import List, Optional
import requests
from bs4 import BeautifulSoup

class GDELTNewsFetcher:
    def __init__(self, project_id: str, service_account_json: Optional[str] = None):
        if service_account_json:
            credentials = service_account.Credentials.from_service_account_file(
                service_account_json
            )
            self.client = bigquery.Client(project=project_id, credentials=credentials)
        else:
            self.client = bigquery.Client(project=project_id)

    def get_articles(
        self,
        start_datetime: int,
        end_datetime: int,
        country_code: str,
        keywords: List[str],
        limit: int = 100,
        sources: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        
        # List of fields to search for keywords
        text_fields = [
            "V2Themes",
            "V2Persons",
            "V2Organizations",
            "AllNames",
            "V2Locations",
        ]

        # Build keyword filter for all fields
        keyword_clauses = []
        for field in text_fields:
            keyword_clauses.extend([f"{field} LIKE '%{k.upper()}%'" for k in keywords])

        # Join all clauses with OR
        keyword_filter = " OR ".join(keyword_clauses)

        # Source filter (domain names)
        source_filter = ""
        if sources:
            source_clauses = [f"DocumentIdentifier LIKE '%{src}%'" for src in sources]
            source_filter = " AND (" + " OR ".join(source_clauses) + ")"

        # Build final query (all columns, no limit)
        query = f"""
        SELECT
            *
        FROM
            `gdelt-bq.gdeltv2.gkg`
        WHERE
            DATE BETWEEN {start_datetime} AND {end_datetime}
            AND Locations LIKE '%{country_code}%'
            AND ({keyword_filter})
            {source_filter}
        """


        df = self.client.query(query).to_dataframe()

        # Extract article text
        urls = df['DocumentIdentifier']
        texts = []

        for url in urls:
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    paragraphs = soup.find_all('p')
                    article_text = " ".join(p.get_text() for p in paragraphs)
                    texts.append(article_text)
                else:
                    texts.append(None)
            except Exception:
                texts.append(None)

        df['text'] = texts

        # Return the important fields (DATE, URL, text, Themes, Tone)
        return df[['DATE', 'DocumentIdentifier', 'text', 'Themes', 'V2Tone']]


