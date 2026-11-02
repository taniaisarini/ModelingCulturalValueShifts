# Value-shift indicator report

Case: USA, F118 (justifiable: homosexuality), wave 5 (2006) -> wave 6 (2011).
Model never told the true direction while generating/judging indicators.

| Direction | Source | Indicator | Expected | Verdict | Reasoning |
|---|---|---|---|---|---|
| A | nyt | Increased frequency of New York Times articles with positive coverage of same-sex marriage legalization efforts. | increase | **confirm** | The frequency of New York Times articles with positive coverage of same-sex marriage legalization efforts nearly doubled from 2006 to 2011, indicating increased acceptance as predicted. |
| A | gdelt | More GDELT events categorized as 01 MAKE PUBLIC STATEMENT with positive tone supporting LGBTQ+ rights. | increase | **deny** | The proportion of positive MAKE PUBLIC STATEMENT events decreased by 9% and the average tone also declined from 5.64 to 5.31, indicating less positive support for LGBTQ+ rights rather than an increase. |
| A | gdelt | Increase in GDELT events tagged as 14 PROTEST with positive tone, representing pro-LGBTQ+ rights demonstrations. | increase | **confirm** | There is a clear 15% increase in the rate of positive-tone PROTEST events related to LGBTQ+ rights from 2006 to 2011, supporting the predicted increase in accepting attitudes. |
| A | nyt | More New York Times articles discussing anti-discrimination laws protecting LGBTQ+ individuals. | increase | **deny** | The number of articles discussing anti-discrimination laws protecting LGBTQ+ individuals decreased from 2 to 0, contradicting the predicted increase. |
| A | gdelt | Increase in GDELT events involving 05 ENGAGE IN DIPLOMATIC COOPERATION or 03 EXPRESS INTENT TO COOPERATE related to LGBTQ+ rights initiatives. | increase | **deny** | The proportion of relevant GDELT events slightly decreased by 1% from 2006 to 2011, contrary to the predicted increase in diplomatic cooperation related to LGBTQ+ rights. |
| B | nyt | Increase in New York Times articles expressing opposition or criticism of homosexuality or LGBTQ+ rights. | increase | **deny** | The data shows a clear decrease of 55% in articles expressing opposition or criticism of homosexuality from 2006 to 2011, contrary to the predicted increase. |
| B | gdelt | More GDELT events tagged as 11 DISAPPROVE or 12 REJECT with negative tone targeting LGBTQ+ rights or homosexuality acceptance. | increase | **deny** | The number of disapprove or reject events per 1,000 decreased by 13%, indicating less negative coverage rather than an increase as predicted. |
| B | gdelt | Increase in GDELT events categorized as 14 PROTEST with negative tone representing anti-LGBTQ+ demonstrations or rallies. | increase | **inconclusive** | Although the number of PROTEST events increased by 15%, the average tone remained positive and did not indicate an increase in negative anti-LGBTQ+ demonstrations specifically. |
| B | nyt | More New York Times articles covering legislative attempts to restrict LGBTQ+ rights or ban same-sex marriage. | increase | **confirm** | The number of relevant articles increased by 77% from 2006 to 2011, indicating more coverage of legislative attempts to restrict LGBTQ+ rights, consistent with the predicted increase. |
| B | gdelt | Increase in GDELT events involving 10 DEMAND or 17 COERCE with negative tone aimed at limiting LGBTQ+ rights or expression. | increase | **confirm** | The number of relevant events per 1,000 increased by 14% and the average tone became more negative, indicating increased negative actions aimed at limiting LGBTQ+ rights as predicted. |

## Ground truth (revealed only here, not shown to the model)
- mean wave 5 (2006): 4.503, mean wave 6 (2011): 5.397 (change +0.894, p=6.3e-10)
- true direction: **A**